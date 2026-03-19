# random-3-bins Document Generation Scheme

## Overview

This scheme generates training documents that include a mix of true contacts, near-miss pairs, distant pairs, and intentionally false contacts with subsequent corrections. The goal is to train a model that can reason about protein contacts in the presence of noise and learn to self-correct.

Each document describes one protein structure. The output is a tokenized sequence containing the amino acid sequence, a set of 6-token groups describing residue pair interactions (with distance bins and correction/non-correction markers), a binned global pLDDT confidence token, and occasional corrections of previously stated false information.

## Token Format

Each contact entry is a 6-token group:
```
<non-correction> <p{i}> <p{j}> <{atom_i}> <{atom_j}> <{distance_bin}>
```
or for corrections:
```
<correction> <p{i}> <p{j}> <{atom_i}> <{atom_j}> <{distance_bin}>
```

- `<non-correction>` — first time this residue pair `(i, j)` appears in the document.
- `<correction>` — this residue pair was seen before; this entry updates/corrects the previous value.

Distance bin tokens:
- `<bin_lt4>` — closest heavy-atom distance < 4 Å
- `<bin_4_12>` — closest heavy-atom distance 4–12 Å
- `<bin_gt12>` — closest heavy-atom distance > 12 Å

pLDDT bin tokens (based on global pLDDT):
- `<plddt_lt70>`
- `<plddt_70_75>`
- `<plddt_75_80>`
- `<plddt_80_85>`
- `<plddt_85_90>`
- `<plddt_90_95>`
- `<plddt_95_100>`

## Document Structure

```
<random-3-bins>
<begin_sequence>
<MET> <LYS> <PHE> <CYS> <ASP> <TYR> <GLY> <LEU>
<begin_contacts>
<non-correction> <p1> <p5> <SD> <CD1> <bin_lt4>
<non-correction> <p3> <p7> <CA> <CB> <bin_4_12>       ← false contact (true bin is bin_lt4)
<non-correction> <p2> <p6> <NZ> <OH> <bin_gt12>
<non-correction> <p4> <p8> <CB> <O> <bin_lt4>
<correction> <p3> <p7> <CG> <CB> <bin_lt4>            ← corrects p3,p7 to correct bin
<plddt_80_85>
<non-correction> <p1> <p6> <CE> <OH> <bin_lt4>
<end_contacts>
<end>
```

Or with pLDDT at end (50% of documents):

```
<random-3-bins>
<begin_sequence>
<MET> <LYS> <PHE> ...
<begin_contacts>
<non-correction> <p1> <p5> <SD> <CD1> <bin_lt4>
...
<end_contacts>
<plddt_80_85>
<end>
```

### Rules for special tokens

- **`<non-correction>` / `<correction>`**: Every 6-token group starts with one of these. `<non-correction>` is used the first time a residue pair appears. `<correction>` is used for any subsequent mention of the same residue pair.
- **pLDDT token**: Exactly one per document. In 50% of documents, placed just before `<end>` (after `<end_contacts>`). In the other 50%, placed at a uniformly random position between complete 6-token groups in the contacts section.

Target document length: ~8000 tokens maximum.

## Document Generation Pipeline

All randomness is seeded deterministically from `SHA1(entry_id)` so that the same structure always produces the same document.

### Step 1: Parse and Extract

1. Parse mmCIF from string using Gemmi.
2. Extract residues (1-based indexing, non-canonical residues mapped to UNK).
3. Apply per-residue pLDDT filter (≥ 70.0): both residues in a pair must pass.
4. Skip adjacent residues (|i - j| ≤ 1).

### Step 2: Compute Pairwise Distances

Use Gemmi's `ContactSearch` at a 4 Å cutoff to find bin-1 contacts (true contacts). For bin-2 (4–12 Å) and bin-3 (> 12 Å), sample random residue pairs and compute their closest heavy-atom distance using cached atom positions. This avoids the expensive large-cutoff Gemmi search.

For each pair, record the closest heavy-atom pair and its distance.

### Step 3: Long-Range Contact Upsampling

True contacts (bin 1, < 4 Å) between residues far apart in primary sequence are biologically more informative. We upweight these as follows:

Let `sep = |i - j|` be the sequence separation. Define an upsampling weight:

```
weight(sep) = 1.0                  if sep < 8
weight(sep) = 1.0 + log2(sep / 8)  if sep ≥ 8
```

Examples:
- sep = 4 → weight 1.0
- sep = 8 → weight 1.0
- sep = 16 → weight 2.0
- sep = 32 → weight 3.0
- sep = 64 → weight 4.0
- sep = 128 → weight 5.0

When the token budget requires us to subsample bin-1 contacts, we use these weights for weighted sampling without replacement rather than uniform sampling.

If the total number of bin-1 contacts (after weighting) fits within the budget, all are included (no subsampling needed).

### Step 4: Budget Calculation and Sampling

The token budget is 8000 tokens. There are no `<newline>` tokens — all tokens flow continuously.

Fixed overhead:
- Task token: 1 token (`<random-3-bins>`)
- Sequence section: `num_residues + 1` tokens (`<begin_sequence>` + residue tokens)
- Contacts framing: 2 tokens (`<begin_contacts>` `<end_contacts>`)
- End: 1 token (`<end>`)
- pLDDT token: 1 token

Each contact 6-token group: 6 tokens (correction marker + 5-tuple).

Available budget for contacts:
```
contact_budget = floor((8000 - 6 - num_residues) / 6)
```

#### Contact allocation:

Let `C_raw` = total number of bin-1 contacts (before any subsampling).

Target counts (before budget constraint):
- Bin 1: `C_raw` (all true contacts, subject to budget)
- Bin 2: `round(C_raw * 0.2)` (sampled uniformly from bin-2 pairs)
- Bin 3: `round(C_raw * 0.1)` (sampled uniformly from bin-3 pairs)
- False contacts: `Poisson(λ=2.0)` (see Step 5)
- Corrections: up to `2 × N_false` worst case (see Step 6)

Total target contacts = `C_raw + round(C_raw * 0.2) + round(C_raw * 0.1) + N_false + N_corrections`.

If total exceeds `contact_budget`:
1. First reduce bin-1 contacts using weighted sampling (Step 3 weights). Set `C = contact_budget - round(C * 0.2 budget share) - round(C * 0.1 budget share) - N_false - N_corrections`. Solve for the largest C that fits.
2. Specifically: let `total_ratio = 1.0 + 0.2 + 0.1 = 1.3`. Then `C = floor(contact_budget * (1.0 / 1.3))` minus space for false+corrections. Bin 2 gets `round(C * 0.2)`, Bin 3 gets `round(C * 0.1)`. Adjust as needed to fit exactly.
3. If C < C_raw, subsample bin-1 using the long-range weights.

### Step 5: False Contact Injection

Sample `N_false ~ Poisson(λ=2.0)` false contacts.

Each false contact is constructed as follows:
1. Sample a residue pair `(i, j)` uniformly from all eligible pairs (|i - j| > 1, both pLDDT ≥ 70).
2. For each residue, pick a random heavy atom from the atoms actually present in that residue in the structure.
3. Sample a distance bin from a categorical distribution matching the bin proportions in the real contact set from Step 4. That is, if the Step 4 set has 100 bin-1, 20 bin-2, 10 bin-3 contacts, then P(bin_lt4) = 100/130, P(bin_4_12) = 20/130, P(bin_gt12) = 10/130.
4. Record whether this false contact happens to be correct (i.e., the sampled bin matches the true bin for this residue pair with these atoms).

### Step 6: Correction Mechanism

Each false contact will eventually be corrected later in the document. Corrections are interleaved among the real contacts.

After constructing the shuffled contact list (Step 7), we iterate through positions to decide where to insert corrections. At position `k` out of `T` total contact slots:

```
P(emit a correction at position k) = 1 - (1 - k/T)^F
```

where `F` is the current number of uncorrected false contacts. This gives:
- Low probability early in the document
- Increasing probability as we approach the end
- Higher probability when more false contacts are outstanding

At position `T` (end), all remaining uncorrected false contacts are flushed.

A correction re-specifies the same residue pair `(i, j)` with:
- The closest heavy-atom pair for that residue pair and the correct distance bin.
- **Exception**: with probability `correction_resample_prob` (default 1%), the correction instead samples a new random bin (same procedure as Step 5.3). This means the "correction" may itself be wrong, and the false contact persists. It may be corrected again later, or may remain uncorrected at document end (in which case a final true correction is appended).

All remaining uncorrected false contacts are corrected (truly) at the end, before `<end_contacts>`.

Correction entries are prefixed with `<correction>`. All other entries (first appearance of a residue pair) are prefixed with `<non-correction>`.

### Step 7: Shuffling and Assembly

1. Combine all contacts from Step 4 (bin 1, 2, 3 samples) into a single list.
2. Shuffle uniformly.
3. Iterate through the shuffled list, inserting corrections per Step 6 at each position.
4. Flip a coin (50/50): if heads, the pLDDT token goes just before `<end>` (after `<end_contacts>`). If tails, insert it at a uniformly random position between complete 6-token groups in the contacts section.
5. Serialize to text.

### Step 8: Atom Resampling (1% per contact)

For each contact in the final document (from Steps 4 and 6), with probability `atom_resample_prob` (default 1%):
1. For each of the two residues, uniformly sample a heavy atom from the atoms actually present in that residue.
2. Compute the actual distance between these two specific atoms.
3. Assign the correct distance bin based on this computed distance.
4. Replace the atom names and distance bin in the output.

This applies to all contacts (bin 1, 2, and 3) but NOT to false contacts (Step 5), which already have randomly sampled atoms. It DOES apply to corrections.

**Special case for bin-3 pairs**: Since we did not compute actual distances for bin-3 pairs, when atom resampling triggers for a bin-3 contact, we must compute the distance between the two resampled atoms using their 3D coordinates from the Gemmi structure.

## Parameters (configurable)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `distance_bins` | `[4.0, 12.0]` | Bin edge thresholds in Å |
| `bin_sample_rates` | `[1.0, 0.2, 0.1]` | Sampling rate per bin (relative to C) |
| `false_contact_lambda` | `2.0` | Poisson λ for number of false contacts |
| `correction_resample_prob` | `0.01` | P(correction is itself wrong) |
| `atom_resample_prob` | `0.01` | P(resampling atoms for a contact) |
| `max_tokens` | `8000` | Maximum document length in tokens |
| `long_range_threshold` | `8` | Sequence separation threshold for upsampling |
| `residue_plddt_min` | `70.0` | Per-residue pLDDT filter |
| `plddt_bin_edges` | `[70, 75, 80, 85, 90, 95]` | Global pLDDT bin boundaries (percentages) |

## New Tokens Required

Added to `tokenizer.py`:

**Task token:**
- `<random-3-bins>`

**Correction markers:**
- `<correction>`
- `<non-correction>`

**Distance bin tokens:**
- `<bin_lt4>`
- `<bin_4_12>`
- `<bin_gt12>`

**pLDDT bin tokens:**
- `<plddt_lt70>`
- `<plddt_70_75>`
- `<plddt_75_80>`
- `<plddt_80_85>`
- `<plddt_85_90>`
- `<plddt_90_95>`
- `<plddt_95_100>`

## Files to Create/Modify

1. **Create** `contactdoc/generators/random_3_bins.py` — main generator
2. **Modify** `contactdoc/generators/__init__.py` — register new generator
3. **Modify** `contactdoc/tokenizer.py` — add new tokens
4. **Modify** `contactdoc/config.py` — add `Random3BinsConfig` dataclass
5. **Create/modify** `contactdoc/contacts.py` — add helper functions
6. **Create** `tests/test_random_3_bins.py` — tests
