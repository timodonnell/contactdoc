# AFDB ContactDoc v1 — Design Spec (agent-implementable)

## 0) Goal
Build a pipeline that:
1) selects AlphaFold DB (AFDB) entries using BigQuery metadata
2) downloads AFDB v4 mmCIFs from GCS
3) generates LLM training documents with:
   - a 3-letter residue-token sequence
   - a list of residue-pair contacts encoded as (pos_i, pos_j, atom_i, atom_j)
4) enforces leakage-resistant train/val/test splits using precomputed cluster IDs (Foldseek AFDB50 by default)
5) writes sharded, compressed outputs plus JSONL metadata and error logs (Modal-friendly)

The training document format:

<begin_sequence>
<MET> <LYS> <PHE> <CYS> ...
<begin_contacts>
<p1> <p57> <CB> <OD1>
...
<end_contacts>
<end>

Contacts are:
- heavy atoms only (no hydrogens)
- exclude same residue and adjacent residues (|i-j| > 1)
- for each residue pair, emit at most 1 contact using the closest heavy-atom pair
- contacts sorted by decreasing sequence separation (j-i), ties by N→C (lower i, then lower j)
- positions always output with i < j (ascending residue order)
- cap maximum contact lines per doc (prefix after sorting)

Additionally:
- skip fragments
- filter proteins by global mean pLDDT threshold (metadata)
- filter contacts by per-residue pLDDT threshold (from mmCIF B-factors)
- deterministic tie-breaking for equal-distance atom pairs

---

## 1) Definitions

### 1.1 Canonical residue tokens
Emit residues as 3-letter uppercase inside angle brackets:
<ALA> <ARG> <ASN> <ASP> <CYS> <GLN> <GLU> <GLY> <HIS> <ILE>
<LEU> <LYS> <MET> <PHE> <PRO> <SER> <THR> <TRP> <TYR> <VAL>

Policy for noncanonical residues:
- map to <UNK>

### 1.2 Position tokens
Local 1-based indices:
<p1> ... <pL>

### 1.3 Atom tokens
Atom names from mmCIF, stripped of whitespace, inside angle brackets:
<CA> <CB> <OD1> <NH1> <SG> etc.

### 1.4 Confidence (pLDDT)
- Global mean pLDDT: BigQuery metadata field `globalMetricValue`
- Per-residue pLDDT: computed from mmCIF B-factors (atom.b_iso) averaged across heavy atoms of that residue

### 1.5 Contact cutoff
Default heavy-atom distance cutoff: 4.0 Å (configurable)

---

## 2) Inputs and outputs

### 2.1 Inputs
A YAML config (see §3)
Precomputed cluster mapping files (Foldseek / AFDB50) (see §6)
Google BigQuery access (public dataset)
Google Cloud Storage access (public AFDB bucket)

### 2.2 Outputs
Sharded compressed training text + sharded compressed JSONL metadata + sharded compressed JSONL errors.

Recommended layout:
gs://YOUR_BUCKET/contactdoc/v1/config_sha=ABC123/
  split=train/
    shard=000000.txt.gz
    shard=000000.metadata.jsonl.gz
    shard=000000.errors.jsonl.gz
    shard=000001.txt.gz
    shard=000001.metadata.jsonl.gz
  split=val/
    shard=000000.txt.gz
    shard=000000.metadata.jsonl.gz
  split=test/
    shard=000000.txt.gz
    shard=000000.metadata.jsonl.gz

Text shard format:
- concatenated documents separated by `<end_of_document>` markers (each doc includes <end> terminal)

Metadata shard format:
- 1 JSON object per emitted doc

Errors shard format:
- 1 JSON object per skipped/failed entry

---

## 3) Configuration (YAML)

Example config:

afdb_version: 4
bigquery_table: "bigquery-public-data.deepmind_alphafold.metadata"
gcs_bucket_prefix: "gs://public-datasets-deepmind-alphafold-v4/"
output_prefix: "gs://YOUR_BUCKET/contactdoc/v1/"

filters:
  skip_fragments: true
  global_mean_plddt_min: 70.0
  residue_plddt_min: 70.0
  max_seq_len: 2048
  require_single_chain: true
  canonical_residue_policy: "skip_entry"  # skip_entry | map_to_unk

contacts:
  cutoff_angstrom: 4.0
  exclude_adjacent_residues: true         # must be true (|i-j|>1)
  heavy_atoms_only: true                  # must be true
  max_contacts_per_doc: 2048
  tie_break: "lex_atom_names"             # must be lexicographic on (atom_i, atom_j)

splits:
  mode: "afdb50"                          # afdb50 | foldseek_structural
  seed: "contactdoc-v1"
  train_frac: 0.98
  val_frac: 0.01
  test_frac: 0.01

cluster_files:
  afdb50_rep_mem_tsv_gz: "/path/to/AFDB50_rep_mem.tsv.gz"
  structural_rep_mem_tsv_gz: "/path/to/AFDB_struct_rep_mem.tsv.gz"  # optional

parallelism:
  shard_size_entries: 2000
  num_workers_local: 16                   # if running locally (optional)
  use_modal: true

---

## 4) Selection logic (BigQuery → manifest)

### 4.1 Required selection criteria
Include entries only if:
- latestVersion == afdb_version (4)
- if skip_fragments:
  - uniprotStart == 1
  - uniprotEnd == LENGTH(uniprotSequence)
- globalMetricValue >= global_mean_plddt_min
- LENGTH(uniprotSequence) <= max_seq_len

Required BigQuery fields returned into manifest:
- entryId
- uniprotAccession
- taxId
- organismScientificName
- latestVersion
- globalMetricValue
- uniprotStart
- uniprotEnd
- uniprotSequence (or at least its length; optional but recommended for validation)

### 4.2 mmCIF URI construction
gcs_uri = f"{gcs_bucket_prefix}{entryId}-model_v4.cif"

### 4.3 Manifest enrichments
After selection, attach:
- split_cluster_id (from cluster mapping; fallback to entryId if missing)
- split (train/val/test from deterministic cluster hashing; see §6)

Write manifest shards (e.g., JSONL or parquet), each containing `shard_size_entries` rows.

Manifest row schema (minimum):
{
  "entryId": str,
  "gcs_uri": str,
  "uniprotAccession": str,
  "taxId": int,
  "organismScientificName": str,
  "latestVersion": int,
  "globalMetricValue": float,
  "uniprotStart": int,
  "uniprotEnd": int,
  "seq_len": int,
  "split_cluster_id": str,
  "split": "train"|"val"|"test"
}

---

## 5) mmCIF parsing and per-residue pLDDT

### 5.1 Parser requirements
Use `gemmi` for mmCIF parsing and neighbor/contact search.
Avoid Biopython here because Gemmi provides ContactSearch with residue-adjacency filtering and faster contact enumeration.

### 5.2 Chain policy (v1)
- Identify polymer chains in model 0.
- If require_single_chain:
  - if not exactly one polymer chain found → skip entry.
- Build residue list R[1..L] for that chain in sequence order.

### 5.3 Residue tokenization
For each residue in R:
- residue.name must be one of canonical 20
- emit as <RESNAME>
If any residue violates policy:
- if canonical_residue_policy == skip_entry: skip
- else map to <UNK>

### 5.4 Per-residue pLDDT
For residue i:
- collect all heavy atoms (exclude hydrogens)
- residue_plddt[i] = mean(atom.b_iso)
If residue has zero heavy atoms:
- residue_plddt[i] = -inf

Also compute:
- residues_passing_plddt = count(i where residue_plddt[i] >= residue_plddt_min)

---

## 6) Leakage-resistant splits via clusters

### 6.1 Cluster mapping sources
Use precomputed cluster mapping TSV.GZ of the form:
rep_id \t member_id

Modes:
- afdb50: sequence similarity clusters (preferred default)
- foldseek_structural: structural clusters (optional stricter mode)

Implementation detail:
- Build a lookup: member_id -> rep_id
- If member_id not found: rep_id = member_id (singleton cluster)

split_cluster_id = rep_id

### 6.2 Deterministic cluster → split assignment
Given seed S and cluster_id C:
- h = sha1(S + "::" + C)
- interpret first 8 bytes as uint64, u = uint64 / 2^64 in [0,1)
Assign:
- train if u < train_frac
- val if train_frac <= u < train_frac+val_frac
- test otherwise

All members of same cluster_id share the same split.

---

## 7) Contact definition and emission

### 7.1 Requirements (must match exactly)
- heavy atoms only
- eligible residue pair iff i < j and |i-j| > 1
- define d_min(i,j) as the minimum heavy-atom distance between residues i and j
- residue pair contact exists iff d_min(i,j) <= cutoff
- emit at most 1 contact per residue pair using atoms achieving d_min
- tie-break equal d_min by lexicographic (atom_name_i, atom_name_j), names stripped of whitespace
- contact line always output in ascending residue order (i<j)

### 7.2 Efficiency requirement
Must NOT do O(L^2 * atoms^2) loops.
Instead:
- use Gemmi NeighborSearch + ContactSearch to enumerate atom pairs within cutoff
- aggregate to the best contact per residue pair

### 7.3 Gemmi-based algorithm (required)
Inputs:
- gemmi.Model model (single chain)
- cutoff = cutoff_angstrom

Procedure:
1) ns = gemmi.NeighborSearch(model, cell, cutoff).populate(include_h=False)
   - For AFDB, set cell to empty UnitCell() if you want non-crystal behavior; mmCIF from AFDB is monomer, so either is fine as long as you do NOT add symmetry mates.
2) cs = gemmi.ContactSearch(cutoff)
3) cs.ignore = gemmi.ContactSearch.Ignore.AdjacentResidues
   - This excludes same and adjacent residues at the atom-contact enumeration stage.
4) atom_contacts = cs.find_contacts(ns)
5) For each atom-contact r in atom_contacts:
   - identify residue objects for both atoms
   - map residue objects to indices i and j in [1..L]
   - normalize (i,j) so i<j
   - update best_contact[(i,j)] using:
     - smaller distance wins
     - if equal distance, lexicographically smaller (atom_i, atom_j) wins

Data structure:
best_contact[(i,j)] = (dist, atom_i_name, atom_j_name)

### 7.4 Confidence filter at contact level
A residue-pair contact (i,j) is eligible for final emission only if:
- residue_plddt[i] >= residue_plddt_min
- residue_plddt[j] >= residue_plddt_min

### 7.5 Sorting rule + truncation
After filtering:
- sort contacts by:
  1) (j-i) descending
  2) i ascending
  3) j ascending
- truncate to first max_contacts_per_doc contacts (prefix of sorted list)

---

## 8) Serialization

### 8.1 Document serialization
Write exactly:

<TASK_TOKEN>
<begin_sequence>
<RES_1> <RES_2> ... <RES_L>
<begin_contacts>
<p{i1}> <p{j1}> <ATOM_1A> <ATOM_1B>
...
<end_contacts>
<end>

Rules:
- each document starts with a task token identifying the generation scheme (e.g. `<deterministic-positives-only>`)
- sequence is a single line with residue tokens separated by spaces
- each contact is its own line, exactly 4 tokens
- end markers exactly as above
- documents within a shard are separated by `<end_of_document>`

### 8.2 Sidecar metadata JSONL (1 record per emitted doc)
Required fields:
- entryId
- uniprotAccession
- taxId
- organismScientificName
- latestVersion
- globalMetricValue_mean_pLDDT
- uniprotStart
- uniprotEnd
- seq_len
- global_mean_plddt_min
- residue_plddt_min
- contact_cutoff_angstrom
- max_contacts_per_doc
- contacts_found_pre_confidence_filter
- contacts_emitted
- residues_passing_plddt
- split
- split_cluster_mode
- split_cluster_id
- source_cif_gcs_uri
- sha1_of_document_text

### 8.3 Error records JSONL
Write an error record for any skipped/failed entry:
- entryId
- gcs_uri (if known)
- reason (enum-ish string)
- exception (string, optional)
- minimal context: globalMetricValue, seq_len, etc. (if available)

---

## 9) Modal-friendly parallelization plan

### 9.1 Pipeline stages
Stage A: Manifest build (run once)
- BigQuery select -> list of entryIds + metadata
- attach cluster_id and split
- write manifest shards (N shards)

Stage B: Shard processing (parallel)
- one worker per manifest shard
- worker streams mmCIFs from GCS
- worker writes sharded outputs:
  - split=train: shard_k.txt.gz, shard_k.metadata.jsonl.gz, shard_k.errors.jsonl.gz
  - split=val: ...
  - split=test: ...

Stage C: Optional compaction
- merge shards into fewer/larger files

### 9.2 Critical constraints
- Do NOT write one output file per protein
- Do NOT have multiple workers append to the same output object/file
- Shard outputs must be deterministic and idempotent (rerun shard safely overwrites)

### 9.3 Shard sizing
Default shard_size_entries = 2000
Tune based on:
- CIF size
- average output size
- retry cost

---

## 10) Golden example (format + ordering only)

Example (not claiming physical correctness of atom choices):

<begin_sequence>
<MET> <LYS> <PHE> <CYS> <ASP> <TYR> <GLY> <LEU>
<begin_contacts>
<p1> <p8> <SD> <CD1>
<p1> <p7> <CG> <CA>
<p2> <p8> <NZ> <O>
<p1> <p6> <CE> <OH>
<p2> <p7> <CE> <CA>
<p3> <p8> <CZ> <CD2>
<p4> <p8> <SG> <CG>
<p2> <p5> <NZ> <OD1>
<p3> <p6> <CD1> <OH>
<end_contacts>
<end>

Ordering justification:
Sorted by (j-i) desc, ties by i asc, then j asc.

---

## 11) Skip conditions (must log)
Skip entry if:
- fragment (uniprotStart != 1 OR uniprotEnd != len(uniprotSequence)) [if configured]
- global mean pLDDT < global_mean_plddt_min
- sequence length > max_seq_len
- chain policy fails (not exactly one polymer chain when required)
- noncanonical residue encountered and policy is skip_entry
- mmCIF parse fails
- no residues extracted
- after filtering, no contacts remain (optional: either skip or emit empty-contact doc; v1 default: skip)

---

## 12) Determinism requirements
Given same inputs (config + mmCIF + cluster mapping), output must be identical across runs:
- tie-breaking must be explicit
- sorting must be explicit
- avoid nondeterministic iteration order influencing output
- ensure stable mapping from Gemmi residue identity → index

---

## 13) Acceptance tests (must pass)
1) Fragment skip:
   - entry with uniprotStart!=1 or uniprotEnd!=len(seq) is skipped
2) Global pLDDT:
   - entries below threshold skipped
3) Per-residue pLDDT contact filter:
   - contacts where either endpoint residue pLDDT < residue_plddt_min must NOT be emitted
4) Adjacent exclusion:
   - no emitted contact with |i-j|<=1
5) Hydrogens excluded:
   - no hydrogen atoms appear in emitted contacts
6) One contact per residue pair:
   - no duplicate (i,j)
7) Tie-break determinism:
   - equal-distance candidates choose lexicographically smallest atom-name pair
8) Ordering:
   - contacts sorted by (j-i) desc, then i asc, then j asc
9) Truncation:
   - emitted list equals prefix of fully sorted list, length <= max_contacts_per_doc
10) Split determinism:
   - all members of same cluster_id share split; split stable across runs
11) Golden example formatting:
   - exact markers and token structure; each contact line has 4 tokens

---

## 14) Suggested repo structure
repo/
  contactdoc/
    config/default.yaml
    scripts/
      bq_make_manifest.py
      build_cluster_kv.py
      process_manifest_shard.py
      compact_shards.py
      modal_app.py                  # optional
    contactdoc/
      afdb_query.py
      manifest.py
      clusters.py
      splits.py
      cif_parse.py
      plddt.py
      contacts.py
      serialize.py
      io.py
      utils.py
    tests/
      test_fragment_skip.py
      test_global_plddt.py
      test_residue_plddt_filter.py
      test_contact_tiebreak.py
      test_ordering_truncation.py
      test_split_determinism.py
      fixtures/tiny_example.cif

---

## 15) Implementation checklist (agent execution order)
1) Parse YAML config; compute config_sha
2) Build cluster lookup db: member_id -> rep_id
3) BigQuery query -> selected entries -> write manifest shards
4) For each manifest shard:
   - for each entry:
     - read CIF from GCS
     - parse with Gemmi
     - enforce chain policy
     - build residue list + index map
     - extract sequence tokens; enforce residue policy
     - compute per-residue pLDDT
     - compute best residue contacts via ContactSearch + aggregation
     - apply residue pLDDT filter
     - sort + truncate
     - serialize doc
     - write doc + metadata (or log skip/error)
   - write shard outputs (train/val/test) for this shard
5) Optional: compaction

DONE
