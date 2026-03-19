"""Random-3-bins document generator.

Generates documents with distance-binned 5-tuples, false contact injection,
corrections, long-range upsampling, and pLDDT bin tokens.
See docs/random-3-bins-scheme.md for full specification.
"""

import hashlib
import math
import random
from dataclasses import dataclass

from ..cif_parse import ParseResult
from ..contacts import (
    Contact,
    _assign_bin,
    compute_atom_distance,
    compute_binned_contacts,
    get_residue_heavy_atoms,
)
from .base import DocumentGenerator, GeneratorResult


@dataclass
class BinnedContact:
    """A contact with its assigned distance bin."""
    i: int
    j: int
    atom_i: str
    atom_j: str
    bin_idx: int
    is_false: bool = False  # True if this is an injected false contact
    is_correction: bool = False  # True if this corrects a previous false


def _bin_token(bin_idx: int, bin_edges: list[float]) -> str:
    """Convert bin index to token string."""
    if bin_idx == 0:
        return f"bin_lt{int(bin_edges[0])}"
    elif bin_idx < len(bin_edges):
        return f"bin_{int(bin_edges[bin_idx - 1])}_{int(bin_edges[bin_idx])}"
    else:
        return f"bin_gt{int(bin_edges[-1])}"


def _plddt_bin_token(global_plddt: float, bin_edges: list[float]) -> str:
    """Convert global pLDDT to a bin token string."""
    # pLDDT is 0-100 scale, bin_edges are in percentage points
    for i, edge in enumerate(bin_edges):
        if global_plddt < edge:
            if i == 0:
                return f"plddt_lt{int(edge)}"
            else:
                return f"plddt_{int(bin_edges[i-1])}_{int(edge)}"
    # Above all edges
    return f"plddt_{int(bin_edges[-1])}_100"


def _long_range_weight(sep: int, threshold: int) -> float:
    """Compute upsampling weight for a contact based on sequence separation."""
    if sep < threshold:
        return 1.0
    return 1.0 + math.log2(sep / threshold)


def _weighted_sample_without_replacement(
    items: list, weights: list[float], k: int, rng: random.Random,
) -> list:
    """Weighted sampling without replacement using repeated draws."""
    if k >= len(items):
        return list(items)
    # Use Efraimidis-Spirakis algorithm: assign key = u^(1/w), take top k
    keys = []
    for item, w in zip(items, weights):
        u = rng.random()
        if u == 0:
            u = 1e-300
        key = u ** (1.0 / max(w, 1e-12))
        keys.append((key, item))
    keys.sort(reverse=True)
    return [item for _, item in keys[:k]]


class Random3Bins(DocumentGenerator):

    @property
    def name(self) -> str:
        return "random-3-bins"

    def generate(self, parse_result, cfg):
        r3b = cfg.random_3_bins
        bin_edges = r3b.distance_bins
        bin_rates = r3b.bin_sample_rates
        plddt_min = cfg.filters.residue_plddt_min

        # Deterministic seed from entry_id
        entry_id = parse_result.structure.name or "unknown"
        seed = int(hashlib.sha1(entry_id.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)

        # Step 1-2: compute binned contacts
        bins = compute_binned_contacts(parse_result, bin_edges, plddt_min)
        eligible_residues = bins.pop("_eligible_residues")
        pairs_within_cutoff = bins.pop("_pairs_within_cutoff")

        bin1_contacts = bins[0]  # < 4 Å
        bin2_contacts = bins[1]  # 4-12 Å
        # bin 2 (index 2) is > 12 Å, sampled lazily

        if not bin1_contacts:
            return "no_contacts_in_bin1"

        C_raw = len(bin1_contacts)

        # Step 3-4: budget calculation
        num_residues = len(parse_result.residues)
        fixed_overhead = 6 + num_residues  # task + begin_seq + residues + begin_contacts + end_contacts + end + plddt
        tokens_per_contact = 5
        contact_budget = (r3b.max_tokens - fixed_overhead) // tokens_per_contact

        if contact_budget <= 0:
            return "sequence_too_long_for_budget"

        # Sample false contacts count
        n_false = _sample_poisson(rng, r3b.false_contact_lambda)
        # Each false contact needs at least one correction. A bad resampled
        # correction (1% chance) adds an extra entry, so budget for worst case.
        n_corrections = n_false * 2

        # Target bin counts
        n_bin2_target = round(C_raw * bin_rates[1])
        n_bin3_target = round(C_raw * bin_rates[2])

        # Total target (before budget constraint)
        total_target = C_raw + n_bin2_target + n_bin3_target + n_false + n_corrections

        # If over budget, scale down bin1 (and proportionally bin2/bin3)
        if total_target > contact_budget:
            available = contact_budget - n_false - n_corrections
            if available <= 0:
                available = max(contact_budget, 1)
                n_false = 0
                n_corrections = 0
            total_ratio = bin_rates[0] + bin_rates[1] + bin_rates[2]
            C = int(available * bin_rates[0] / total_ratio)
            n_bin2_target = round(C * bin_rates[1])
            n_bin3_target = round(C * bin_rates[2])
            # Ensure we don't exceed budget
            while C + n_bin2_target + n_bin3_target + n_false + n_corrections > contact_budget:
                C -= 1
        else:
            C = C_raw

        # Step 3: subsample bin1 with long-range weighting if needed
        if C < C_raw:
            weights = [
                _long_range_weight(abs(c.j - c.i), r3b.long_range_threshold)
                for c in bin1_contacts
            ]
            selected_bin1 = _weighted_sample_without_replacement(
                bin1_contacts, weights, C, rng,
            )
        else:
            selected_bin1 = list(bin1_contacts)

        # Sample bin2
        n_bin2 = min(n_bin2_target, len(bin2_contacts))
        selected_bin2 = rng.sample(bin2_contacts, n_bin2) if n_bin2 > 0 else []

        # Sample bin3 (lazy: sample eligible pairs not in cutoff set)
        selected_bin3 = _sample_bin3_pairs(
            eligible_residues, pairs_within_cutoff, n_bin3_target,
            parse_result, rng,
        )

        # Build the contact list with bin assignments
        contacts_pre_filter = C_raw + len(bin2_contacts)
        contact_list: list[BinnedContact] = []

        for c in selected_bin1:
            contact_list.append(BinnedContact(
                i=c.i, j=c.j, atom_i=c.atom_i, atom_j=c.atom_j,
                bin_idx=0,
            ))
        for c in selected_bin2:
            contact_list.append(BinnedContact(
                i=c.i, j=c.j, atom_i=c.atom_i, atom_j=c.atom_j,
                bin_idx=1,
            ))
        for c in selected_bin3:
            contact_list.append(BinnedContact(
                i=c.i, j=c.j, atom_i=c.atom_i, atom_j=c.atom_j,
                bin_idx=2,
            ))

        # Step 8: atom resampling (1% per real contact)
        for bc in contact_list:
            if rng.random() < r3b.atom_resample_prob:
                atoms_i = get_residue_heavy_atoms(parse_result, bc.i)
                atoms_j = get_residue_heavy_atoms(parse_result, bc.j)
                if atoms_i and atoms_j:
                    bc.atom_i = rng.choice(atoms_i)
                    bc.atom_j = rng.choice(atoms_j)
                    dist = compute_atom_distance(
                        parse_result, bc.i, bc.atom_i, bc.j, bc.atom_j,
                    )
                    bc.bin_idx = _assign_bin(dist, bin_edges)

        # Step 5: false contact injection
        # Compute bin proportions from the real contact set
        bin_counts = [0] * (len(bin_edges) + 1)
        for bc in contact_list:
            bin_counts[bc.bin_idx] += 1
        total_real = sum(bin_counts)
        if total_real == 0:
            bin_probs = [1.0 / (len(bin_edges) + 1)] * (len(bin_edges) + 1)
        else:
            bin_probs = [c / total_real for c in bin_counts]

        false_contacts: list[BinnedContact] = []
        false_true_bins: dict[int, int] = {}  # index in false_contacts -> true bin

        for fi in range(n_false):
            if len(eligible_residues) < 2:
                break
            ri, rj = _sample_pair(eligible_residues, rng)
            atoms_i = get_residue_heavy_atoms(parse_result, ri)
            atoms_j = get_residue_heavy_atoms(parse_result, rj)
            if not atoms_i or not atoms_j:
                continue
            atom_i = rng.choice(atoms_i)
            atom_j = rng.choice(atoms_j)
            # Sample a bin from the categorical distribution
            sampled_bin = _sample_categorical(bin_probs, rng)

            # Determine true bin for this pair+atoms
            dist = compute_atom_distance(
                parse_result, ri, atom_i, rj, atom_j,
            )
            true_bin = _assign_bin(dist, bin_edges)
            false_true_bins[len(false_contacts)] = true_bin

            fc = BinnedContact(
                i=ri, j=rj, atom_i=atom_i, atom_j=atom_j,
                bin_idx=sampled_bin, is_false=True,
            )
            false_contacts.append(fc)

        # Mark false contacts that happen to be correct
        actually_false = []
        for fi, fc in enumerate(false_contacts):
            if fc.bin_idx != false_true_bins.get(fi, -1):
                actually_false.append(fi)

        # Step 7: combine and shuffle
        all_contacts = contact_list + false_contacts
        rng.shuffle(all_contacts)

        # Step 6: insert corrections
        # Build correction entries for false contacts
        corrections_needed = {}  # maps (i,j) of false contact -> (true atoms, true bin)
        for fi, fc in enumerate(false_contacts):
            if fi not in actually_false:
                continue  # happened to be correct, no correction needed
            # Find the closest atom pair for this residue pair
            true_bin = false_true_bins[fi]
            # Get the closest atoms from the binned contacts if available
            closest = _find_closest_contact(
                fc.i, fc.j, bins, parse_result, bin_edges,
            )
            corrections_needed[(fc.i, fc.j)] = closest

        # Now iterate through positions and insert corrections.
        # We budget at most n_corrections total correction entries to avoid
        # exceeding the token budget.
        T = len(all_contacts)
        output_entries: list[BinnedContact] = []
        uncorrected: dict[tuple[int, int], tuple[str, str, int]] = {}
        corrections_emitted = 0

        position = 0
        for bc in all_contacts:
            output_entries.append(bc)
            position += 1

            # If this is a false contact, add to uncorrected set
            if bc.is_false and (bc.i, bc.j) in corrections_needed:
                uncorrected[(bc.i, bc.j)] = corrections_needed[(bc.i, bc.j)]

            # Try to emit corrections
            if uncorrected:
                F = len(uncorrected)
                frac = position / max(T, 1)
                p_correct = 1.0 - (1.0 - frac) ** F
                if rng.random() < p_correct:
                    # Pick a random uncorrected false contact to correct
                    pair = rng.choice(list(uncorrected.keys()))
                    corr_atom_i, corr_atom_j, corr_bin = uncorrected[pair]

                    # With small probability, the correction is itself wrong
                    if rng.random() < r3b.correction_resample_prob:
                        resampled_bin = _sample_categorical(bin_probs, rng)
                        true_info = corrections_needed.get(pair)
                        if true_info and resampled_bin != true_info[2]:
                            # Still wrong — emit the bad correction but keep
                            # the pair in uncorrected with its true values so
                            # it will be corrected at the end.
                            correction = BinnedContact(
                                i=pair[0], j=pair[1],
                                atom_i=corr_atom_i, atom_j=corr_atom_j,
                                bin_idx=resampled_bin, is_correction=True,
                            )
                            output_entries.append(correction)
                            corrections_emitted += 1
                            # Don't remove from uncorrected — will get a true
                            # correction at the end
                            continue
                        else:
                            corr_bin = resampled_bin

                    # Apply atom resampling to corrections too
                    if rng.random() < r3b.atom_resample_prob:
                        atoms_i = get_residue_heavy_atoms(parse_result, pair[0])
                        atoms_j = get_residue_heavy_atoms(parse_result, pair[1])
                        if atoms_i and atoms_j:
                            corr_atom_i = rng.choice(atoms_i)
                            corr_atom_j = rng.choice(atoms_j)
                            dist = compute_atom_distance(
                                parse_result, pair[0], corr_atom_i, pair[1], corr_atom_j,
                            )
                            corr_bin = _assign_bin(dist, bin_edges)

                    correction = BinnedContact(
                        i=pair[0], j=pair[1],
                        atom_i=corr_atom_i, atom_j=corr_atom_j,
                        bin_idx=corr_bin, is_correction=True,
                    )
                    output_entries.append(correction)
                    corrections_emitted += 1
                    del uncorrected[pair]

        # Flush remaining uncorrected false contacts at the end
        for pair, (corr_atom_i, corr_atom_j, corr_bin) in uncorrected.items():
            correction = BinnedContact(
                i=pair[0], j=pair[1],
                atom_i=corr_atom_i, atom_j=corr_atom_j,
                bin_idx=corr_bin, is_correction=True,
            )
            output_entries.append(correction)
            corrections_emitted += 1

        # Step 5 (pLDDT): insert pLDDT bin token at random position
        global_plddt = sum(r.plddt for r in parse_result.residues) / len(parse_result.residues)
        plddt_token = _plddt_bin_token(global_plddt, r3b.plddt_bin_edges)

        # Choose a random position between 5-tuple boundaries
        n_entries = len(output_entries)
        plddt_insert_pos = rng.randint(0, n_entries)  # 0 to n inclusive

        # Serialize
        doc_text = _serialize_random_3_bins(
            parse_result.residues,
            output_entries,
            plddt_token,
            plddt_insert_pos,
            bin_edges,
            self.name,
        )

        return GeneratorResult(
            doc_text=doc_text,
            contacts_pre_filter=contacts_pre_filter,
            contacts_emitted=len(output_entries),
        )


def _serialize_random_3_bins(
    residues, entries, plddt_token, plddt_pos, bin_edges, task_token,
):
    """Serialize a random-3-bins document to text."""
    tokens = []
    tokens.append(f"<{task_token}>")
    tokens.append("<begin_sequence>")
    for r in residues:
        tokens.append(f"<{r.name}>")
    tokens.append("<begin_contacts>")

    for idx, entry in enumerate(entries):
        if idx == plddt_pos:
            tokens.append(f"<{plddt_token}>")
        tokens.append(f"<p{entry.i}>")
        tokens.append(f"<p{entry.j}>")
        tokens.append(f"<{entry.atom_i}>")
        tokens.append(f"<{entry.atom_j}>")
        tokens.append(f"<{_bin_token(entry.bin_idx, bin_edges)}>")

    # If plddt_pos == len(entries), insert at end
    if plddt_pos >= len(entries):
        tokens.append(f"<{plddt_token}>")

    tokens.append("<end_contacts>")
    tokens.append("<end>")

    return " ".join(tokens) + "\n"


def _sample_poisson(rng: random.Random, lam: float) -> int:
    """Sample from Poisson distribution using inverse CDF."""
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= rng.random()
        if p < L:
            return k - 1


def _sample_pair(eligible: list[int], rng: random.Random) -> tuple[int, int]:
    """Sample a random eligible pair (i, j) with i < j and |i-j| > 1."""
    while True:
        a, b = rng.sample(eligible, 2)
        i, j = min(a, b), max(a, b)
        if j - i > 1:
            return i, j


def _sample_categorical(probs: list[float], rng: random.Random) -> int:
    """Sample from a categorical distribution."""
    u = rng.random()
    cumsum = 0.0
    for i, p in enumerate(probs):
        cumsum += p
        if u < cumsum:
            return i
    return len(probs) - 1


def _sample_bin3_pairs(
    eligible_residues: list[int],
    pairs_within_cutoff: set[tuple[int, int]],
    n_target: int,
    parse_result: ParseResult,
    rng: random.Random,
) -> list[Contact]:
    """Sample residue pairs for bin 3 (beyond max cutoff)."""
    if len(eligible_residues) < 2 or n_target <= 0:
        return []

    # Sample pairs not in the cutoff set
    results = []
    attempts = 0
    max_attempts = n_target * 20

    while len(results) < n_target and attempts < max_attempts:
        attempts += 1
        a, b = rng.sample(eligible_residues, 2)
        i, j = min(a, b), max(a, b)
        if j - i <= 1:
            continue
        if (i, j) in pairs_within_cutoff:
            continue
        # Already sampled this pair?
        if any(c.i == i and c.j == j for c in results):
            continue

        # Get closest heavy atoms (we don't have them precomputed,
        # so pick arbitrary atoms — the distance is known to be > cutoff)
        atoms_i = get_residue_heavy_atoms(parse_result, i)
        atoms_j = get_residue_heavy_atoms(parse_result, j)
        if not atoms_i or not atoms_j:
            continue

        # Use CA if available, otherwise first heavy atom
        atom_i = "CA" if "CA" in atoms_i else atoms_i[0]
        atom_j = "CA" if "CA" in atoms_j else atoms_j[0]

        results.append(Contact(i=i, j=j, atom_i=atom_i, atom_j=atom_j, distance=float("inf")))

    return results


def _find_closest_contact(
    i: int, j: int,
    bins: dict,
    parse_result: ParseResult,
    bin_edges: list[float],
) -> tuple[str, str, int]:
    """Find the closest heavy-atom pair for a residue pair and its bin.

    Searches precomputed bins first, falls back to computing distance.
    """
    # Check if this pair exists in any bin
    for bin_idx in range(len(bin_edges) + 1):
        if bin_idx not in bins:
            continue
        for c in bins[bin_idx]:
            if c.i == i and c.j == j:
                return (c.atom_i, c.atom_j, bin_idx)

    # Not found in precomputed — compute distance for CA-CA
    atoms_i = get_residue_heavy_atoms(parse_result, i)
    atoms_j = get_residue_heavy_atoms(parse_result, j)
    if not atoms_i or not atoms_j:
        return ("CA", "CA", len(bin_edges))

    # Find closest pair by brute force
    best_dist = float("inf")
    best_ai, best_aj = atoms_i[0], atoms_j[0]
    for ai in atoms_i:
        for aj in atoms_j:
            d = compute_atom_distance(parse_result, i, ai, j, aj)
            if d < best_dist:
                best_dist = d
                best_ai, best_aj = ai, aj

    bin_idx = _assign_bin(best_dist, bin_edges)
    return (best_ai, best_aj, bin_idx)
