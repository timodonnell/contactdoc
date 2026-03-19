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
    build_atom_position_cache,
    compute_contacts,
    filter_contacts_by_plddt,
    min_distance_from_cache,
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

        # Step 1: compute bin-1 contacts (< 4 Å) using fast Gemmi search
        first_edge = bin_edges[0]
        bin1_contacts = compute_contacts(parse_result, first_edge)
        bin1_contacts = filter_contacts_by_plddt(
            bin1_contacts, parse_result, plddt_min,
        )
        bin1_pairs = {(c.i, c.j) for c in bin1_contacts}

        if not bin1_contacts:
            return "no_contacts_in_bin1"

        C_raw = len(bin1_contacts)

        # Build atom position cache for fast distance computation
        atom_cache = build_atom_position_cache(parse_result, plddt_min)
        eligible_residues = list(atom_cache.keys())

        # Step 3-4: budget calculation
        # Overhead: task(1) + begin_seq(1) + residues + begin_contacts(1)
        #           + end_contacts(1) + end(1) + plddt(1) = 6
        num_residues = len(parse_result.residues)
        fixed_overhead = 6 + num_residues
        tokens_per_contact = 6  # correction marker + 5-tuple
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

        # Step 2 (bin 2 & 3): sample random pairs and classify by distance
        selected_bin2, selected_bin3 = _sample_bin2_bin3(
            eligible_residues, bin1_pairs, atom_cache, bin_edges,
            n_bin2_target, n_bin3_target, rng,
        )

        # Build the contact list with bin assignments
        contacts_pre_filter = C_raw
        contact_list: list[BinnedContact] = []

        for c in selected_bin1:
            contact_list.append(BinnedContact(
                i=c.i, j=c.j, atom_i=c.atom_i, atom_j=c.atom_j, bin_idx=0,
            ))
        for c in selected_bin2:
            contact_list.append(BinnedContact(
                i=c.i, j=c.j, atom_i=c.atom_i, atom_j=c.atom_j, bin_idx=1,
            ))
        for c in selected_bin3:
            contact_list.append(BinnedContact(
                i=c.i, j=c.j, atom_i=c.atom_i, atom_j=c.atom_j, bin_idx=2,
            ))

        # Step 8: atom resampling (1% per real contact)
        for bc in contact_list:
            if rng.random() < r3b.atom_resample_prob:
                cached_i = atom_cache.get(bc.i)
                cached_j = atom_cache.get(bc.j)
                if cached_i and cached_j:
                    ai_name, _, _, _ = rng.choice(cached_i)
                    aj_name, _, _, _ = rng.choice(cached_j)
                    bc.atom_i = ai_name
                    bc.atom_j = aj_name
                    dist, _, _ = min_distance_from_cache(
                        [(ai_name, x, y, z) for (n, x, y, z) in cached_i if n == ai_name],
                        [(aj_name, x, y, z) for (n, x, y, z) in cached_j if n == aj_name],
                    )
                    bc.bin_idx = _assign_bin(dist, bin_edges)

        # Step 5: false contact injection
        bin_counts = [0] * (len(bin_edges) + 1)
        for bc in contact_list:
            bin_counts[bc.bin_idx] += 1
        total_real = sum(bin_counts)
        if total_real == 0:
            bin_probs = [1.0 / (len(bin_edges) + 1)] * (len(bin_edges) + 1)
        else:
            bin_probs = [c / total_real for c in bin_counts]

        false_contacts: list[BinnedContact] = []
        false_true_bins: dict[int, int] = {}

        for fi in range(n_false):
            if len(eligible_residues) < 2:
                break
            ri, rj = _sample_pair(eligible_residues, rng)
            cached_i = atom_cache.get(ri)
            cached_j = atom_cache.get(rj)
            if not cached_i or not cached_j:
                continue
            ai_name, _, _, _ = rng.choice(cached_i)
            aj_name, _, _, _ = rng.choice(cached_j)
            sampled_bin = _sample_categorical(bin_probs, rng)

            # Compute true distance for these specific atoms
            ai_entries = [(n, x, y, z) for (n, x, y, z) in cached_i if n == ai_name]
            aj_entries = [(n, x, y, z) for (n, x, y, z) in cached_j if n == aj_name]
            dist, _, _ = min_distance_from_cache(ai_entries, aj_entries)
            true_bin = _assign_bin(dist, bin_edges)
            false_true_bins[len(false_contacts)] = true_bin

            fc = BinnedContact(
                i=ri, j=rj, atom_i=ai_name, atom_j=aj_name,
                bin_idx=sampled_bin, is_false=True,
            )
            false_contacts.append(fc)

        actually_false = [
            fi for fi, fc in enumerate(false_contacts)
            if fc.bin_idx != false_true_bins.get(fi, -1)
        ]

        # Step 7: combine and shuffle
        all_contacts = contact_list + false_contacts
        rng.shuffle(all_contacts)

        # Step 6: build correction info for false contacts
        corrections_needed = {}
        for fi in actually_false:
            fc = false_contacts[fi]
            # Find closest atom pair for the correction
            cached_i = atom_cache.get(fc.i)
            cached_j = atom_cache.get(fc.j)
            if cached_i and cached_j:
                dist, ai, aj = min_distance_from_cache(cached_i, cached_j)
                corr_bin = _assign_bin(dist, bin_edges)
                corrections_needed[(fc.i, fc.j)] = (ai, aj, corr_bin)

        # Insert corrections during iteration
        T = len(all_contacts)
        output_entries: list[BinnedContact] = []
        uncorrected: dict[tuple[int, int], tuple[str, str, int]] = {}
        corrections_emitted = 0

        position = 0
        for bc in all_contacts:
            output_entries.append(bc)
            position += 1

            if bc.is_false and (bc.i, bc.j) in corrections_needed:
                uncorrected[(bc.i, bc.j)] = corrections_needed[(bc.i, bc.j)]

            if uncorrected:
                F = len(uncorrected)
                frac = position / max(T, 1)
                p_correct = 1.0 - (1.0 - frac) ** F
                if rng.random() < p_correct:
                    pair = rng.choice(list(uncorrected.keys()))
                    corr_atom_i, corr_atom_j, corr_bin = uncorrected[pair]

                    if rng.random() < r3b.correction_resample_prob:
                        resampled_bin = _sample_categorical(bin_probs, rng)
                        true_info = corrections_needed.get(pair)
                        if true_info and resampled_bin != true_info[2]:
                            correction = BinnedContact(
                                i=pair[0], j=pair[1],
                                atom_i=corr_atom_i, atom_j=corr_atom_j,
                                bin_idx=resampled_bin, is_correction=True,
                            )
                            output_entries.append(correction)
                            corrections_emitted += 1
                            continue
                        else:
                            corr_bin = resampled_bin

                    # Apply atom resampling to corrections
                    if rng.random() < r3b.atom_resample_prob:
                        cached_i = atom_cache.get(pair[0])
                        cached_j = atom_cache.get(pair[1])
                        if cached_i and cached_j:
                            corr_atom_i, _, _, _ = rng.choice(cached_i)
                            corr_atom_j, _, _, _ = rng.choice(cached_j)
                            ai_e = [(n, x, y, z) for (n, x, y, z) in cached_i if n == corr_atom_i]
                            aj_e = [(n, x, y, z) for (n, x, y, z) in cached_j if n == corr_atom_j]
                            dist, _, _ = min_distance_from_cache(ai_e, aj_e)
                            corr_bin = _assign_bin(dist, bin_edges)

                    correction = BinnedContact(
                        i=pair[0], j=pair[1],
                        atom_i=corr_atom_i, atom_j=corr_atom_j,
                        bin_idx=corr_bin, is_correction=True,
                    )
                    output_entries.append(correction)
                    corrections_emitted += 1
                    del uncorrected[pair]

        # Flush remaining uncorrected at end
        for pair, (corr_atom_i, corr_atom_j, corr_bin) in uncorrected.items():
            correction = BinnedContact(
                i=pair[0], j=pair[1],
                atom_i=corr_atom_i, atom_j=corr_atom_j,
                bin_idx=corr_bin, is_correction=True,
            )
            output_entries.append(correction)
            corrections_emitted += 1

        # pLDDT token
        global_plddt = sum(r.plddt for r in parse_result.residues) / len(parse_result.residues)
        plddt_token = _plddt_bin_token(global_plddt, r3b.plddt_bin_edges)
        n_entries = len(output_entries)

        # 50% of documents: pLDDT at end (just before <end>)
        # 50% of documents: pLDDT at random position in contacts section
        plddt_at_end = rng.random() < 0.5
        if plddt_at_end:
            plddt_insert_pos = None  # handled specially in serializer
        else:
            plddt_insert_pos = rng.randint(0, n_entries)

        # Serialize
        doc_text = _serialize_random_3_bins(
            parse_result.residues,
            output_entries,
            plddt_token,
            plddt_insert_pos,
            plddt_at_end,
            bin_edges,
            self.name,
        )

        return GeneratorResult(
            doc_text=doc_text,
            contacts_pre_filter=contacts_pre_filter,
            contacts_emitted=len(output_entries),
        )


def _serialize_random_3_bins(
    residues, entries, plddt_token, plddt_pos, plddt_at_end,
    bin_edges, task_token,
):
    """Serialize a random-3-bins document to text."""
    tokens = []
    tokens.append(f"<{task_token}>")
    tokens.append("<begin_sequence>")
    for r in residues:
        tokens.append(f"<{r.name}>")
    tokens.append("<begin_contacts>")

    seen_pairs: set[tuple[int, int]] = set()

    for idx, entry in enumerate(entries):
        # Insert pLDDT token at this position (if not at end)
        if not plddt_at_end and idx == plddt_pos:
            tokens.append(f"<{plddt_token}>")

        pair = (entry.i, entry.j)
        if pair in seen_pairs:
            tokens.append("<correction>")
        else:
            tokens.append("<non-correction>")
            seen_pairs.add(pair)

        tokens.append(f"<p{entry.i}>")
        tokens.append(f"<p{entry.j}>")
        tokens.append(f"<{entry.atom_i}>")
        tokens.append(f"<{entry.atom_j}>")
        tokens.append(f"<{_bin_token(entry.bin_idx, bin_edges)}>")

    # Insert pLDDT at end of contacts section if not already placed
    if not plddt_at_end and plddt_pos is not None and plddt_pos >= len(entries):
        tokens.append(f"<{plddt_token}>")

    tokens.append("<end_contacts>")

    # pLDDT at end: just before <end>
    if plddt_at_end:
        tokens.append(f"<{plddt_token}>")

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


def _sample_bin2_bin3(
    eligible_residues: list[int],
    bin1_pairs: set[tuple[int, int]],
    atom_cache: dict[int, list[tuple[str, float, float, float]]],
    bin_edges: list[float],
    n_bin2_target: int,
    n_bin3_target: int,
    rng: random.Random,
) -> tuple[list[Contact], list[Contact]]:
    """Sample residue pairs for bin 2 and bin 3 by random pair sampling.

    Avoids the expensive large-cutoff Gemmi search by sampling random pairs
    and computing their closest-atom distance from cached positions.
    """
    if len(eligible_residues) < 2:
        return [], []

    bin2_results: list[Contact] = []
    bin3_results: list[Contact] = []
    seen: set[tuple[int, int]] = set()
    max_attempts = (n_bin2_target + n_bin3_target) * 50

    for _ in range(max_attempts):
        if len(bin2_results) >= n_bin2_target and len(bin3_results) >= n_bin3_target:
            break

        a, b = rng.sample(eligible_residues, 2)
        i, j = min(a, b), max(a, b)
        if j - i <= 1 or (i, j) in bin1_pairs or (i, j) in seen:
            continue
        seen.add((i, j))

        cached_i = atom_cache.get(i)
        cached_j = atom_cache.get(j)
        if not cached_i or not cached_j:
            continue

        dist, best_ai, best_aj = min_distance_from_cache(cached_i, cached_j)
        bin_idx = _assign_bin(dist, bin_edges)

        if bin_idx == 1 and len(bin2_results) < n_bin2_target:
            bin2_results.append(Contact(
                i=i, j=j, atom_i=best_ai, atom_j=best_aj, distance=dist,
            ))
        elif bin_idx >= 2 and len(bin3_results) < n_bin3_target:
            bin3_results.append(Contact(
                i=i, j=j, atom_i=best_ai, atom_j=best_aj, distance=dist,
            ))

    return bin2_results, bin3_results
