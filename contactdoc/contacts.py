"""Contact computation using Gemmi NeighborSearch + ContactSearch."""

import math
from dataclasses import dataclass

import gemmi

from .cif_parse import ParseResult, build_residue_index_map, _residue_key


@dataclass
class Contact:
    i: int  # 1-based residue index, i < j
    j: int
    atom_i: str  # atom name stripped
    atom_j: str
    distance: float


def compute_contacts(
    parse_result: ParseResult,
    cutoff: float = 4.0,
) -> list[Contact]:
    """Compute best contact per residue pair using Gemmi ContactSearch.

    Returns unsorted list of contacts (one per eligible residue pair).
    """
    model = parse_result.model
    structure = parse_result.structure

    ns = gemmi.NeighborSearch(model, structure.cell, cutoff)
    ns.populate(include_h=False)

    cs = gemmi.ContactSearch(cutoff)
    cs.ignore = gemmi.ContactSearch.Ignore.AdjacentResidues

    results = cs.find_contacts(ns)

    rmap = build_residue_index_map(parse_result)
    best: dict[tuple[int, int], tuple[float, str, str]] = {}

    for contact in results:
        cra1 = contact.partner1
        cra2 = contact.partner2

        key1 = _residue_key(cra1.residue)
        key2 = _residue_key(cra2.residue)

        idx1 = rmap.get(key1)
        idx2 = rmap.get(key2)
        if idx1 is None or idx2 is None:
            continue

        # Normalize so i < j
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1
            cra1, cra2 = cra2, cra1

        if idx1 == idx2:
            continue
        # |i-j| > 1 should already be enforced by AdjacentResidues, but be safe
        if idx2 - idx1 <= 1:
            continue

        atom_name1 = cra1.atom.name.strip()
        atom_name2 = cra2.atom.name.strip()
        dist = contact.dist

        pair = (idx1, idx2)
        candidate = (dist, atom_name1, atom_name2)
        existing = best.get(pair)

        if existing is None:
            best[pair] = candidate
        else:
            # Smaller distance wins; tie-break by lex (atom_i, atom_j)
            if (candidate[0], candidate[1], candidate[2]) < (existing[0], existing[1], existing[2]):
                best[pair] = candidate

    return [
        Contact(i=i, j=j, atom_i=a1, atom_j=a2, distance=d)
        for (i, j), (d, a1, a2) in best.items()
    ]


def filter_contacts_by_plddt(
    contacts: list[Contact],
    parse_result: ParseResult,
    residue_plddt_min: float,
) -> list[Contact]:
    """Remove contacts where either endpoint residue has pLDDT below threshold."""
    plddt_map = {r.index: r.plddt for r in parse_result.residues}
    return [
        c for c in contacts
        if plddt_map.get(c.i, float("-inf")) >= residue_plddt_min
        and plddt_map.get(c.j, float("-inf")) >= residue_plddt_min
    ]


def sort_and_truncate(
    contacts: list[Contact],
    max_contacts: int,
) -> list[Contact]:
    """Sort by (j-i) desc, i asc, j asc. Truncate to max_contacts."""
    contacts.sort(key=lambda c: (-(c.j - c.i), c.i, c.j))
    return contacts[:max_contacts]


def compute_binned_contacts(
    parse_result: ParseResult,
    bin_edges: list[float],
    residue_plddt_min: float,
) -> dict[int, list[Contact]]:
    """Compute contacts binned by distance.

    Uses the largest bin edge as the Gemmi cutoff. Pairs beyond the cutoff
    are inferred as belonging to the last bin without computing distances.

    Args:
        parse_result: Parsed structure.
        bin_edges: Distance thresholds, e.g. [4.0, 12.0] gives 3 bins:
                   bin 0: < 4.0, bin 1: 4.0-12.0, bin 2: > 12.0
        residue_plddt_min: Per-residue pLDDT filter threshold.

    Returns:
        Dict mapping bin index to list of Contact objects.
        The last bin (beyond max cutoff) has contacts with distance=inf
        and atoms set to the closest pair from the max-cutoff search
        (or empty strings if no atom pair was found within cutoff).
    """
    max_cutoff = max(bin_edges)

    # Get all contacts within max cutoff
    all_within = compute_contacts(parse_result, max_cutoff)
    all_within = filter_contacts_by_plddt(all_within, parse_result, residue_plddt_min)

    # Bin the contacts
    num_bins = len(bin_edges) + 1
    bins: dict[int, list[Contact]] = {i: [] for i in range(num_bins)}

    pairs_within_cutoff = set()
    for c in all_within:
        pairs_within_cutoff.add((c.i, c.j))
        bin_idx = _assign_bin(c.distance, bin_edges)
        bins[bin_idx].append(c)

    # The last bin is all eligible pairs NOT in pairs_within_cutoff.
    # We don't enumerate these eagerly; instead we store the set of
    # eligible residues and the pairs to exclude. The generator will
    # sample from this set on demand.
    # Store metadata for lazy sampling on the bins dict.
    bins["_eligible_residues"] = _get_eligible_residues(parse_result, residue_plddt_min)
    bins["_pairs_within_cutoff"] = pairs_within_cutoff

    return bins


def _assign_bin(distance: float, bin_edges: list[float]) -> int:
    """Assign a distance to a bin index."""
    for i, edge in enumerate(bin_edges):
        if distance < edge:
            return i
    return len(bin_edges)


def _get_eligible_residues(parse_result: ParseResult, residue_plddt_min: float) -> list[int]:
    """Get 1-based indices of residues passing pLDDT filter."""
    return [r.index for r in parse_result.residues if r.plddt >= residue_plddt_min]


def get_residue_heavy_atoms(parse_result: ParseResult, residue_index: int) -> list[str]:
    """Get list of heavy atom names present in a residue."""
    for r in parse_result.residues:
        if r.index == residue_index:
            return [
                atom.name.strip()
                for atom in r.gemmi_residue
                if not atom.is_hydrogen()
            ]
    return []


def compute_atom_distance(
    parse_result: ParseResult,
    res_i: int, atom_name_i: str,
    res_j: int, atom_name_j: str,
) -> float:
    """Compute distance between two specific atoms in the structure.

    Returns float('inf') if either atom is not found.
    """
    atom_i = _find_atom(parse_result, res_i, atom_name_i)
    atom_j = _find_atom(parse_result, res_j, atom_name_j)
    if atom_i is None or atom_j is None:
        return float("inf")
    pos_i = atom_i.pos
    pos_j = atom_j.pos
    dx = pos_i.x - pos_j.x
    dy = pos_i.y - pos_j.y
    dz = pos_i.z - pos_j.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _find_atom(parse_result: ParseResult, residue_index: int, atom_name: str):
    """Find a gemmi Atom by residue index and atom name."""
    for r in parse_result.residues:
        if r.index == residue_index:
            for atom in r.gemmi_residue:
                if atom.name.strip() == atom_name:
                    return atom
            return None
    return None
