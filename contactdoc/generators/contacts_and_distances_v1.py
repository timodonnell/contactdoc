"""Contacts-and-distances-v1 document generator.

Generates documents with two types of statements:
- Contact statements (3 tokens): <mode> <res1> <res2>
  where mode is long-range-contact, medium-range-contact, or short-range-contact
  based on CB-CB distance ≤ 8Å and sequence separation.
- Distance statements (6 tokens): <distance> <res1> <res2> <atom1> <atom2> <dist>
  with fine-grained 0.5Å resolution distance bins for randomly sampled atom pairs.

Statements are rank-ordered: contacts get rank ~ N(2, 1), distances get rank ~ N(0, 1),
so contacts tend to appear earlier in the document.

See prompts/contacts-and-distances-v1.txt for full specification.
"""

import hashlib
import math
import random
from dataclasses import dataclass

from ..cif_parse import ParseResult
from ..contacts import build_atom_position_cache, min_distance_from_cache
from .base import DocumentGenerator, GeneratorResult


CONTACT_TOKENS_PER_STATEMENT = 3  # <mode> <p_i> <p_j>
DISTANCE_TOKENS_PER_STATEMENT = 6  # <distance> <p_i> <p_j> <atom1> <atom2> <d_val>


def _plddt_bin_token(global_plddt: float, bin_edges: list[float]) -> str:
    """Convert global pLDDT to a bin token string."""
    for i, edge in enumerate(bin_edges):
        if global_plddt < edge:
            if i == 0:
                return f"plddt_lt{int(edge)}"
            else:
                return f"plddt_{int(bin_edges[i-1])}_{int(edge)}"
    return f"plddt_{int(bin_edges[-1])}_100"


def _distance_token(dist: float) -> str:
    """Convert a distance in angstroms to a fine-grained bin token.

    64 bins at 0.5 Å resolution: d0.5 (< 0.5), d1.0 (0.5-1.0), ..., d32.0 (31.5-32.0).
    Distances > 32.0 are clamped to d32.0.
    """
    bin_idx = int(math.ceil(dist / 0.5))
    if bin_idx < 1:
        bin_idx = 1
    if bin_idx > 64:
        bin_idx = 64
    return f"d{bin_idx * 0.5:.1f}"


def _get_cb_or_ca(residue_atoms: list[tuple[str, float, float, float]], residue_name: str):
    """Get CB position (or CA for glycine). Returns (x, y, z) or None."""
    target = "CA" if residue_name == "GLY" else "CB"
    for name, x, y, z in residue_atoms:
        if name == target:
            return (x, y, z)
    # Fallback to CA if CB not found
    for name, x, y, z in residue_atoms:
        if name == "CA":
            return (x, y, z)
    return None


def _cb_distance(pos1, pos2) -> float:
    """Euclidean distance between two (x, y, z) positions."""
    dx = pos1[0] - pos2[0]
    dy = pos1[1] - pos2[1]
    dz = pos1[2] - pos2[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


class ContactsAndDistancesV1(DocumentGenerator):

    @property
    def name(self) -> str:
        return "contacts-and-distances-v1"

    def generate(self, parse_result, cfg):
        c = cfg.contacts_and_distances_v1
        plddt_min = c.residue_plddt_min

        # Deterministic seed from entry_id
        entry_id = parse_result.structure.name or "unknown"
        seed = int(hashlib.sha1(entry_id.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)

        residues = parse_result.residues
        num_residues = len(residues)

        # Build atom cache (no pLDDT filter — distance statements use all residues)
        atom_cache_all = build_atom_position_cache(parse_result, -float("inf"))
        # Build pLDDT-filtered cache for contact eligibility
        atom_cache_plddt = build_atom_position_cache(parse_result, plddt_min)

        # Residue name lookup
        res_name = {r.index: r.name for r in residues}

        # --- Compute contacts (CB-CB ≤ 8Å) ---
        # Get CB/CA positions for pLDDT-passing residues
        cb_positions = {}
        for idx, atoms in atom_cache_plddt.items():
            pos = _get_cb_or_ca(atoms, res_name[idx])
            if pos is not None:
                cb_positions[idx] = pos

        # Find all contacts
        contacts_long = []   # sep >= 24
        contacts_medium = []  # 12 <= sep < 24
        contacts_short = []   # 6 <= sep < 12

        cb_indices = sorted(cb_positions.keys())
        for ii in range(len(cb_indices)):
            for jj in range(ii + 1, len(cb_indices)):
                i = cb_indices[ii]
                j = cb_indices[jj]
                sep = j - i
                if sep <= 1:
                    continue
                if sep < c.short_range_sep:
                    continue  # sep < 6, skip

                dist = _cb_distance(cb_positions[i], cb_positions[j])
                if dist > c.contact_cutoff:
                    continue

                if sep >= c.long_range_sep:
                    contacts_long.append((i, j))
                elif sep >= c.medium_range_sep:
                    contacts_medium.append((i, j))
                else:  # sep >= short_range_sep
                    contacts_short.append((i, j))

        # --- Budget calculation ---
        # Overhead: task(1) + begin_sequence(1) + residues + begin_statements(1) + end(1) + plddt(1) = 5
        fixed_overhead = 5 + num_residues
        available_tokens = c.max_tokens - fixed_overhead

        if available_tokens <= 0:
            return "sequence_too_long_for_budget"

        # Sample fractions for each contact type
        f_long_raw = rng.uniform(c.contact_f_range[0], c.contact_f_range[1])
        f_medium_raw = rng.uniform(c.contact_f_range[0], c.contact_f_range[1])
        f_short_raw = rng.uniform(c.contact_f_range[0], c.contact_f_range[1])
        f_long = max(0.0, f_long_raw)
        f_medium = max(0.0, f_medium_raw)
        f_short = max(0.0, f_short_raw)

        # Allocate tokens for each contact type
        tokens_for_long = int(available_tokens * f_long)
        tokens_for_medium = int(available_tokens * f_medium)
        tokens_for_short = int(available_tokens * f_short)

        # Number of contact statements (3 tokens each)
        n_long = min(tokens_for_long // CONTACT_TOKENS_PER_STATEMENT, len(contacts_long))
        n_medium = min(tokens_for_medium // CONTACT_TOKENS_PER_STATEMENT, len(contacts_medium))
        n_short = min(tokens_for_short // CONTACT_TOKENS_PER_STATEMENT, len(contacts_short))

        # Tokens used by contacts
        contact_tokens_used = (n_long + n_medium + n_short) * CONTACT_TOKENS_PER_STATEMENT

        # Remaining tokens for distance statements (6 tokens each)
        remaining_tokens = available_tokens - contact_tokens_used
        n_distance = remaining_tokens // DISTANCE_TOKENS_PER_STATEMENT

        # --- Sample contacts ---
        selected_long = rng.sample(contacts_long, n_long) if n_long > 0 else []
        selected_medium = rng.sample(contacts_medium, n_medium) if n_medium > 0 else []
        selected_short = rng.sample(contacts_short, n_short) if n_short > 0 else []

        # --- Sample distance statements ---
        # Eligible pairs: all pairs with |i-j| > 1, no pLDDT filter
        all_indices = sorted(atom_cache_all.keys())
        if len(all_indices) < 2:
            return "too_few_residues"

        distance_statements = []
        for _ in range(n_distance):
            # Sample pair uniformly
            a, b = rng.sample(all_indices, 2)
            i, j = min(a, b), max(a, b)
            if j - i <= 1:
                # Resample
                for _ in range(10):
                    a, b = rng.sample(all_indices, 2)
                    i, j = min(a, b), max(a, b)
                    if j - i > 1:
                        break
                else:
                    continue

            # Sample atoms uniformly from each residue
            atoms_i = atom_cache_all[i]
            atoms_j = atom_cache_all[j]
            atom_i_name, ax, ay, az = rng.choice(atoms_i)
            atom_j_name, bx, by, bz = rng.choice(atoms_j)

            # Compute distance
            dist = math.sqrt((ax - bx)**2 + (ay - by)**2 + (az - bz)**2)
            dist_tok = _distance_token(dist)

            distance_statements.append((i, j, atom_i_name, atom_j_name, dist_tok))

        # --- Build statements with ranks ---
        statements = []  # (rank, token_list)

        for i, j in selected_long:
            rank = rng.gauss(c.contact_rank_mean, c.rank_std)
            statements.append((rank, [f"<long-range-contact>", f"<p{i}>", f"<p{j}>"]))

        for i, j in selected_medium:
            rank = rng.gauss(c.contact_rank_mean, c.rank_std)
            statements.append((rank, [f"<medium-range-contact>", f"<p{i}>", f"<p{j}>"]))

        for i, j in selected_short:
            rank = rng.gauss(c.contact_rank_mean, c.rank_std)
            statements.append((rank, [f"<short-range-contact>", f"<p{i}>", f"<p{j}>"]))

        for i, j, atom_i, atom_j, dist_tok in distance_statements:
            rank = rng.gauss(c.distance_rank_mean, c.rank_std)
            statements.append((rank, [f"<distance>", f"<p{i}>", f"<p{j}>", f"<{atom_i}>", f"<{atom_j}>", f"<{dist_tok}>"]))

        # Sort by rank descending (high rank = earlier)
        statements.sort(key=lambda x: -x[0])

        # --- pLDDT token ---
        global_plddt = sum(r.plddt for r in residues) / len(residues)
        plddt_token = _plddt_bin_token(global_plddt, c.plddt_bin_edges)

        plddt_at_end = rng.random() < 0.5
        if plddt_at_end:
            plddt_insert_idx = None
        else:
            plddt_insert_idx = rng.randint(0, len(statements))

        # --- Serialize ---
        tokens = []
        tokens.append(f"<{self.name}>")
        tokens.append("<begin_sequence>")
        for r in residues:
            tokens.append(f"<{r.name}>")
        tokens.append("<begin_statements>")

        for idx, (rank, stmt_tokens) in enumerate(statements):
            if not plddt_at_end and idx == plddt_insert_idx:
                tokens.append(f"<{plddt_token}>")
            tokens.extend(stmt_tokens)

        # pLDDT at end of statements or after all statements if random pos == len
        if not plddt_at_end and plddt_insert_idx is not None and plddt_insert_idx >= len(statements):
            tokens.append(f"<{plddt_token}>")
        if plddt_at_end:
            tokens.append(f"<{plddt_token}>")

        tokens.append("<end>")

        doc_text = " ".join(tokens) + "\n"

        contacts_emitted = n_long + n_medium + n_short
        total_statements = contacts_emitted + len(distance_statements)

        return GeneratorResult(
            doc_text=doc_text,
            contacts_pre_filter=len(contacts_long) + len(contacts_medium) + len(contacts_short),
            contacts_emitted=total_statements,
        )
