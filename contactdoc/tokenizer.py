"""Fixed vocabulary tokenizer for ContactDoc documents.

Vocabulary layout (deterministic, no gaps):
  0: <pad>
  1: <begin_sequence>
  2: <begin_contacts>
  3: <end_contacts>
  4: <end>
  5: <newline>
  6: <end_of_document>
  7-8: task tokens (alphabetical)
  9-29: 20 canonical residues + <UNK> (alphabetical)
  30-66: 37 heavy atom names (alphabetical)
  67-2114: position tokens <p1> .. <p2048>
"""

import re

# Structural / control tokens
CONTROL_TOKENS = ["<pad>", "<begin_sequence>", "<begin_contacts>", "<end_contacts>", "<end>", "<newline>", "<end_of_document>"]

# Task tokens: one per document generation scheme, alphabetical
TASK_TOKENS = sorted([
    "contacts-and-distances-v1",
    "deterministic-positives-only",
    "random-3-bins",
])

# 20 canonical amino acids + UNK, alphabetical
RESIDUE_NAMES = sorted([
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "UNK",
])

# All standard heavy atom names from the 20 amino acids, alphabetical
ATOM_NAMES = sorted([
    "C", "CA", "CB", "CD", "CD1", "CD2", "CE", "CE1", "CE2", "CE3",
    "CG", "CG1", "CG2", "CH2", "CZ", "CZ2", "CZ3",
    "N", "ND1", "ND2", "NE", "NE1", "NE2", "NH1", "NH2", "NZ",
    "O", "OD1", "OD2", "OE1", "OE2", "OG", "OG1", "OH", "OXT",
    "SD", "SG",
])

# Correction markers
CORRECTION_TOKENS = sorted([
    "correction",
    "non-correction",
])

# Distance bin tokens
DISTANCE_BIN_TOKENS = sorted([
    "bin_4_12",
    "bin_gt12",
    "bin_lt4",
])

# pLDDT bin tokens
PLDDT_BIN_TOKENS = sorted([
    "plddt_70_75",
    "plddt_75_80",
    "plddt_80_85",
    "plddt_85_90",
    "plddt_90_95",
    "plddt_95_100",
    "plddt_lt70",
])

# Contact mode tokens for contacts-and-distances-v1
CONTACT_MODE_TOKENS = sorted([
    "distance",
    "long-range-contact",
    "medium-range-contact",
    "short-range-contact",
])

# Begin statements token
STATEMENT_TOKENS = sorted([
    "begin_statements",
])

# Fine-grained distance tokens: d0.5, d1.0, ..., d32.0 (64 bins)
FINE_DISTANCE_TOKENS = [f"d{v:.1f}" for v in [i * 0.5 for i in range(1, 65)]]

MAX_POSITION = 2048

_TOKEN_PATTERN = re.compile(r"<[^>]+>")


def build_vocab() -> tuple[dict[str, int], dict[int, str]]:
    """Build the full token vocabulary. Returns (token_to_id, id_to_token)."""
    tokens = []
    tokens.extend(CONTROL_TOKENS)
    tokens.extend(f"<{name}>" for name in TASK_TOKENS)
    tokens.extend(f"<{name}>" for name in RESIDUE_NAMES)
    tokens.extend(f"<{name}>" for name in ATOM_NAMES)
    tokens.extend(f"<{name}>" for name in CORRECTION_TOKENS)
    tokens.extend(f"<{name}>" for name in CONTACT_MODE_TOKENS)
    tokens.extend(f"<{name}>" for name in STATEMENT_TOKENS)
    tokens.extend(f"<{name}>" for name in DISTANCE_BIN_TOKENS)
    tokens.extend(f"<{name}>" for name in PLDDT_BIN_TOKENS)
    tokens.extend(f"<{name}>" for name in FINE_DISTANCE_TOKENS)
    tokens.extend(f"<p{i}>" for i in range(1, MAX_POSITION + 1))

    token_to_id = {tok: i for i, tok in enumerate(tokens)}
    id_to_token = {i: tok for i, tok in enumerate(tokens)}
    return token_to_id, id_to_token


_VOCAB, _ID_TO_TOKEN = build_vocab()
VOCAB_SIZE = len(_VOCAB)
PAD_ID = _VOCAB["<pad>"]


def encode(doc_text: str) -> list[int]:
    """Tokenize a single document string into a list of token IDs.

    Each line becomes tokens followed by a <newline> token.
    The final <end> line's newline is included.
    """
    ids = []
    for line in doc_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        tokens = _TOKEN_PATTERN.findall(line)
        for tok in tokens:
            tok_id = _VOCAB.get(tok)
            if tok_id is None:
                raise ValueError(f"Unknown token: {tok}")
            ids.append(tok_id)
        ids.append(_VOCAB["<newline>"])
    return ids


def decode(ids: list[int]) -> str:
    """Convert token IDs back to document text."""
    lines = []
    current_line = []
    for tok_id in ids:
        if tok_id == PAD_ID:
            continue
        tok = _ID_TO_TOKEN[tok_id]
        if tok == "<newline>":
            lines.append(" ".join(current_line))
            current_line = []
        else:
            current_line.append(tok)
    if current_line:
        lines.append(" ".join(current_line))
    return "\n".join(lines) + "\n"
