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
    "deterministic-positives-only",
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

MAX_POSITION = 2048

_TOKEN_PATTERN = re.compile(r"<[^>]+>")


def build_vocab() -> tuple[dict[str, int], dict[int, str]]:
    """Build the full token vocabulary. Returns (token_to_id, id_to_token)."""
    tokens = []
    tokens.extend(CONTROL_TOKENS)
    tokens.extend(f"<{name}>" for name in TASK_TOKENS)
    tokens.extend(f"<{name}>" for name in RESIDUE_NAMES)
    tokens.extend(f"<{name}>" for name in ATOM_NAMES)
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
