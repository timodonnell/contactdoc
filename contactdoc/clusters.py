"""Load cluster mapping files (TSV.GZ): member_id -> rep_id."""

import gzip
from pathlib import Path


def load_afdb50_mapping(tsv_gz_path: str | Path) -> dict[str, str]:
    """Load AFDB50 sequence-similarity cluster file.

    Format: rep_id<TAB>member_id per line.
    Returns dict mapping member_id -> rep_id.
    """
    mapping: dict[str, str] = {}
    with gzip.open(tsv_gz_path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            rep_id, member_id = parts[0], parts[1]
            mapping[member_id] = rep_id
    return mapping


def load_structural_mapping(tsv_gz_path: str | Path) -> dict[str, str]:
    """Load Foldseek structural cluster file.

    Format: member_id<TAB>rep_id<TAB>tax_id per line.
    Returns dict mapping member_id -> rep_id.
    """
    mapping: dict[str, str] = {}
    with gzip.open(tsv_gz_path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            member_id, rep_id = parts[0], parts[1]
            mapping[member_id] = rep_id
    return mapping


def get_cluster_id(entry_id: str, cluster_map: dict[str, str]) -> str | None:
    """Get cluster rep ID for an entry. Returns None if not found."""
    return cluster_map.get(entry_id)
