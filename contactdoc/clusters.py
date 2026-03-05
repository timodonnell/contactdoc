"""Load cluster mapping files (TSV.GZ): uniprot_accession -> rep_id.

Cluster files from the Steinegger lab use bare UniProt accessions as IDs,
NOT AFDB entry IDs (which look like AF-{accession}-F1). All lookups in
this module use UniProt accessions.
"""

import gzip
from pathlib import Path


def load_afdb50_mapping(tsv_gz_path: str | Path) -> dict[str, str]:
    """Load AFDB50 sequence-similarity cluster file (file 7).

    Format: rep_id<TAB>member_id per line.
    IDs are UniProt accessions.
    Returns dict mapping member_accession -> rep_accession.
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
    """Load structural cluster mapping from the all-members file (file 5).

    Format: rep_id<TAB>member_id<TAB>cluFlag<TAB>tax_id per line.
    Only loads entries with cluFlag=2 (structurally clustered).
    IDs are UniProt accessions.
    Returns dict mapping member_accession -> rep_accession.
    """
    mapping: dict[str, str] = {}
    with gzip.open(tsv_gz_path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            rep_id, member_id, clu_flag = parts[0], parts[1], parts[2]
            if clu_flag == "2":
                mapping[member_id] = rep_id
    return mapping


def get_cluster_id(accession: str, cluster_map: dict[str, str]) -> str | None:
    """Get cluster rep ID for a UniProt accession. Returns None if not found."""
    return cluster_map.get(accession)
