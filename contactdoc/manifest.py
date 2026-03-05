"""Manifest building: enrich entries with cluster_id + split, write shards."""

import json
from pathlib import Path

from .clusters import get_cluster_id
from .config import PipelineConfig
from .splits import assign_split


def enrich_entries(
    entries: list[dict],
    seq_cluster_map: dict[str, str],
    struct_cluster_map: dict[str, str],
    cfg: PipelineConfig,
) -> tuple[list[dict], int]:
    """Attach cluster IDs, split, and gcs_uri to each entry.

    An entry must be present in BOTH the sequence (AFDB50) and structural
    cluster maps to be included. Entries missing from either are dropped.
    Split assignment uses the structural cluster (stricter grouping), so
    proteins with similar folds always land in the same split.

    Returns (enriched_entries, num_dropped).
    """
    enriched = []
    dropped = 0
    for entry in entries:
        entry_id = entry["entryId"]
        seq_cluster_id = get_cluster_id(entry_id, seq_cluster_map)
        struct_cluster_id = get_cluster_id(entry_id, struct_cluster_map)
        if seq_cluster_id is None or struct_cluster_id is None:
            dropped += 1
            continue
        split = assign_split(
            cfg.splits.seed,
            struct_cluster_id,
            cfg.splits.train_frac,
            cfg.splits.val_frac,
        )
        gcs_uri = f"{cfg.gcs_bucket_prefix}{entry_id}-model_v{cfg.afdb_version}.cif"

        enriched.append({
            **entry,
            "seq_cluster_id": seq_cluster_id,
            "struct_cluster_id": struct_cluster_id,
            "split_cluster_id": struct_cluster_id,
            "split": split,
            "gcs_uri": gcs_uri,
        })
    return enriched, dropped


def write_manifest_shards(
    entries: list[dict],
    output_dir: str | Path,
    shard_size: int,
) -> list[str]:
    """Write manifest entries as sharded JSONL files. Returns paths written."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for shard_idx in range(0, len(entries), shard_size):
        shard_entries = entries[shard_idx:shard_idx + shard_size]
        shard_num = shard_idx // shard_size
        path = output_dir / f"manifest_shard_{shard_num:06d}.jsonl"
        with open(path, "w") as f:
            for entry in shard_entries:
                # Drop uniprotSequence from manifest to save space
                row = {k: v for k, v in entry.items() if k != "uniprotSequence"}
                f.write(json.dumps(row, ensure_ascii=True) + "\n")
        paths.append(str(path))

    return paths


def read_manifest_shard(path: str | Path) -> list[dict]:
    """Read a manifest shard JSONL file."""
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries
