#!/usr/bin/env python3
"""Run random-3-bins generator on afdb-1.6M, output Parquet matching protein-docs format."""

import glob
import hashlib
import os
import sys
import time
from multiprocessing import Pool

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from contactdoc.cif_parse import parse_cif, extract_residues
from contactdoc.config import PipelineConfig
from contactdoc.generators import get_generator


SRC = "/home/ubuntu/tim-us-east-1/datasets/afdb-1.6M"
DST = "/home/ubuntu/protein-docs/random-3-bins"
WORKERS = 64

OUTPUT_SCHEMA = pa.schema([
    ("document", pa.string()),
    ("entry_id", pa.string()),
    ("uniprot_accession", pa.string()),
    ("tax_id", pa.int64()),
    ("organism_name", pa.string()),
    ("global_plddt", pa.float32()),
    ("seq_len", pa.int32()),
    ("contacts_pre_filter", pa.int32()),
    ("contacts_emitted", pa.int32()),
    ("residues_passing_plddt", pa.int32()),
    ("split", pa.string()),
    ("seq_cluster_id", pa.string()),
    ("struct_cluster_id", pa.string()),
    ("sha1", pa.string()),
])


def process_shard(shard_path):
    """Process one input parquet shard, return (shard_path, n_docs, n_errors, elapsed)."""
    t0 = time.time()
    cfg = PipelineConfig()
    gen = get_generator("random-3-bins")

    table = pq.read_table(shard_path)
    n_docs = 0
    n_errors = 0

    rows = {col: [] for col in OUTPUT_SCHEMA.names}

    for i in range(len(table)):
        row = table.slice(i, 1).to_pydict()
        entry_id = row["entry_id"][0]
        cif = row["cif_content"][0]

        try:
            structure = parse_cif(cif)
            structure.name = entry_id
            pr = extract_residues(structure)
            if isinstance(pr, str):
                n_errors += 1
                continue
            result = gen.generate(pr, cfg)
            if isinstance(result, str):
                n_errors += 1
                continue

            plddt_min = cfg.filters.residue_plddt_min
            residues_passing = sum(
                1 for r in pr.residues if r.plddt >= plddt_min
            )

            rows["document"].append(result.doc_text)
            rows["entry_id"].append(entry_id)
            rows["uniprot_accession"].append(row["uniprot_accession"][0])
            rows["tax_id"].append(row["tax_id"][0])
            rows["organism_name"].append(row["organism_name"][0])
            rows["global_plddt"].append(row["global_plddt"][0])
            rows["seq_len"].append(row["seq_len"][0])
            rows["contacts_pre_filter"].append(result.contacts_pre_filter)
            rows["contacts_emitted"].append(result.contacts_emitted)
            rows["residues_passing_plddt"].append(residues_passing)
            rows["split"].append(row["split"][0])
            rows["seq_cluster_id"].append(row["seq_cluster_id"][0])
            rows["struct_cluster_id"].append(row["struct_cluster_id"][0])
            rows["sha1"].append(
                hashlib.sha1(result.doc_text.encode()).hexdigest()
            )
            n_docs += 1
        except Exception:
            n_errors += 1

    # Write output split by train/val/test
    shard_name = os.path.basename(shard_path)
    for split in ("train", "val", "test"):
        split_indices = [
            j for j, s in enumerate(rows["split"]) if s == split
        ]
        if not split_indices:
            continue

        split_rows = {
            col: [rows[col][j] for j in split_indices]
            for col in OUTPUT_SCHEMA.names
        }
        out_table = pa.table(split_rows, schema=OUTPUT_SCHEMA)
        split_dir = os.path.join(DST, split)
        os.makedirs(split_dir, exist_ok=True)
        out_path = os.path.join(split_dir, shard_name)
        pq.write_table(out_table, out_path, compression="zstd")

    elapsed = time.time() - t0
    return shard_path, n_docs, n_errors, elapsed


def main():
    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(DST, split), exist_ok=True)

    shard_paths = sorted(glob.glob(os.path.join(SRC, "**", "shard_*.parquet"), recursive=True))
    print(f"Found {len(shard_paths)} input shards")
    print(f"Using {WORKERS} workers")
    print(f"Output: {DST}")
    print()

    total_docs = 0
    total_errors = 0
    total_shards = 0
    t_start = time.time()

    with Pool(WORKERS) as pool:
        for result in pool.imap_unordered(process_shard, shard_paths):
            shard_path, n_docs, n_errors, elapsed = result
            total_shards += 1
            total_docs += n_docs
            total_errors += n_errors

            if total_shards % 50 == 0 or total_shards <= 5:
                wall = time.time() - t_start
                rate = total_shards / wall * 3600
                eta_h = (len(shard_paths) - total_shards) / (total_shards / wall) / 3600
                print(
                    f"[{total_shards}/{len(shard_paths)}] "
                    f"docs={total_docs} errors={total_errors} "
                    f"shard_time={elapsed:.1f}s "
                    f"rate={rate:.0f} shards/hr "
                    f"ETA={eta_h:.1f}h"
                )

    wall = time.time() - t_start
    print(f"\nDone! {total_docs} documents, {total_errors} errors in {wall/3600:.1f}h")


if __name__ == "__main__":
    main()
