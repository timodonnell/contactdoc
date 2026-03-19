#!/usr/bin/env python3
"""Run random-3-bins generator on all afdb-24M shards using multiprocessing."""

import glob
import os
import sys
import time
from multiprocessing import Pool

import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from contactdoc.cif_parse import parse_cif, extract_residues
from contactdoc.config import PipelineConfig
from contactdoc.generators import get_generator


SRC = "/lambda/nfs/tim-us-east-1/datasets/afdb-24M"
DST = "/home/ubuntu/random_3_bins_output"
WORKERS = 64


def process_shard(shard_path):
    """Process one parquet shard, return (shard_path, n_docs, n_errors, elapsed)."""
    t0 = time.time()
    cfg = PipelineConfig()
    gen = get_generator("random-3-bins")

    table = pq.read_table(shard_path)
    n_docs = 0
    n_errors = 0
    documents = []

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
            documents.append(result.doc_text)
            n_docs += 1
        except Exception:
            n_errors += 1

    # Write all documents to a single output file
    shard_name = os.path.basename(shard_path).replace(".parquet", ".txt")
    out_path = os.path.join(DST, shard_name)
    with open(out_path, "w") as f:
        for doc in documents:
            f.write(doc)
            f.write("<end_of_document>\n")

    elapsed = time.time() - t0
    return shard_path, n_docs, n_errors, elapsed


def main():
    os.makedirs(DST, exist_ok=True)

    shard_paths = sorted(
        glob.glob(os.path.join(SRC, "shard_*", "shard_*.parquet"))
    )
    print(f"Found {len(shard_paths)} shards")
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

            if total_shards % 100 == 0 or total_shards <= 5:
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
