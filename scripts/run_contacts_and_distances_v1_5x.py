#!/usr/bin/env python3
"""Generate contacts-and-distances-v1 documents from afdb-24M with up to 5 entries per struct_cluster_id.

Same structure as run_random_3_bins_5x.py: round-ordered, shuffled within rounds.
"""

import glob
import hashlib
import os
import random
import sys
import time
from collections import defaultdict
from multiprocessing import Pool

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from contactdoc.cif_parse import parse_cif, extract_residues
from contactdoc.config import PipelineConfig
from contactdoc.generators import get_generator

SRC = "/home/ubuntu/tim-us-east-1/datasets/afdb-24M"
DST = "/home/ubuntu/protein-docs/contacts-and-distances-v1-5x"
WORKERS = 64
SHUFFLE_SEED = 456
MAX_PER_CLUSTER = 5
ROWS_PER_OUTPUT_SHARD = 2000
SCHEME = "contacts-and-distances-v1"

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
    ("round", pa.int32()),
    ("sha1", pa.string()),
])


def process_shard_entries(args):
    shard_path, entries = args
    cfg = PipelineConfig()
    gen = get_generator(SCHEME)
    plddt_min = cfg.contacts_and_distances_v1.residue_plddt_min

    table = pq.read_table(shard_path)
    results = []

    for row_idx, round_num in entries:
        row = table.slice(row_idx, 1).to_pydict()
        entry_id = row["entry_id"][0]
        cif = row["cif_content"][0]

        try:
            structure = parse_cif(cif)
            structure.name = entry_id
            pr = extract_residues(structure)
            if isinstance(pr, str):
                continue
            result = gen.generate(pr, cfg)
            if isinstance(result, str):
                continue

            residues_passing = sum(1 for r in pr.residues if r.plddt >= plddt_min)

            results.append({
                "document": result.doc_text,
                "entry_id": entry_id,
                "uniprot_accession": row["uniprot_accession"][0],
                "tax_id": row["tax_id"][0],
                "organism_name": row["organism_name"][0],
                "global_plddt": row["global_plddt"][0],
                "seq_len": row["seq_len"][0],
                "contacts_pre_filter": result.contacts_pre_filter,
                "contacts_emitted": result.contacts_emitted,
                "residues_passing_plddt": residues_passing,
                "split": row["split"][0],
                "seq_cluster_id": row["seq_cluster_id"][0],
                "struct_cluster_id": row["struct_cluster_id"][0],
                "round": round_num,
                "sha1": hashlib.sha1(result.doc_text.encode()).hexdigest(),
            })
        except Exception:
            continue

    return results


def main():
    shard_paths = sorted(
        glob.glob(os.path.join(SRC, "shard_*", "shard_*.parquet"))
    )
    print(f"Found {len(shard_paths)} source shards")

    # --- Phase 1: scan metadata, select top 5 per cluster ---
    print("Phase 1: scanning metadata...")
    cluster_entries = defaultdict(list)

    for path in tqdm(shard_paths, desc="Scanning"):
        table = pq.read_table(
            path, columns=["struct_cluster_id", "global_plddt"]
        )
        cluster_ids = table.column("struct_cluster_id").to_pylist()
        plddts = table.column("global_plddt").to_pylist()
        for i, (cid, plddt) in enumerate(zip(cluster_ids, plddts)):
            entries = cluster_entries[cid]
            if len(entries) < MAX_PER_CLUSTER:
                entries.append((plddt, path, i))
            else:
                min_idx = min(range(len(entries)), key=lambda x: entries[x][0])
                if plddt > entries[min_idx][0]:
                    entries[min_idx] = (plddt, path, i)

    total_entries = sum(len(v) for v in cluster_entries.values())
    print(f"Clusters: {len(cluster_entries)}, total entries: {total_entries}")

    # Randomly assign each cluster's entries to rounds
    rng = random.Random(SHUFFLE_SEED)
    round_entries = defaultdict(list)

    for cid, entries in cluster_entries.items():
        rounds = list(range(len(entries)))
        rng.shuffle(rounds)
        for (plddt, path, row_idx), round_num in zip(entries, rounds):
            round_entries[round_num].append((path, row_idx))

    del cluster_entries

    for r in sorted(round_entries.keys()):
        print(f"  Round {r}: {len(round_entries[r])} entries")

    # --- Phase 2: process each round ---
    for split_dir in ("train", "val", "test"):
        os.makedirs(os.path.join(DST, split_dir), exist_ok=True)

    global_shard_idx = 0

    for round_num in sorted(round_entries.keys()):
        entries = round_entries[round_num]
        print(f"\nRound {round_num}: processing {len(entries)} entries...")

        shard_to_entries = defaultdict(list)
        for path, row_idx in entries:
            shard_to_entries[path].append((row_idx, round_num))

        work_items = list(shard_to_entries.items())
        all_results = []
        t0 = time.time()

        with Pool(WORKERS) as pool:
            for i, batch_results in enumerate(
                pool.imap_unordered(process_shard_entries, work_items)
            ):
                all_results.extend(batch_results)
                if (i + 1) % 500 == 0:
                    elapsed = time.time() - t0
                    print(
                        f"  [{i+1}/{len(work_items)}] "
                        f"docs={len(all_results)} "
                        f"elapsed={elapsed:.0f}s"
                    )

        elapsed = time.time() - t0
        print(f"  Round {round_num}: {len(all_results)} docs in {elapsed:.0f}s")

        rng_round = random.Random(SHUFFLE_SEED + round_num)
        rng_round.shuffle(all_results)

        for split in ("train", "val", "test"):
            split_results = [r for r in all_results if r["split"] == split]
            if not split_results:
                continue

            for chunk_start in range(0, len(split_results), ROWS_PER_OUTPUT_SHARD):
                chunk = split_results[chunk_start:chunk_start + ROWS_PER_OUTPUT_SHARD]
                rows = {col: [] for col in OUTPUT_SCHEMA.names}
                for r in chunk:
                    for col in OUTPUT_SCHEMA.names:
                        rows[col].append(r[col])

                out_table = pa.table(rows, schema=OUTPUT_SCHEMA)
                out_path = os.path.join(
                    DST, split, f"shard_{global_shard_idx:06d}.parquet"
                )
                pq.write_table(out_table, out_path, compression="zstd")
                global_shard_idx += 1

        del all_results

    print(f"\nDone! Wrote {global_shard_idx} output shards to {DST}")


if __name__ == "__main__":
    main()
