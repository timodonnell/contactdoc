#!/usr/bin/env python3
"""Generate a manifest JSONL for the Parquet dataset.

Each line describes one Parquet shard: file name, row count, file size,
and summary statistics (pLDDT range, split counts, etc.).
"""

import json
from pathlib import Path

import click
import pyarrow.parquet as pq


@click.command()
@click.option("--parquet-dir", required=True, help="Directory containing shard_*.parquet files")
@click.option("--output", required=True, help="Output manifest JSONL path")
def main(parquet_dir: str, output: str):
    parquet_path = Path(parquet_dir)
    shards = sorted(parquet_path.glob("shard_*.parquet"))
    if not shards:
        click.echo("No parquet shards found!")
        return

    click.echo(f"Scanning {len(shards)} shards...")

    total_rows = 0
    total_bytes = 0
    split_counts = {"train": 0, "val": 0, "test": 0}
    records = []

    for shard_path in shards:
        try:
            pf = pq.ParquetFile(shard_path)
        except Exception as e:
            click.echo(f"  Skipping {shard_path.name} (incomplete or corrupt: {e})")
            continue
        meta = pf.metadata
        num_rows = meta.num_rows
        file_size = shard_path.stat().st_size

        # Read just the lightweight columns for stats
        table = pq.read_table(shard_path, columns=["split", "global_plddt", "seq_len", "tax_id"])
        splits = table.column("split").to_pylist()
        plddts = table.column("global_plddt").to_pylist()
        seq_lens = table.column("seq_len").to_pylist()
        tax_ids = set(table.column("tax_id").to_pylist())

        shard_splits = {}
        for s in splits:
            shard_splits[s] = shard_splits.get(s, 0) + 1
            split_counts[s] = split_counts.get(s, 0) + 1

        record = {
            "file": shard_path.name,
            "num_rows": num_rows,
            "file_size_bytes": file_size,
            "splits": shard_splits,
            "plddt_min": round(min(plddts), 2),
            "plddt_max": round(max(plddts), 2),
            "plddt_mean": round(sum(plddts) / len(plddts), 2),
            "seq_len_min": min(seq_lens),
            "seq_len_max": max(seq_lens),
            "seq_len_mean": round(sum(seq_lens) / len(seq_lens), 1),
            "unique_tax_ids": len(tax_ids),
        }
        records.append(record)
        total_rows += num_rows
        total_bytes += file_size

        if len(records) % 100 == 0:
            click.echo(f"  Scanned {len(records)} / {len(shards)} shards ({total_rows:,} rows)")

    # Write manifest
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        # First line is a summary record
        summary = {
            "_type": "summary",
            "total_shards": len(records),
            "total_rows": total_rows,
            "total_bytes": total_bytes,
            "split_counts": split_counts,
        }
        f.write(json.dumps(summary) + "\n")
        for record in records:
            f.write(json.dumps(record) + "\n")

    click.echo(f"Done: {len(records)} shards, {total_rows:,} rows, {total_bytes / 1e9:.1f} GB")
    click.echo(f"Splits: {split_counts}")
    click.echo(f"Manifest written to {output}")


if __name__ == "__main__":
    main()
