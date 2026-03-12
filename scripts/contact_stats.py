#!/usr/bin/env python3
"""Count contacts by (res_name_i, res_name_j, seq_separation, atom_i, atom_j).

Reads parquet shards, parses CIFs, computes contacts, and accumulates counts
across the entire dataset. Outputs a CSV sorted by descending count.
"""

import collections
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import click
import pyarrow.parquet as pq

from contactdoc.cif_parse import extract_residues, parse_cif
from contactdoc.config import load_config
from contactdoc.contacts import compute_contacts, filter_contacts_by_plddt


def _process_shard(args: tuple) -> tuple[dict[tuple, int], int, int]:
    """Process one parquet shard. Returns (counter, success_count, error_count)."""
    parquet_path, cfg_path = args
    from contactdoc.config import load_config as _load
    cfg = _load(cfg_path)

    counter: dict[tuple, int] = {}
    success = 0
    errors = 0

    table = pq.read_table(parquet_path)
    for i in range(len(table)):
        cif_content = table.column("cif_content")[i].as_py()
        try:
            structure = parse_cif(cif_content)
            result = extract_residues(
                structure,
                require_single_chain=cfg.filters.require_single_chain,
                canonical_residue_policy=cfg.filters.canonical_residue_policy,
            )
            if isinstance(result, str):
                errors += 1
                continue

            contacts = compute_contacts(result, cfg.contacts.cutoff_angstrom)
            contacts = filter_contacts_by_plddt(contacts, result, cfg.filters.residue_plddt_min)

            # Build index -> residue name lookup
            res_names = {r.index: r.name for r in result.residues}

            for c in contacts:
                key = (res_names[c.i], res_names[c.j], c.j - c.i, c.atom_i, c.atom_j)
                counter[key] = counter.get(key, 0) + 1

            success += 1
        except Exception:
            traceback.print_exc()
            errors += 1

    return counter, success, errors


@click.command()
@click.option("--config", "config_path", required=True, help="Path to YAML config")
@click.option("--parquet-dir", required=True, help="Directory with Parquet shards (searched recursively)")
@click.option("--output", required=True, help="Output CSV path")
@click.option("--workers", default=16, type=int, help="Number of parallel workers")
def main(config_path: str, parquet_dir: str, output: str, workers: int):
    parquet_paths = sorted(Path(parquet_dir).rglob("shard_*.parquet"))
    if not parquet_paths:
        click.echo("No parquet shards found!")
        return

    click.echo(f"Found {len(parquet_paths)} shards, using {workers} workers")

    tasks = [(str(p), config_path) for p in parquet_paths]
    total_counter: dict[tuple, int] = {}
    total_success = 0
    total_errors = 0
    done = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_process_shard, t): t for t in tasks}
        for future in as_completed(futures):
            counter, success, errors = future.result()
            for key, count in counter.items():
                total_counter[key] = total_counter.get(key, 0) + count
            total_success += success
            total_errors += errors
            done += 1
            if done % 100 == 0:
                click.echo(f"  {done}/{len(tasks)} shards, {total_success} proteins, {len(total_counter)} unique tuples")

    click.echo(f"Done: {total_success} proteins, {total_errors} errors, {len(total_counter)} unique contact tuples")

    # Write CSV sorted by descending count
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_items = sorted(total_counter.items(), key=lambda x: -x[1])

    with open(out_path, "w") as f:
        f.write("res1,res2,seq_separation,atom1,atom2,count\n")
        for (res1, res2, sep, atom1, atom2), count in sorted_items:
            f.write(f"{res1},{res2},{sep},{atom1},{atom2},{count}\n")

    click.echo(f"Wrote {len(sorted_items)} rows to {output}")


if __name__ == "__main__":
    main()
