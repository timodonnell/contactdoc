#!/usr/bin/env python3
"""CLI: generate ContactDoc documents from one or more Parquet shards.

Uses the generator registry to select a document generation scheme.
Multiple input shards can be merged into a single output shard.
"""

import traceback

import click
import pyarrow.parquet as pq

from contactdoc.cif_parse import extract_residues, parse_cif
from contactdoc.config import load_config
from contactdoc.generators import get_generator
from contactdoc.io import ShardWriter
from contactdoc.serialize import (
    make_error_record,
    make_metadata_record,
    metadata_to_jsonl,
    error_to_jsonl,
)


def process_row(row: dict, cfg, generator):
    """Process a single Parquet row.

    Returns (doc_text, metadata_jsonl, error_jsonl, split).
    """
    split = row["split"]
    entry_id = row["entry_id"]
    entry = {
        "entryId": entry_id,
        "uniprotAccession": row.get("uniprot_accession", ""),
        "taxId": row.get("tax_id", 0),
        "organismScientificName": row.get("organism_name", ""),
        "globalMetricValue": row.get("global_plddt", 0.0),
        "seq_len": row.get("seq_len", 0),
        "uniprotStart": 0,
        "uniprotEnd": 0,
        "latestVersion": 0,
        "gcs_uri": row.get("gcs_uri", ""),
        "split": split,
        "seq_cluster_id": row.get("seq_cluster_id", ""),
        "struct_cluster_id": row.get("struct_cluster_id", ""),
        "split_cluster_id": row.get("struct_cluster_id", ""),
    }

    try:
        structure = parse_cif(row["cif_content"])
        result = extract_residues(
            structure,
            require_single_chain=cfg.filters.require_single_chain,
            canonical_residue_policy=cfg.filters.canonical_residue_policy,
        )
        if isinstance(result, str):
            err = make_error_record(entry, result)
            return None, None, error_to_jsonl(err), split

        gen_result = generator.generate(result, cfg)
        if isinstance(gen_result, str):
            # Error reason string
            err = make_error_record(entry, gen_result)
            return None, None, error_to_jsonl(err), split

        meta = make_metadata_record(
            entry, result, gen_result.contacts_pre_filter,
            gen_result.contacts_emitted, gen_result.doc_text, cfg,
        )
        return gen_result.doc_text, metadata_to_jsonl(meta), None, split

    except Exception:
        err = make_error_record(entry, "exception", traceback.format_exc())
        return None, None, error_to_jsonl(err), split


def shard_already_generated(output_dir: str, shard_index: int) -> bool:
    """Check if this shard has already been generated (any split output exists)."""
    from pathlib import Path
    output_path = Path(output_dir)
    shard_name = f"shard={shard_index:06d}"
    for split in ("train", "val", "test"):
        split_dir = output_path / f"split={split}"
        if (split_dir / f"{shard_name}.txt.gz").exists():
            return True
        if (split_dir / f"{shard_name}.errors.jsonl.gz").exists():
            return True
    return False


@click.command()
@click.option("--config", "config_path", required=True, help="Path to YAML config")
@click.option("--parquet-shard", "parquet_shards", required=True, multiple=True,
              help="Path(s) to Parquet shard file(s). Can be specified multiple times.")
@click.option("--shard-index", required=True, type=int, help="Output shard index")
@click.option("--output-dir", required=True, help="Output directory for doc shards")
@click.option("--scheme", required=True, help="Document generation scheme name")
@click.option("--skip-existing", is_flag=True, default=False, help="Skip if output already exists")
def main(config_path: str, parquet_shards: tuple[str, ...], shard_index: int,
         output_dir: str, scheme: str, skip_existing: bool):
    cfg = load_config(config_path)
    generator = get_generator(scheme)

    if skip_existing and shard_already_generated(output_dir, shard_index):
        click.echo(f"Skipping shard {shard_index} (output already exists)")
        return

    total_entries = 0
    all_rows = []
    for parquet_shard in parquet_shards:
        table = pq.read_table(parquet_shard)
        for i in range(len(table)):
            all_rows.append({col: table.column(col)[i].as_py() for col in table.column_names})
        total_entries += len(table)

    click.echo(f"Processing output shard {shard_index}: {total_entries} entries from {len(parquet_shards)} input shard(s), scheme={scheme}")

    writer = ShardWriter(output_dir, shard_index)
    success_count = 0
    error_count = 0

    for row in all_rows:
        doc_text, meta_jsonl, err_jsonl, split = process_row(row, cfg, generator)
        if doc_text is not None:
            writer.add_document(split, doc_text, meta_jsonl)
            success_count += 1
        if err_jsonl is not None:
            writer.add_error(split, err_jsonl)
            error_count += 1

    paths = writer.flush()
    click.echo(f"Done: {success_count} docs, {error_count} errors, {len(paths)} files written")


if __name__ == "__main__":
    main()
