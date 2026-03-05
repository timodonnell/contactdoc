#!/usr/bin/env python3
"""CLI: config -> BigQuery -> manifest shards."""

import click

from contactdoc.afdb_query import run_selection_query
from contactdoc.clusters import load_afdb50_mapping, load_structural_mapping
from contactdoc.config import config_sha, load_config
from contactdoc.manifest import enrich_entries, write_manifest_shards


@click.command()
@click.option("--config", "config_path", required=True, help="Path to YAML config")
@click.option("--output-dir", default=None, help="Override manifest output directory")
@click.option("--limit", default=None, type=int, help="Limit BigQuery results (useful for smoke tests)")
def main(config_path: str, output_dir: str | None, limit: int | None):
    cfg = load_config(config_path)
    sha = config_sha(cfg)
    click.echo(f"Config SHA: {sha}")

    # Load both cluster files (required for clean splits)
    if not cfg.cluster_files.afdb50_rep_mem_tsv_gz:
        raise click.ClickException(
            "cluster_files.afdb50_rep_mem_tsv_gz must be set in config. "
            "Download 7-AFDB50-repId_memId.tsv.gz from "
            "https://afdb-cluster.steineggerlab.workers.dev/"
        )
    if not cfg.cluster_files.structural_rep_mem_tsv_gz:
        raise click.ClickException(
            "cluster_files.structural_rep_mem_tsv_gz must be set in config. "
            "Download 1-AFDBClusters-entryId_repId_taxId.tsv.gz from "
            "https://afdb-cluster.steineggerlab.workers.dev/"
        )

    click.echo(f"Loading AFDB50 sequence clusters from {cfg.cluster_files.afdb50_rep_mem_tsv_gz}...")
    seq_cluster_map = load_afdb50_mapping(cfg.cluster_files.afdb50_rep_mem_tsv_gz)
    click.echo(f"Loaded {len(seq_cluster_map)} sequence cluster members")

    click.echo(f"Loading structural clusters from {cfg.cluster_files.structural_rep_mem_tsv_gz}...")
    struct_cluster_map = load_structural_mapping(cfg.cluster_files.structural_rep_mem_tsv_gz)
    click.echo(f"Loaded {len(struct_cluster_map)} structural cluster members")

    # Run BigQuery selection
    click.echo("Running BigQuery selection query...")
    entries = run_selection_query(cfg, limit=limit)
    click.echo(f"Selected {len(entries)} entries from BigQuery")

    # Enrich with clusters + split (drops entries not in both cluster files)
    enriched, dropped = enrich_entries(entries, seq_cluster_map, struct_cluster_map, cfg)
    click.echo(f"Kept {len(enriched)} entries with cluster assignments, dropped {dropped} missing from cluster files")

    # Write manifest shards
    if output_dir is None:
        output_dir = f"{cfg.output_prefix}config_sha={sha}/manifests"
    paths = write_manifest_shards(enriched, output_dir, cfg.parallelism.shard_size_entries)
    click.echo(f"Wrote {len(paths)} manifest shards to {output_dir}")


if __name__ == "__main__":
    main()
