#!/usr/bin/env python3
"""Upload Parquet dataset to HuggingFace Hub.

Uploads shards in parallel using the HuggingFace Hub API.
Supports resuming — only uploads files not already present on the Hub.
"""

from pathlib import Path

import click
from huggingface_hub import HfApi


@click.command()
@click.option("--parquet-dir", required=True, help="Directory containing shard_*.parquet files")
@click.option("--manifest", default=None, help="Path to manifest.jsonl (uploaded alongside shards)")
@click.option("--dataset-card", default=None, help="Path to DATASET_CARD.md (uploaded as README.md)")
@click.option("--repo-id", required=True, help="HuggingFace dataset repo ID (e.g. timodonnell/afdb-structures)")
@click.option("--path-in-repo", default="data", help="Directory in repo for parquet files")
@click.option("--revision", default="main", help="Branch to upload to")
def main(parquet_dir: str, manifest: str | None, dataset_card: str | None,
         repo_id: str, path_in_repo: str, revision: str):
    api = HfApi()

    # Upload dataset card as README.md
    if dataset_card:
        click.echo(f"Uploading dataset card as README.md...")
        api.upload_file(
            path_or_fileobj=dataset_card,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
        )
        click.echo("Dataset card uploaded.")

    # Upload manifest
    if manifest:
        click.echo(f"Uploading manifest...")
        api.upload_file(
            path_or_fileobj=manifest,
            path_in_repo=f"{path_in_repo}/manifest.jsonl",
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
        )
        click.echo("Manifest uploaded.")

    # Upload parquet shards using upload_large_folder for resilience
    parquet_path = Path(parquet_dir)
    shards = sorted(parquet_path.glob("shard_*.parquet"))
    error_files = sorted(parquet_path.glob("shard_*.errors.jsonl"))
    all_files = shards + error_files

    if not all_files:
        click.echo("No files to upload!")
        return

    click.echo(f"Uploading {len(shards)} parquet shards and {len(error_files)} error files to {repo_id}...")
    click.echo(f"Target: {path_in_repo}/")

    # Use upload_large_folder which handles resumption, chunked uploads, and parallelism
    api.upload_large_folder(
        folder_path=parquet_dir,
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=["shard_*.parquet", "shard_*.errors.jsonl"],
        revision=revision,
    )

    click.echo(f"Upload complete: {repo_id}")


if __name__ == "__main__":
    main()
