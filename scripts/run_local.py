#!/usr/bin/env python3
"""CLI: orchestrate pipeline stages locally with multiprocessing.

Supports three stages:
  download   - Download CIFs from GCS into Parquet shards
  generate   - Generate documents from Parquet shards (no GCS needed)
  process    - Original pipeline: download + generate in one step (from manifest)
"""

import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import click

from contactdoc.config import config_sha, load_config


def _run_worker(cmd: list[str], label: str) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return f"FAIL {label}: {result.stderr[-500:]}"
    return f"OK {label}: {result.stdout.strip()}"


def _download_shard(args: tuple) -> str:
    manifest_shard, shard_index, output_dir = args
    script = str(Path(__file__).parent / "download_to_parquet.py")
    cmd = [
        sys.executable, script,
        "--manifest-shard", manifest_shard,
        "--shard-index", str(shard_index),
        "--output-dir", output_dir,
    ]
    return _run_worker(cmd, f"shard {shard_index}")


def _generate_shard(args: tuple) -> str:
    config_path, parquet_shards, shard_index, output_dir, skip_existing, scheme = args
    script = str(Path(__file__).parent / "generate_docs.py")
    cmd = [
        sys.executable, script,
        "--config", config_path,
        "--shard-index", str(shard_index),
        "--output-dir", output_dir,
        "--scheme", scheme,
    ]
    for ps in parquet_shards:
        cmd.extend(["--parquet-shard", ps])
    if skip_existing:
        cmd.append("--skip-existing")
    return _run_worker(cmd, f"shard {shard_index}")


def _process_shard(args: tuple) -> str:
    config_path, shard_path, shard_index, output_dir, use_gcs = args
    script = str(Path(__file__).parent / "process_manifest_shard.py")
    cmd = [
        sys.executable, script,
        "--config", config_path,
        "--manifest-shard", shard_path,
        "--shard-index", str(shard_index),
        "--output-dir", output_dir,
    ]
    if use_gcs:
        cmd.append("--use-gcs")
    else:
        cmd.append("--local")
    return _run_worker(cmd, f"shard {shard_index}")


def _filter_shards(all_shards, retry_from, retry_list):
    """Apply retry filters to shard list. Returns (filtered_shards, description)."""
    if retry_list is not None:
        retry_indices = set()
        with open(retry_list) as f:
            for line in f:
                line = line.strip()
                if line:
                    retry_indices.add(int(line))
        filtered = [(idx, p) for idx, p in all_shards if idx in retry_indices]
        return filtered, f"Retrying {len(filtered)} shards from {retry_list}"
    elif retry_from is not None:
        filtered = [(idx, p) for idx, p in all_shards if idx >= retry_from]
        return filtered, f"Retrying shards {retry_from}+ ({len(filtered)} shards)"
    return all_shards, f"Found {len(all_shards)} shards"


@click.command()
@click.option("--config", "config_path", default=None, help="Path to YAML config (required for generate/process)")
@click.option("--stage", type=click.Choice(["download", "generate", "process"]), default="process",
              help="Pipeline stage to run")
@click.option("--manifest-dir", default=None, help="Directory with manifest shard JSONL files (download/process)")
@click.option("--parquet-dir", default=None, help="Directory with Parquet shard files (generate)")
@click.option("--output-dir", required=True, help="Output directory")
@click.option("--use-gcs/--local", default=False, help="Download from GCS or use local paths (process only)")
@click.option("--workers", default=None, type=int, help="Number of workers (default from config or 16)")
@click.option("--retry-from", default=None, type=int, help="Only process shards with index >= this value")
@click.option("--retry-list", default=None, type=str, help="File with shard indices to retry (one per line)")
@click.option("--skip-existing", is_flag=True, default=False, help="Skip shards whose output already exists (generate only)")
@click.option("--scheme", default=None, type=str, help="Document generation scheme name (generate only, creates subdirectory)")
def main(config_path: str | None, stage: str, manifest_dir: str | None, parquet_dir: str | None,
         output_dir: str, use_gcs: bool, workers: int | None, retry_from: int | None, retry_list: str | None,
         skip_existing: bool, scheme: str | None):

    if workers is None:
        if config_path:
            cfg = load_config(config_path)
            workers = cfg.parallelism.num_workers_local
        else:
            workers = 16

    if stage == "download":
        if not manifest_dir:
            raise click.ClickException("--manifest-dir is required for download stage")
        manifest_paths = sorted(Path(manifest_dir).glob("manifest_shard_*.jsonl"))
        if not manifest_paths:
            click.echo("No manifest shards found!")
            return
        all_shards = list(enumerate(manifest_paths))
        all_shards, desc = _filter_shards(all_shards, retry_from, retry_list)
        click.echo(f"{desc}, using {workers} workers (stage=download)")
        tasks = [(str(p), idx, output_dir) for idx, p in all_shards]
        worker_fn = _download_shard

    elif stage == "generate":
        if not config_path:
            raise click.ClickException("--config is required for generate stage")
        if not parquet_dir:
            raise click.ClickException("--parquet-dir is required for generate stage")
        if not scheme:
            raise click.ClickException("--scheme is required for generate stage")
        # Recursively find parquet shards (supports subdirectories)
        parquet_paths = sorted(Path(parquet_dir).rglob("shard_*.parquet"))
        if not parquet_paths:
            click.echo("No Parquet shards found!")
            return
        effective_output_dir = str(Path(output_dir) / scheme)

        # Extract input shard indices and sort
        input_shards = []
        for p in parquet_paths:
            idx = int(p.stem.split("_")[1])
            input_shards.append((idx, p))
        input_shards.sort()

        # Group N input shards into 1 output shard (10:1 ratio)
        group_size = 10
        grouped = []  # list of (output_shard_index, [parquet_paths])
        for i in range(0, len(input_shards), group_size):
            group = input_shards[i:i + group_size]
            output_idx = i // group_size
            grouped.append((output_idx, [str(p) for _, p in group]))

        # Apply retry filters on output shard indices
        if retry_list is not None:
            retry_indices = set()
            with open(retry_list) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        retry_indices.add(int(line))
            grouped = [(idx, paths) for idx, paths in grouped if idx in retry_indices]
            desc = f"Retrying {len(grouped)} output shards from {retry_list}"
        elif retry_from is not None:
            grouped = [(idx, paths) for idx, paths in grouped if idx >= retry_from]
            desc = f"Retrying output shards {retry_from}+ ({len(grouped)} shards)"
        else:
            desc = f"Found {len(input_shards)} input shards -> {len(grouped)} output shards (group_size={group_size})"

        click.echo(f"{desc}, using {workers} workers (stage=generate, scheme={scheme}, skip_existing={skip_existing})")
        tasks = [(config_path, paths, idx, effective_output_dir, skip_existing, scheme) for idx, paths in grouped]
        worker_fn = _generate_shard

    elif stage == "process":
        if not config_path:
            raise click.ClickException("--config is required for process stage")
        if not manifest_dir:
            raise click.ClickException("--manifest-dir is required for process stage")
        manifest_paths = sorted(Path(manifest_dir).glob("manifest_shard_*.jsonl"))
        if not manifest_paths:
            click.echo("No manifest shards found!")
            return
        all_shards = list(enumerate(manifest_paths))
        all_shards, desc = _filter_shards(all_shards, retry_from, retry_list)
        click.echo(f"{desc}, using {workers} workers (stage=process)")
        tasks = [(config_path, str(p), idx, output_dir, use_gcs) for idx, p in all_shards]
        worker_fn = _process_shard

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(worker_fn, t): t for t in tasks}
        for future in as_completed(futures):
            click.echo(future.result())

    click.echo("Pipeline complete.")


if __name__ == "__main__":
    main()
