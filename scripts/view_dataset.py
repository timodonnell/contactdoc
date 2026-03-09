#!/usr/bin/env python3
"""View decoded documents from a tokenized HuggingFace Arrow dataset.

Usage:
  uv run python scripts/view_dataset.py --dataset-dir /data/tim/contactdoc/dataset_partial/train
  uv run python scripts/view_dataset.py --dataset-dir /data/tim/contactdoc/dataset_partial/train --index 42
  uv run python scripts/view_dataset.py --dataset-dir /data/tim/contactdoc/dataset_partial/train --random 5
"""

import random

import click
import datasets

from contactdoc.tokenizer import decode


def _print_doc(idx: int, row: dict):
    text = decode(row["input_ids"])
    n_tokens = len(row["input_ids"])
    lines = text.strip().split("\n")
    n_contacts = sum(1 for l in lines if l.startswith("<p"))
    click.echo(f"--- Document {idx} ---")
    click.echo(f"entry_id: {row['entry_id']}  seq_len: {row['seq_len']}  "
               f"contacts: {row['contacts_emitted']}  plddt: {row['global_plddt']:.1f}  "
               f"tokens: {n_tokens}")
    click.echo()
    click.echo(text)


@click.command()
@click.option("--dataset-dir", required=True, help="Path to a split directory (e.g. dataset/train)")
@click.option("--index", default=None, type=int, help="Show a specific document by index")
@click.option("--random", "n_random", default=None, type=int, help="Show N random documents")
@click.option("--head", "n_head", default=None, type=int, help="Show the first N documents")
def main(dataset_dir: str, index: int | None, n_random: int | None, n_head: int | None):
    ds = datasets.load_from_disk(dataset_dir)
    click.echo(f"Loaded {len(ds)} documents from {dataset_dir}")
    click.echo()

    if index is not None:
        _print_doc(index, ds[index])
    elif n_random is not None:
        indices = random.sample(range(len(ds)), min(n_random, len(ds)))
        for i in indices:
            _print_doc(i, ds[i])
    elif n_head is not None:
        for i in range(min(n_head, len(ds))):
            _print_doc(i, ds[i])
    else:
        # Default: show first document
        _print_doc(0, ds[0])


if __name__ == "__main__":
    main()
