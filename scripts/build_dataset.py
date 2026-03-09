#!/usr/bin/env python3
"""Convert sharded txt.gz + metadata.jsonl.gz output into a HuggingFace datasets Arrow format.

Produces one Arrow dataset per split with columns:
  - input_ids: list[int]    (tokenized document)
  - entry_id: string
  - seq_len: int
  - contacts_emitted: int
  - global_plddt: float

Uses a generator-based approach to avoid loading all documents into memory.
"""

import gzip
import json
from pathlib import Path

import click
import datasets

from contactdoc.tokenizer import encode, VOCAB_SIZE


def _truncate_contacts(doc_text: str, max_contacts: int) -> tuple[str, int]:
    """Truncate contacts section to at most max_contacts lines.

    When truncated, the trailing <end_contacts> and <end> tokens are omitted
    so the model learns to generate contacts until the context window is full.

    Returns (truncated_text, actual_contact_count).
    """
    lines = doc_text.strip().split("\n")
    out = []
    contact_count = 0
    total_contacts = 0
    in_contacts = False
    truncated = False
    for line in lines:
        if line.strip() == "<begin_contacts>":
            in_contacts = True
            out.append(line)
        elif line.strip() == "<end_contacts>":
            in_contacts = False
            if not truncated:
                out.append(line)
        elif line.strip() == "<end>":
            if not truncated:
                out.append(line)
        elif in_contacts:
            total_contacts += 1
            if contact_count < max_contacts:
                out.append(line)
                contact_count += 1
            if total_contacts > max_contacts:
                truncated = True
        else:
            out.append(line)
    return "\n".join(out) + "\n", contact_count


def _iter_shard_rows(txt_gz_path: Path, meta_gz_path: Path, max_contacts_ratio: float | None = None):
    """Yield tokenized rows from one shard, streaming docs one at a time."""
    with gzip.open(meta_gz_path, "rt") as mf:
        meta_records = []
        for line in mf:
            line = line.strip()
            if line:
                meta_records.append(json.loads(line))

    with gzip.open(txt_gz_path, "rt") as f:
        current = []
        doc_idx = 0
        for line in f:
            current.append(line.rstrip("\n"))
            if line.strip() == "<end>":
                doc_text = "\n".join(current) + "\n"
                current = []
                meta = meta_records[doc_idx]
                doc_idx += 1
                seq_len = meta["seq_len"]
                contacts_emitted = meta["contacts_emitted"]
                if max_contacts_ratio is not None:
                    max_contacts = int(max_contacts_ratio * seq_len)
                    doc_text, contacts_emitted = _truncate_contacts(doc_text, max_contacts)
                yield {
                    "input_ids": encode(doc_text),
                    "entry_id": meta["entryId"],
                    "seq_len": seq_len,
                    "contacts_emitted": contacts_emitted,
                    "global_plddt": meta["globalMetricValue_mean_pLDDT"],
                }


def _make_split_generator(txt_files, meta_files, max_contacts_ratio: float | None = None):
    """Return a generator function (not a generator) for Dataset.from_generator."""
    def gen():
        for txt_gz, meta_gz in zip(txt_files, meta_files):
            yield from _iter_shard_rows(txt_gz, meta_gz, max_contacts_ratio)
    return gen


@click.command()
@click.option("--input-dir", required=True, help="Directory with split=*/shard=*.txt.gz output")
@click.option("--output-dir", required=True, help="Directory to write HF datasets (one subdir per split)")
@click.option("--num-proc", default=1, type=int, help="Parallel workers for Arrow writing")
@click.option("--max-contacts-ratio", default=None, type=float, help="Limit contacts to ratio * seq_len (e.g. 1.0)")
def main(input_dir: str, output_dir: str, num_proc: int, max_contacts_ratio: float | None):
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    click.echo(f"Vocab size: {VOCAB_SIZE}")
    if max_contacts_ratio is not None:
        click.echo(f"Max contacts ratio: {max_contacts_ratio}x seq_len")

    features = datasets.Features({
        "input_ids": datasets.Sequence(datasets.Value("int32")),
        "entry_id": datasets.Value("string"),
        "seq_len": datasets.Value("int32"),
        "contacts_emitted": datasets.Value("int32"),
        "global_plddt": datasets.Value("float32"),
    })

    for split_dir in sorted(input_path.glob("split=*")):
        split_name = split_dir.name.split("=")[1]
        txt_files = sorted(split_dir.glob("shard=*.txt.gz"))
        meta_files = sorted(split_dir.glob("shard=*.metadata.jsonl.gz"))

        if not txt_files:
            continue

        click.echo(f"Processing split={split_name}: {len(txt_files)} shards")

        gen_fn = _make_split_generator(txt_files, meta_files, max_contacts_ratio)
        ds = datasets.Dataset.from_generator(gen_fn, features=features)

        split_output = output_path / split_name
        ds.save_to_disk(str(split_output), num_proc=num_proc)
        click.echo(f"  Saved to {split_output} ({len(ds)} rows, {ds.data.nbytes / 1e6:.1f} MB)")

    click.echo("Done.")


if __name__ == "__main__":
    main()
