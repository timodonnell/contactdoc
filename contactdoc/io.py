"""I/O utilities: write gzipped shards for text, metadata JSONL, and error JSONL."""

import gzip
import os
from pathlib import Path


class ShardWriter:
    """Accumulates documents/records per split, writes gzipped shard files."""

    def __init__(self, output_dir: str | Path, shard_index: int):
        self.output_dir = Path(output_dir)
        self.shard_index = shard_index
        self.shard_name = f"shard={shard_index:06d}"
        # Buffers: split -> list of strings
        self._text_buffers: dict[str, list[str]] = {}
        self._meta_buffers: dict[str, list[str]] = {}
        self._error_buffers: dict[str, list[str]] = {}

    def add_document(self, split: str, doc_text: str, metadata_jsonl: str):
        buf = self._text_buffers.setdefault(split, [])
        if buf:
            buf.append("<end_of_document>\n")
        buf.append(doc_text)
        self._meta_buffers.setdefault(split, []).append(metadata_jsonl + "\n")

    def add_error(self, split: str, error_jsonl: str):
        self._error_buffers.setdefault(split, []).append(error_jsonl + "\n")

    def flush(self) -> list[str]:
        """Write all buffers to disk. Returns list of written file paths."""
        written = []
        all_splits = set(self._text_buffers) | set(self._error_buffers)
        for split in sorted(all_splits):
            split_dir = self.output_dir / f"split={split}"
            split_dir.mkdir(parents=True, exist_ok=True)

            if split in self._text_buffers and self._text_buffers[split]:
                path = split_dir / f"{self.shard_name}.txt.gz"
                _write_gz(path, "".join(self._text_buffers[split]))
                written.append(str(path))

            if split in self._meta_buffers and self._meta_buffers[split]:
                path = split_dir / f"{self.shard_name}.metadata.jsonl.gz"
                _write_gz(path, "".join(self._meta_buffers[split]))
                written.append(str(path))

            if split in self._error_buffers and self._error_buffers[split]:
                path = split_dir / f"{self.shard_name}.errors.jsonl.gz"
                _write_gz(path, "".join(self._error_buffers[split]))
                written.append(str(path))

        return written


def _write_gz(path: Path, content: str):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(content)


def read_gz(path: str | Path) -> str:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return f.read()
