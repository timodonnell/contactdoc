#!/usr/bin/env python3
"""Create the fixed ContactDoc tokenizer as a Hugging Face fast tokenizer.

This mirrors the deterministic vocabulary in ``contactdoc/tokenizer.py``.
"""

from __future__ import annotations

import argparse
import json
import os

from huggingface_hub import HfApi, upload_folder
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from contactdoc.tokenizer import ATOM_NAMES, CONTROL_TOKENS, MAX_POSITION, RESIDUE_NAMES, TASK_TOKENS


def _build_vocab() -> dict[str, int]:
    tokens: list[str] = []
    tokens.extend(CONTROL_TOKENS)
    tokens.extend(f"<{name}>" for name in TASK_TOKENS)
    tokens.extend(f"<{name}>" for name in RESIDUE_NAMES)
    tokens.extend(f"<{name}>" for name in ATOM_NAMES)
    tokens.extend(f"<p{i}>" for i in range(1, MAX_POSITION + 1))
    return {token: idx for idx, token in enumerate(tokens)}


def _build_hf_tokenizer() -> PreTrainedTokenizerFast:
    vocab = _build_vocab()
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<UNK>"))
    tokenizer.normalizer = normalizers.Sequence([normalizers.Replace("\n", " <newline> "), normalizers.Strip()])
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<UNK>",
        pad_token="<pad>",
        bos_token="<begin_sequence>",
        eos_token="<end>",
    )
    return hf_tokenizer


def _patch_tokenizer_config(output_path: str) -> None:
    """Force a tokenizer_class that AutoTokenizer can import everywhere."""
    tokenizer_config_path = os.path.join(output_path, "tokenizer_config.json")
    with open(tokenizer_config_path) as handle:
        config = json.load(handle)
    config["tokenizer_class"] = "PreTrainedTokenizerFast"
    config.pop("extra_special_tokens", None)
    with open(tokenizer_config_path, "w") as handle:
        json.dump(config, handle, indent=2)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the fixed ContactDoc tokenizer.")
    parser.add_argument("--output-path", required=True, help="Directory where tokenizer files will be written.")
    parser.add_argument(
        "--push-to-hub",
        default=None,
        help="Optional HF model repo ID (e.g. WillHeld/contactdoc-tokenizer).",
    )
    parser.add_argument("--private", action="store_true", help="Create/push private repo when --push-to-hub is set.")
    parser.add_argument("--token", default=None, help="Optional HF token. Defaults to HF_TOKEN env var if set.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    tokenizer = _build_hf_tokenizer()
    tokenizer.save_pretrained(args.output_path)
    _patch_tokenizer_config(args.output_path)
    if args.push_to_hub:
        api = HfApi(token=args.token)
        api.create_repo(repo_id=args.push_to_hub, repo_type="model", private=args.private, exist_ok=True)
        upload_folder(
            repo_id=args.push_to_hub,
            repo_type="model",
            folder_path=args.output_path,
            token=args.token,
            commit_message="Upload ContactDoc fixed-vocab tokenizer",
        )


if __name__ == "__main__":
    main()
