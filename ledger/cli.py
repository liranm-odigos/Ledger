"""Command line for training a tokenizer and counting tokens."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ledger.budget import SECTION_ORDER, parse_sections
from ledger.tokenizer import Tokenizer


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except BrokenPipeError:
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ledger",
        description="Train a byte-level byte-pair encoder and count tokens.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="learn merges from a corpus")
    train.add_argument("--corpus", required=True, help="UTF-8 text to train on")
    train.add_argument("--vocab-size", required=True, type=int, help="byte ids + specials + merges")
    train.add_argument("--out", required=True, help="path to write tokenizer JSON")
    train.set_defaults(handler=_train)

    count = sub.add_parser("count", help="count tokens in a string")
    count.add_argument("text", help="text to encode")
    count.add_argument("--tokenizer", required=True, help="tokenizer JSON")
    count.set_defaults(handler=_count)

    compare = sub.add_parser("compare", help="compare token counts of two files")
    compare.add_argument("left", help="first UTF-8 file")
    compare.add_argument("right", help="second UTF-8 file")
    compare.add_argument("--tokenizer", required=True, help="tokenizer JSON")
    compare.set_defaults(handler=_compare)

    budget = sub.add_parser("budget", help="count a sectioned prompt against a window")
    budget.add_argument("prompt", help="prompt file with --- section headers")
    budget.add_argument("--window", required=True, type=int, help="maximum tokens")
    budget.add_argument("--tokenizer", required=True, help="tokenizer JSON")
    budget.set_defaults(handler=_budget)
    return parser


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _train(args: argparse.Namespace) -> int:
    if args.vocab_size <= 0:
        raise ValueError("vocab-size must be positive")
    tokenizer = Tokenizer.train(_read(args.corpus), args.vocab_size)
    tokenizer.save(args.out)
    print(
        f"wrote {args.out} vocab_size={tokenizer.vocab_size} merges={len(tokenizer.merges)}"
    )
    if tokenizer.vocab_size != args.vocab_size:
        print(
            f"warning: requested vocab_size {args.vocab_size} but only "
            f"{tokenizer.vocab_size} ids were learned",
            file=sys.stderr,
        )
    return 0


def _count(args: argparse.Namespace) -> int:
    tokenizer = Tokenizer.load(args.tokenizer)
    print(tokenizer.count(args.text))
    return 0


def _compare(args: argparse.Namespace) -> int:
    tokenizer = Tokenizer.load(args.tokenizer)
    print("file\ttokens\tchars\tper_100_chars")
    for path in (args.left, args.right):
        text = _read(path)
        tokens = tokenizer.count(text)
        chars = len(text)
        rate = (100.0 * tokens / chars) if chars else 0.0
        print(f"{path}\t{tokens}\t{chars}\t{rate:.2f}")
    return 0


def _budget(args: argparse.Namespace) -> int:
    tokenizer = Tokenizer.load(args.tokenizer)
    sections = parse_sections(_read(args.prompt))
    counts = {name: tokenizer.count(sections[name]) for name in SECTION_ORDER}
    total = sum(counts.values())
    print("section\ttokens")
    for name in SECTION_ORDER:
        print(f"{name}\t{counts[name]}")
    print(f"total\t{total}")
    print(f"window\t{args.window}")
    over = total > args.window
    print(f"status\t{'over' if over else 'ok'}")
    return 1 if over else 0
