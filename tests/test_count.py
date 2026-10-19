"""Behavior tests for Ledger's byte-level counter."""

import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

from ledger.budget import SECTION_ORDER, parse_sections
from ledger.cli import main
from ledger.tokenizer import Tokenizer, select_pair

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts" / "tokenizer.json"
PROSE = ROOT / "examples" / "prose.txt"
CODE = ROOT / "examples" / "code.py"
PROMPT = ROOT / "examples" / "prompt.txt"
UUID = "550e8400-e29b-41d4-a716-446655440000"


@pytest.fixture(scope="module")
def trained() -> Tokenizer:
    return Tokenizer.load(ARTIFACT)


def test_single_ascii_bytes_keep_their_ids(trained: Tokenizer) -> None:
    for index in range(128):
        char = chr(index)
        assert trained.encode(char) == [index]
        assert trained.decode([index]) == char


def test_unicode_round_trip(trained: Tokenizer) -> None:
    samples = [
        "",
        "token",
        "Hello, world!",
        "café",
        "naïve façade",
        "你好世界",
        "こんにちは",
        "Привет",
        "🙂",
        "👨‍💻",
        "Ωmega — em dash",
        "line1\nline2\n",
        "tab\there",
        "quotes \"\"''",
        "<|end|>",
        "before <|end|> after",
        "你好<|end|>世界<|end|>",
        "mixed café 🙂 42",
        "a\u0000b",
    ]
    for text in samples:
        assert trained.decode(trained.encode(text)) == text


def test_random_unicode_round_trip(trained: Tokenizer) -> None:
    rng = random.Random(0)
    alphabet = list("abcé你🙂 \n—Ω")
    for length in (1, 2, 7, 40, 200):
        chars: list[str] = []
        for _ in range(length):
            chars.append("<|end|>" if rng.random() < 0.1 else rng.choice(alphabet))
        text = "".join(chars)
        assert trained.decode(trained.encode(text)) == text


def test_special_token_is_one_id_and_blocks_merges(trained: Tokenizer) -> None:
    special = trained.special_ids["<|end|>"]
    assert special == 256
    assert trained.encode("<|end|>") == [special]
    assert trained.encode("x<|end|>y") == trained.encode("x") + [special] + trained.encode("y")
    assert trained.decode([special]) == "<|end|>"


def test_ab_merges_first_in_ababab() -> None:
    tokenizer = Tokenizer.train("ababab", vocab_size=258)
    assert tokenizer.merges[0] == (ord("a"), ord("b"))
    assert tokenizer.encode("ab") == [257]
    assert tokenizer.encode("ababab") == [257, 257, 257]
    assert tokenizer.decode(tokenizer.encode("ababab")) == "ababab"


def test_frequency_beats_a_smaller_pair() -> None:
    tokenizer = Tokenizer.train("bbbbbaaaa", vocab_size=258, special_tokens=[])
    assert tokenizer.merges[0] == (ord("b"), ord("b"))


def test_tie_breaks_to_the_smaller_pair() -> None:
    # cd, da, and ab each occur once. ab is the smaller pair.
    tied = Tokenizer.train("cdab", vocab_size=258)
    assert tied.merges[0] == (ord("a"), ord("b"))

    # aa and bb occur equally often. aa is smaller.
    both = Tokenizer.train("aaaabbbb", vocab_size=258, special_tokens=[])
    assert both.merges[0] == (ord("a"), ord("a"))


def test_tie_break_compares_token_bytes_not_ids() -> None:
    stats = {(256, 10): 3, (97, 98): 3}
    id_to_bytes = {256: b"\x01\x02", 10: b"\n", 97: b"a", 98: b"b"}
    assert select_pair(stats, id_to_bytes) == (256, 10)


def test_overlapping_pair_merges_left_to_right() -> None:
    tokenizer = Tokenizer.train("aaaa", vocab_size=257, special_tokens=[])
    assert tokenizer.merges == [(ord("a"), ord("a"))]
    pair = 256
    assert tokenizer.encode("aa") == [pair]
    assert tokenizer.encode("aaa") == [pair, ord("a")]
    assert tokenizer.encode("aaaa") == [pair, pair]


def test_chained_merges_round_trip() -> None:
    tokenizer = Tokenizer.train("ababab", vocab_size=300, special_tokens=[])
    assert tokenizer.merges[0] == (ord("a"), ord("b"))
    assert tokenizer.merges[1] == (256, 256)
    assert tokenizer.decode(tokenizer.encode("ababab")) == "ababab"
    assert tokenizer.encode("abab") == [257]


def test_training_does_not_merge_across_special_tokens() -> None:
    text = "ab<|end|>ab"
    tokenizer = Tokenizer.train(text, vocab_size=258)
    assert tokenizer.merges[0] == (ord("a"), ord("b"))
    assert tokenizer.encode(text) == [257, 256, 257]


def test_save_load_preserves_encoding(tmp_path: Path) -> None:
    tokenizer = Tokenizer.train("hello hello there 你好", vocab_size=280)
    path = tmp_path / "tok.json"
    tokenizer.save(path)
    loaded = Tokenizer.load(path)
    sample = "hello there 你好 <|end|>"
    assert loaded.merges == tokenizer.merges
    assert loaded.encode(sample) == tokenizer.encode(sample)
    assert loaded.decode(loaded.encode(sample)) == sample


def test_artifact_vocab_and_merge_count(trained: Tokenizer) -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert payload["special_tokens"] == ["<|end|>"]
    assert len(payload["merges"]) == 255
    assert payload["vocab_size"] == 512
    assert trained.vocab_size == 512
    assert trained.merges[0][0] < 256 and trained.merges[0][1] < 256


def test_uuid_costs_more_tokens_than_the_word_token(trained: Tokenizer) -> None:
    word = trained.count("token")
    uuid = trained.count(UUID)
    assert word < uuid
    assert trained.decode(trained.encode(UUID)) == UUID


def test_code_costs_more_tokens_than_prose_at_similar_length(trained: Tokenizer) -> None:
    prose = PROSE.read_text(encoding="utf-8")
    code = CODE.read_text(encoding="utf-8")
    assert abs(len(prose) - len(code)) / max(len(prose), len(code)) <= 0.02
    assert trained.count(code) > trained.count(prose)


def test_budget_sections_drop_headers_and_do_not_cross() -> None:
    text = (
        "preamble that is ignored\n"
        "--- system\n"
        "Hi\n"
        "--- history\n"
        "--- retrieved\n"
        "notes\n"
        "--- question\n"
        "Why?\n"
    )
    sections = parse_sections(text)
    assert sections["system"] == "Hi\n"
    assert sections["history"] == ""
    assert sections["retrieved"] == "notes\n"
    assert sections["question"] == "Why?\n"
    assert all(not body.startswith("---") for body in sections.values())

    # A merge that would apply across a section boundary must not apply
    # when the sections are counted separately.
    tokenizer = Tokenizer.train("\nb" * 30, vocab_size=257, special_tokens=[])
    assert tokenizer.merges[0] == (ord("\n"), ord("b"))
    prompt = "--- system\n\n--- history\nb\n--- retrieved\n--- question\n"
    sections = parse_sections(prompt)
    separate = sum(tokenizer.count(sections[name]) for name in SECTION_ORDER)
    joined = tokenizer.count("".join(sections[name] for name in SECTION_ORDER))
    assert separate > joined


def test_cli_train_count_and_budget(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("ababab", encoding="utf-8")
    out = tmp_path / "tok.json"
    assert main(["train", "--corpus", str(corpus), "--vocab-size", "258", "--out", str(out)]) == 0
    tokenizer = Tokenizer.load(out)
    assert tokenizer.merges[0] == (ord("a"), ord("b"))

    count = subprocess.run(
        [sys.executable, "-m", "ledger", "count", "ab", "--tokenizer", str(out)],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert count.stdout.strip() == "1"

    prompt = tmp_path / "prompt.txt"
    prompt.write_text(
        "--- system\nab\n--- history\n\n--- retrieved\n\n--- question\nabab\n",
        encoding="utf-8",
    )
    within = subprocess.run(
        [
            sys.executable,
            "-m",
            "ledger",
            "budget",
            str(prompt),
            "--window",
            "100",
            "--tokenizer",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert within.returncode == 0
    assert "status\tok" in within.stdout
    rows = dict(
        line.split("\t")
        for line in within.stdout.splitlines()
        if "\t" in line and not line.startswith("section\t")
    )
    assert rows["system"] == "2"  # "ab" plus the newline
    assert rows["question"] == "3"  # "abab" plus the newline
    total = int(rows["total"])
    assert int(rows["window"]) == 100

    over = subprocess.run(
        [
            sys.executable,
            "-m",
            "ledger",
            "budget",
            str(prompt),
            "--window",
            str(total),
            "--tokenizer",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert over.returncode == 0

    exceeded = subprocess.run(
        [
            sys.executable,
            "-m",
            "ledger",
            "budget",
            str(prompt),
            "--window",
            str(total - 1),
            "--tokenizer",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert exceeded.returncode == 1
    assert "status\tover" in exceeded.stdout


def test_cli_compare_and_shipped_budget(trained: Tokenizer) -> None:
    compared = subprocess.run(
        [
            sys.executable,
            "-m",
            "ledger",
            "compare",
            str(PROSE),
            str(CODE),
            "--tokenizer",
            str(ARTIFACT),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    lines = [line.split("\t") for line in compared.stdout.splitlines()]
    assert lines[0] == ["file", "tokens", "chars", "per_100_chars"]
    assert int(lines[1][1]) == trained.count(PROSE.read_text(encoding="utf-8"))
    assert int(lines[2][1]) == trained.count(CODE.read_text(encoding="utf-8"))
    assert int(lines[2][1]) > int(lines[1][1])

    counted = subprocess.run(
        [sys.executable, "-m", "ledger", "count", "token", "--tokenizer", str(ARTIFACT)],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert int(counted.stdout.strip()) == trained.count("token")

    budget = subprocess.run(
        [
            sys.executable,
            "-m",
            "ledger",
            "budget",
            str(PROMPT),
            "--window",
            "8192",
            "--tokenizer",
            str(ARTIFACT),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert budget.returncode == 0
    sections = parse_sections(PROMPT.read_text(encoding="utf-8"))
    expected = sum(trained.count(sections[name]) for name in SECTION_ORDER)
    assert f"total\t{expected}" in budget.stdout
    assert expected <= 8192

    tight = subprocess.run(
        [
            sys.executable,
            "-m",
            "ledger",
            "budget",
            str(PROMPT),
            "--window",
            str(expected - 1),
            "--tokenizer",
            str(ARTIFACT),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert tight.returncode == 1
