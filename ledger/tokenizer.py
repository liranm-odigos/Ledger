"""Byte-level byte-pair encoder.

Token ids 0 through 255 are the UTF-8 bytes with the same values. The next
ids are special tokens, in declaration order. Every id after that is a merge,
in the order it was learned.

Training repeatedly merges the most frequent adjacent pair. When several pairs
share that frequency, the lexicographically smaller pair wins. Pair order is
the order of the token byte strings, first token then second, not the numeric
token ids. That matters once a merge has produced a multi-byte token whose id
is large but whose bytes are still small.

`<|end|>` is a special token: the encoder emits one id for the whole string
and never merges across it. `decode(encode(text))` returns `text`.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

BYTE_VOCAB = 256
DEFAULT_SPECIAL_TOKENS = ("<|end|>",)
_VERSION = 1


def select_pair(
    stats: dict[tuple[int, int], int],
    id_to_bytes: dict[int, bytes],
) -> tuple[int, int]:
    """Pick the merge: highest count, then the smaller pair of token bytes."""

    def sort_key(pair: tuple[int, int]) -> tuple:
        left, right = pair
        return (-stats[pair], id_to_bytes[left], id_to_bytes[right])

    return min(stats, key=sort_key)


def _apply_merge(ids: list[int], left: int, right: int, new_id: int) -> list[int]:
    """Replace non-overlapping left-to-right occurrences of (left, right)."""
    n = len(ids)
    index = 0
    while index + 1 < n and not (ids[index] == left and ids[index + 1] == right):
        index += 1
    if index + 1 >= n:
        return ids
    merged = ids[:index]
    merged.append(new_id)
    index += 2
    while index < n:
        if index + 1 < n and ids[index] == left and ids[index + 1] == right:
            merged.append(new_id)
            index += 2
        else:
            merged.append(ids[index])
            index += 1
    return merged


class Tokenizer:
    """A trained byte-level vocabulary."""

    def __init__(
        self,
        merges: list[tuple[int, int]],
        special_tokens: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        specials = list(DEFAULT_SPECIAL_TOKENS if special_tokens is None else special_tokens)
        self._validate_specials(specials)
        self.special_tokens = specials
        self.merges = [(int(left), int(right)) for left, right in merges]

        self.id_to_bytes: dict[int, bytes] = {index: bytes([index]) for index in range(BYTE_VOCAB)}
        self.special_ids: dict[str, int] = {}
        next_id = BYTE_VOCAB
        for token in self.special_tokens:
            self.special_ids[token] = next_id
            self.id_to_bytes[next_id] = token.encode("utf-8")
            next_id += 1

        self._merge_table: list[tuple[int, int, int]] = []
        for left, right in self.merges:
            if left not in self.id_to_bytes or right not in self.id_to_bytes:
                raise ValueError(f"merge ({left}, {right}) refers to an unknown token")
            self.id_to_bytes[next_id] = self.id_to_bytes[left] + self.id_to_bytes[right]
            self._merge_table.append((left, right, next_id))
            next_id += 1

        self.vocab_size = next_id
        self._special_pattern = _compile_specials(self.special_tokens)

    @staticmethod
    def _validate_specials(specials: list[str]) -> None:
        if any(not isinstance(token, str) or token == "" for token in specials):
            raise ValueError("special tokens must be non-empty strings")
        if len(set(specials)) != len(specials):
            raise ValueError("special tokens must be unique")

    @classmethod
    def train(
        cls,
        text: str,
        vocab_size: int,
        special_tokens: list[str] | tuple[str, ...] | None = None,
    ) -> Tokenizer:
        """Learn merges from `text` until `vocab_size` ids exist or pairs run out.

        `vocab_size` counts byte ids, special tokens, and merges together.
        Training stops early when every remaining segment is shorter than two
        tokens; the returned vocabulary is then smaller than requested.
        """
        specials = list(DEFAULT_SPECIAL_TOKENS if special_tokens is None else special_tokens)
        cls._validate_specials(specials)
        minimum = BYTE_VOCAB + len(specials)
        if vocab_size < minimum:
            raise ValueError(
                f"vocab_size {vocab_size} is below {minimum}, the size of the "
                f"{BYTE_VOCAB} byte ids plus {len(specials)} special tokens"
            )

        id_to_bytes: dict[int, bytes] = {index: bytes([index]) for index in range(BYTE_VOCAB)}
        next_id = BYTE_VOCAB
        for token in specials:
            id_to_bytes[next_id] = token.encode("utf-8")
            next_id += 1

        segments = [
            list(part.encode("utf-8"))
            for part in _split_specials(text, specials)
            if part
        ]
        merges: list[tuple[int, int]] = []
        target_merges = vocab_size - minimum
        while len(merges) < target_merges:
            stats: Counter[tuple[int, int]] = Counter()
            for segment in segments:
                stats.update(zip(segment, segment[1:]))
            if not stats:
                break
            left, right = select_pair(stats, id_to_bytes)
            new_id = next_id
            next_id += 1
            id_to_bytes[new_id] = id_to_bytes[left] + id_to_bytes[right]
            merges.append((left, right))
            segments = [_apply_merge(segment, left, right, new_id) for segment in segments]
        return cls(merges, specials)

    def encode(self, text: str) -> list[int]:
        """Encode `text` to token ids. Special tokens are a single id each."""
        if not text:
            return []
        ids: list[int] = []
        for kind, chunk in self._partition(text):
            if kind == "special":
                ids.append(self.special_ids[chunk])
            elif chunk:
                ids.extend(self._encode_bytes(chunk.encode("utf-8")))
        return ids

    def decode(self, ids: list[int]) -> str:
        """Invert `encode` for any id sequence this tokenizer produced."""
        try:
            blob = b"".join(self.id_to_bytes[token_id] for token_id in ids)
        except KeyError as exc:
            raise ValueError(f"unknown token id {exc.args[0]}") from exc
        return blob.decode("utf-8")

    def count(self, text: str) -> int:
        return len(self.encode(text))

    def token_bytes(self, token_id: int) -> bytes:
        try:
            return self.id_to_bytes[token_id]
        except KeyError as exc:
            raise ValueError(f"unknown token id {token_id}") from exc

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _VERSION,
            "vocab_size": self.vocab_size,
            "special_tokens": list(self.special_tokens),
            "merges": [[left, right] for left, right in self.merges],
        }
        destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Tokenizer:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("tokenizer file must be a JSON object")
        if raw.get("version") != _VERSION:
            raise ValueError(f"unsupported tokenizer version {raw.get('version')!r}")
        specials = raw.get("special_tokens")
        merges = raw.get("merges")
        if not isinstance(specials, list) or not all(isinstance(item, str) for item in specials):
            raise ValueError("special_tokens must be a list of strings")
        if not isinstance(merges, list):
            raise ValueError("merges must be a list of pairs")
        parsed: list[tuple[int, int]] = []
        for pair in merges:
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or not all(isinstance(part, int) and not isinstance(part, bool) for part in pair)
            ):
                raise ValueError(f"invalid merge pair {pair!r}")
            parsed.append((pair[0], pair[1]))
        tokenizer = cls(parsed, specials)
        recorded = raw.get("vocab_size")
        if recorded != tokenizer.vocab_size:
            raise ValueError(
                f"vocab_size {recorded} does not match {tokenizer.vocab_size} "
                "ids reconstructed from special tokens and merges"
            )
        return tokenizer

    def _partition(self, text: str) -> list[tuple[str, str]]:
        pattern = self._special_pattern
        if pattern is None:
            return [("text", text)]
        parts: list[tuple[str, str]] = []
        cursor = 0
        for match in pattern.finditer(text):
            if match.start() > cursor:
                parts.append(("text", text[cursor : match.start()]))
            parts.append(("special", match.group()))
            cursor = match.end()
        if cursor < len(text):
            parts.append(("text", text[cursor:]))
        return parts

    def _encode_bytes(self, data: bytes) -> list[int]:
        ids = list(data)
        if len(ids) < 2:
            return ids
        for left, right, new_id in self._merge_table:
            if len(ids) < 2:
                break
            ids = _apply_merge(ids, left, right, new_id)
        return ids


def _compile_specials(specials: list[str]) -> re.Pattern[str] | None:
    if not specials:
        return None
    # Longer tokens first so one special cannot swallow the prefix of another.
    ordered = sorted(specials, key=len, reverse=True)
    return re.compile("|".join(re.escape(token) for token in ordered))


def _split_specials(text: str, specials: list[str]) -> list[str]:
    """Drop special tokens and return the ordinary stretches between them."""
    pattern = _compile_specials(specials)
    if pattern is None:
        return [text] if text else []
    return [part for part in pattern.split(text) if part]
