# Ledger

Ledger measures text before it is spent against a context window.

## Count

Count is a byte-level byte-pair encoder. It is the standard library only.

- Token ids `0`–`255` are the UTF-8 bytes with those values.
- The next ids are special tokens. `<|end|>` is one id, and merges never cross it.
- Every id after that is a merge, learned most-frequent pair first.
- When two pairs tie, the lexicographically smaller pair of token byte strings is merged.
- `decode(encode(text))` returns the original text, including non-ASCII characters and `<|end|>`.

`vocab_size` counts byte ids, special tokens, and merges together. With the default special token, a vocabulary of 512 is 256 bytes + `<|end|>` + 255 merges.

The trained tokenizer for the prose corpus lives at `artifacts/tokenizer.json`.

### Train

```bash
ledger train --corpus data/prose.txt --vocab-size 512 --out artifacts/tokenizer.json
```

### Count a string

```bash
ledger count "token" --tokenizer artifacts/tokenizer.json
```

Prints the number of tokens.

### Compare two files

```bash
ledger compare examples/prose.txt examples/code.py --tokenizer artifacts/tokenizer.json
```

Prints a tab-separated table: `file`, `tokens`, `chars`, `per_100_chars`.

### Budget a prompt

```bash
ledger budget examples/prompt.txt --window 8192 --tokenizer artifacts/tokenizer.json
```

The prompt is divided by header lines:

```
--- system
--- history
--- retrieved
--- question
```

Header lines are ignored. Each section is counted on its own, so a merge cannot cross from one section into the next. The command prints every section, the total, and the window. It exits 1 when the total is greater than the window.

## A trained vocabulary

`artifacts/tokenizer.json` was trained on `data/prose.txt` at a vocabulary size of 512.

```text
$ ledger count "token" --tokenizer artifacts/tokenizer.json
3

$ ledger compare examples/prose.txt examples/code.py --tokenizer artifacts/tokenizer.json
file	tokens	chars	per_100_chars
examples/prose.txt	400	988	40.49
examples/code.py	807	988	81.68
```

Those two files are the same length. The code costs more tokens because the merges were learned from ordinary English. The UUID `550e8400-e29b-41d4-a716-446655440000` costs 36 tokens, against 3 for the word `token`.
