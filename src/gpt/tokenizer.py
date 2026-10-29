"""Byte-level BPE tokenizer, written from scratch (no tiktoken, no HF tokenizers).

Pipeline (same shape as GPT-2's tokenizer, Radford et al. 2019):

  text --regex pre-tokenize--> chunks --utf-8--> bytes --apply merges by rank--> ids

* The base vocabulary is the 256 byte values, so *any* string is encodable and
  decode(encode(s)) == s exactly; there is no <unk> token.
* Pre-tokenization splits text into word-like chunks and BPE never merges across a
  chunk boundary, so e.g. "dog." and "dog!" share the token for "dog".
* Merge i creates token id 256 + i.  Encoding repeatedly applies the lowest-rank
  (earliest learned) merge present in a chunk, which reproduces training exactly.

Training uses an incremental pair-count index plus a lazy max-heap, so each merge only
touches the words that contain the merged pair instead of rescanning the corpus.
"""

from __future__ import annotations

import heapq
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

# GPT-2-style split, expressed with the stdlib `re` module (no `regex` dependency):
#   contractions | optional-space + letters | optional-space + digits |
#   optional-space + punctuation/symbols | trailing whitespace | whitespace | anything else.
# `[^\W\d_]` is "unicode letter"; the final `[\s\S]` guarantees every character is
# covered, so pre-tokenization is lossless (asserted in tests).
DEFAULT_PATTERN = (
    r"'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+|_+|[\s\S]"
)

Pair = tuple[int, int]


def _merge(ids: list[int], pair: Pair, new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of `pair` (left to right) with `new_id`."""
    out: list[int] = []
    i, n = 0, len(ids)
    a, b = pair
    while i < n:
        if i < n - 1 and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def train_bpe(corpus: str, vocab_size: int, pattern: str = DEFAULT_PATTERN,
              verbose: bool = False) -> list[Pair]:
    """Learn `vocab_size - 256` merges from `corpus`. Returns merges in rank order.

    Deterministic: ties on pair frequency are broken by the smaller pair (a, b).
    Stops early if no pair occurs at least twice (nothing useful left to merge).
    """
    if vocab_size < 256:
        raise ValueError("vocab_size must be >= 256 (the byte alphabet)")
    chunk_counts = Counter(re.findall(pattern, corpus))
    words: list[list[int]] = [list(c.encode("utf-8")) for c in chunk_counts]
    freqs: list[int] = list(chunk_counts.values())

    pair_counts: dict[Pair, int] = defaultdict(int)
    where: dict[Pair, set[int]] = defaultdict(set)  # pair -> indices of words containing it
    for wi, w in enumerate(words):
        for p in zip(w, w[1:]):
            pair_counts[p] += freqs[wi]
            where[p].add(wi)

    # max-heap via negated counts; entries can go stale and are re-validated on pop
    heap = [(-c, p) for p, c in pair_counts.items()]
    heapq.heapify(heap)

    merges: list[Pair] = []
    for new_id in range(256, vocab_size):
        best = None
        while heap:
            neg_c, p = heapq.heappop(heap)
            if pair_counts.get(p, 0) == -neg_c:
                best = p
                break
        if best is None or pair_counts[best] < 2:
            break
        merges.append(best)

        touched: set[Pair] = set()
        for wi in list(where.pop(best)):
            w = words[wi]
            if len(w) < 2:
                continue
            f = freqs[wi]
            for p in zip(w, w[1:]):          # retract this word's old pairs ...
                pair_counts[p] -= f
                touched.add(p)
            w = _merge(w, best, new_id)
            words[wi] = w
            for p in zip(w, w[1:]):          # ... and add its new ones
                pair_counts[p] += f
                where[p].add(wi)
                touched.add(p)
        for p in touched:
            c = pair_counts.get(p, 0)
            if c <= 0:
                pair_counts.pop(p, None)
            else:
                heapq.heappush(heap, (-c, p))
        if verbose and (new_id - 255) % 100 == 0:
            print(f"  merge {new_id - 255}/{vocab_size - 256}: {best} -> {new_id}")
    return merges


class ByteBPETokenizer:
    def __init__(self, merges: list[Pair] | None = None, pattern: str = DEFAULT_PATTERN):
        self.merges: list[Pair] = [tuple(m) for m in (merges or [])]  # type: ignore[misc]
        self.pattern = pattern
        self._re = re.compile(pattern)
        self.ranks: dict[Pair, int] = {p: i for i, p in enumerate(self.merges)}
        self.vocab: list[bytes] = [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            self.vocab.append(self.vocab[a] + self.vocab[b])
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges)

    @classmethod
    def train(cls, corpus: str, vocab_size: int, pattern: str = DEFAULT_PATTERN,
              verbose: bool = False) -> "ByteBPETokenizer":
        return cls(train_bpe(corpus, vocab_size, pattern, verbose), pattern)

    def _encode_chunk(self, chunk: str) -> list[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            # the earliest-learned merge available in this chunk
            pair = min(zip(ids, ids[1:]), key=lambda p: self.ranks.get(p, float("inf")))
            rank = self.ranks.get(pair)
            if rank is None:
                break
            ids = _merge(ids, pair, 256 + rank)
        if len(self._cache) < 200_000:
            self._cache[chunk] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        for chunk in self._re.findall(text):
            out.extend(self._encode_chunk(chunk))
        return out

    def decode(self, ids: list[int]) -> str:
        # errors="replace": an arbitrary id sequence (e.g. sampled from a model) may split
        # a multi-byte character; ids produced by encode() always round-trip exactly.
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        data = {"type": "byte-bpe", "version": 1, "pattern": self.pattern,
                "vocab_size": self.vocab_size, "merges": [list(m) for m in self.merges]}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ByteBPETokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("type") != "byte-bpe":
            raise ValueError(f"{path} is not a byte-bpe tokenizer file")
        return cls([tuple(m) for m in data["merges"]], data["pattern"])
