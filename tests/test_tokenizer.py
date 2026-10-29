import random
import re

import pytest

from gpt.tokenizer import DEFAULT_PATTERN, ByteBPETokenizer, _merge, train_bpe

CORPUS = (
    "First Citizen:\nBefore we proceed any further, hear me speak.\n\n"
    "All:\nSpeak, speak.\n\nFirst Citizen:\nYou are all resolved rather to die than to famish?\n\n"
    "All:\nResolved. resolved.\n\nFirst Citizen:\nFirst, you know Caius Marcius is chief enemy "
    "to the people.\n\nAll:\nWe know't, we know't.\n"
) * 20 + "naïve café — 東京 🙂🙂 emoji 👩‍👩‍👧 done.\n"

ROUND_TRIP = [
    "",
    "hello world",
    "  leading and trailing spaces   ",
    "tabs\tand\nnewlines\r\n\n\n  mixed \t \n",
    "naïve café résumé — “quotes” «guillemets»",
    "日本語のテキストと한국어 그리고 中文",
    "emoji 🙂🚀🔥 and ZWJ family 👩‍👩‍👧 and flags 🇮🇳🇺🇸",
    "combining: é ä Z̵̡",
    "under_scores__ and 123,456.78 and x²+y³",
    "\x00 control \x07 chars ​ zero-width",
    "Ünïcödé ÄÖÜ ß ẞ ﬁ ligature",
]


@pytest.fixture(scope="module")
def tok():
    return ByteBPETokenizer.train(CORPUS, vocab_size=400)


def test_merge_helper():
    assert _merge([1, 2, 1, 2, 3], (1, 2), 9) == [9, 9, 3]
    assert _merge([1, 1, 1], (1, 1), 9) == [9, 1]  # non-overlapping, left to right


def test_train_returns_requested_number_of_merges():
    merges = train_bpe(CORPUS, 300)
    assert len(merges) == 44
    assert all(isinstance(a, int) and isinstance(b, int) for a, b in merges)


def test_training_is_deterministic():
    assert train_bpe(CORPUS, 350) == train_bpe(CORPUS, 350)


def test_vocab_bytes_are_concatenations(tok):
    for i, (a, b) in enumerate(tok.merges):
        assert tok.vocab[256 + i] == tok.vocab[a] + tok.vocab[b]


@pytest.mark.parametrize("text", ROUND_TRIP)
def test_round_trip(tok, text):
    assert tok.decode(tok.encode(text)) == text


def test_round_trip_random_unicode(tok):
    rng = random.Random(0)
    planes = [(0x20, 0x7F), (0xA0, 0x2FF), (0x400, 0x4FF), (0x3040, 0x30FF), (0x4E00, 0x4FFF),
              (0x1F300, 0x1F6FF), (0x0, 0x20)]
    for _ in range(200):
        s = "".join(chr(rng.randint(*rng.choice(planes))) for _ in range(rng.randint(0, 40)))
        assert tok.decode(tok.encode(s)) == s


def test_pretokenizer_is_lossless():
    pat = re.compile(DEFAULT_PATTERN)
    for s in ROUND_TRIP + [CORPUS]:
        assert "".join(pat.findall(s)) == s


def test_bpe_compresses_training_text(tok):
    ids = tok.encode(CORPUS)
    assert len(ids) < 0.5 * len(CORPUS.encode("utf-8"))
    assert max(ids) < tok.vocab_size


def test_learned_tokens_are_frequent_words(tok):
    pieces = {tok.vocab[i] for i in range(256, tok.vocab_size)}
    assert b" Citizen" in pieces or b"Citizen" in pieces


def test_save_load_identical(tok, tmp_path):
    path = tmp_path / "tok.json"
    tok.save(path)
    tok2 = ByteBPETokenizer.load(path)
    assert tok2.merges == tok.merges
    assert tok2.vocab_size == tok.vocab_size
    for s in ROUND_TRIP:
        assert tok2.encode(s) == tok.encode(s)


def test_hand_worked_example():
    """chunks: "ab" x1, " ab" x2, " abc" x1.
    pair counts: (a,b)=4, (' ',a)=3, (b,c)=1  -> merge 256 = a+b
    then:        (' ',256)=3, (256,c)=1       -> merge 257 = ' '+ab
    then only (257,c)=1 remains (< 2)         -> stop early with 2 merges."""
    a, b, c, sp = 97, 98, 99, 32
    merges = train_bpe("ab ab ab abc", vocab_size=300)
    assert merges == [(a, b), (sp, 256)]
    t = ByteBPETokenizer(merges)
    assert t.encode(" ab") == [257]
    assert t.encode("abc") == [256, c]
    assert t.encode("ba") == [b, a]


def test_vocab_size_below_256_rejected():
    with pytest.raises(ValueError):
        train_bpe("abc", 100)


def naive_train_bpe(corpus, vocab_size):
    """Textbook BPE: recount every pair from scratch after each merge (slow, obviously correct)."""
    words = [list(c.encode("utf-8")) for c in re.findall(DEFAULT_PATTERN, corpus)]
    merges = []
    for new_id in range(256, vocab_size):
        counts = {}
        for w in words:
            for p in zip(w, w[1:]):
                counts[p] = counts.get(p, 0) + 1
        if not counts:
            break
        best = min(counts, key=lambda p: (-counts[p], p))
        if counts[best] < 2:
            break
        merges.append(best)
        words = [_merge(w, best, new_id) for w in words]
    return merges


def test_incremental_trainer_matches_naive():
    text = CORPUS + " ".join(ROUND_TRIP) * 3
    assert train_bpe(text, 420) == naive_train_bpe(text, 420)
