"""Tests for the from-scratch byte-level BPE tokenizer."""

from helix.tokenizer import ByteLevelBPETokenizer


def test_roundtrip_is_lossless_for_ascii():
    tok = ByteLevelBPETokenizer()
    tok.train(["hello world", "The quick brown fox.", ""], vocab_size=300)
    original = "hello world, helix!"
    ids = tok.encode(original)
    assert tok.decode(ids) == original
    assert len(tok) <= 300


def test_encode_produces_token_ids():
    tok = ByteLevelBPETokenizer()
    tok.train(["abc def"], vocab_size=300)
    ids = tok.encode("abc def")
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)


def test_special_tokens_roundtrip():
    tok = ByteLevelBPETokenizer()
    tok.train(["<|endoftext|> abc <|pad|>"], vocab_size=300)
    for special in ("<|endoftext|>", "<|pad|>"):
        assert tok.decode(tok.encode(special)) == special


def test_vocab_size_cap():
    tok = ByteLevelBPETokenizer()
    tok.train(["a b c d e f g h i j"] * 50, vocab_size=260)
    assert len(tok) <= 260


def test_save_load_roundtrip(tmp_path):
    tok = ByteLevelBPETokenizer()
    tok.train(["round trip test data"], vocab_size=300)
    vocab_file = tmp_path / "vocab.json"
    merges_file = tmp_path / "merges.json"
    tok.save(str(vocab_file), str(merges_file))
    tok2 = ByteLevelBPETokenizer.load(str(vocab_file), str(merges_file))
    assert len(tok2) == len(tok)
    s = "round trip test data"
    assert tok.encode(s) == tok2.encode(s)
    assert tok.decode(tok.encode(s)) == s
