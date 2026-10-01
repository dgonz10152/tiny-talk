from tiny_talk.tokenizer import BPETokenizer

TEXT = "hello hello world, the world says hello! 123 ünïcödé"


def test_bpe_roundtrip_and_compresses():
    tok = BPETokenizer()
    tok.train(TEXT * 5, 280)
    ids = tok.encode(TEXT)
    assert tok.decode(ids) == TEXT
    assert len(ids) < len(TEXT.encode("utf-8"))


def test_bpe_counts_pairs_across_chunks():
    # "ab" appears once per chunk; summed it must win over the in-chunk "cd" pair.
    tok = BPETokenizer()
    tok.train("ab ab ab cdcd", 257)
    assert list(tok.merges) == [(ord("a"), ord("b"))]


def test_bpe_weights_repeated_chunks():
    # " xy" is one unique chunk seen 3 times; its pairs must outscore "cd" (2).
    tok = BPETokenizer()
    tok.train("cdcd xy xy xy", 257)
    assert (ord("c"), ord("d")) not in tok.merges
