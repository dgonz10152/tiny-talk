from tiny_talk.tokenizer import SPECIAL_TOKENS, BPETokenizer

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
    tok.train("ab ab ab cdcd", 257 + len(SPECIAL_TOKENS))
    assert list(tok.merges) == [(ord("a"), ord("b"))]


def test_bpe_weights_repeated_chunks():
    # " xy" is one unique chunk seen 3 times; its pairs must outscore "cd" (2).
    tok = BPETokenizer()
    tok.train("cdcd xy xy xy", 257 + len(SPECIAL_TOKENS))
    assert (ord("c"), ord("d")) not in tok.merges


CHAT = "<|user_start|>hi there<|user_end|><|assistant_start|>hello<|assistant_end|>"


def test_special_tokens_encode_to_single_ids():
    tok = BPETokenizer()
    tok.train(TEXT * 5, 280)
    special = tok.special_tokens
    ids = tok.encode(CHAT, allow_special=True)
    assert ids[0] == special["<|user_start|>"]
    assert ids[-1] == special["<|assistant_end|>"]
    assert tok.decode(ids) == CHAT


def test_special_tokens_ignored_without_allow_special():
    tok = BPETokenizer()
    tok.train(TEXT * 5, 280)
    ids = tok.encode(CHAT)
    assert not set(ids) & set(tok.special_tokens.values())
    assert tok.decode(ids) == CHAT


def test_vocab_size_includes_special_tokens():
    tok = BPETokenizer()
    tok.train(TEXT * 5, 280)
    assert tok.vocab_size == 280
    assert sorted(tok.special_tokens.values()) == list(
        range(280 - len(SPECIAL_TOKENS), 280)
    )


def test_training_skips_special_token_strings():
    tok = BPETokenizer()
    tok.train("<|endoftext|>".join(["ab"] * 50), 258 + len(SPECIAL_TOKENS))
    assert list(tok.merges) == [(ord("a"), ord("b"))]
