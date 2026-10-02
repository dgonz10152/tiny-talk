import numpy as np

from tiny_talk.prepare import tokenize_docs
from tiny_talk.tokenizer import BPETokenizer


def test_tokenize_docs_prefixes_each_doc_with_eot():
    tok = BPETokenizer()
    tok.train("hello world again " * 5, 280)
    eot = tok.special_tokens["<|endoftext|>"]
    ids = tokenize_docs(["hello world", "<|endoftext|> again"], tok)
    assert ids.dtype == np.uint16
    # The literal "<|endoftext|>" in doc text must not become a real EOT.
    assert ids[0] == eot and (ids == eot).sum() == 2
    assert tok.decode(ids.tolist()) == (
        "<|endoftext|>hello world<|endoftext|><|endoftext|> again"
    )


def test_tokenizer_save_load_roundtrip(tmp_path):
    tok = BPETokenizer()
    tok.train("hello world again " * 5, 280)
    tok.save(tmp_path / "merges.json")
    assert BPETokenizer.load(tmp_path / "merges.json").merges == tok.merges
