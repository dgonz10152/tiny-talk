import json
from collections import Counter
from itertools import pairwise

import regex as re

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|system_start|>",
    "<|system_end|>",
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
]
SPECIAL_PATTERN = re.compile("(" + "|".join(map(re.escape, SPECIAL_TOKENS)) + ")")


class BPETokenizer:
    """Byte-level BPE tokenizer using the GPT-4 regex split pattern."""

    def __init__(self, pattern=None):
        self.merges = {}
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)

    @property
    def special_tokens(self):
        """Special token ids, placed after the bytes and merges."""
        start = 256 + len(self.merges)
        return {tok: start + i for i, tok in enumerate(SPECIAL_TOKENS)}

    @property
    def vocab_size(self):
        return 256 + len(self.merges) + len(SPECIAL_TOKENS)

    def _get_stats(self, ids, counts=None, weight=1):
        """Count adjacent id pairs.

        Args:
            ids: List of token ids
            counts: Optional dict to accumulate counts into
            weight: Amount to add per occurrence

        Returns:
            Dict mapping each pair to its count
        """
        counts = {} if counts is None else counts
        for pair in pairwise(ids):
            counts[pair] = counts.get(pair, 0) + weight
        return counts

    def _merge(self, ids, pair, idx):
        """Replace every occurrence of pair in ids with idx.

        Args:
            ids: List of token ids
            pair: Pair of ids to merge
            idx: New id for the pair

        Returns:
            Updated list with pair merged
        """
        new_ids = []
        i = 0

        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1

        return new_ids

    def train(self, text, vocab_size):
        min_size = 256 + len(SPECIAL_TOKENS)
        if vocab_size < min_size:
            raise ValueError(
                f"vocab_size must be at least {min_size}, got {vocab_size}"
            )
        num_merges = vocab_size - min_size

        merges = {}
        # Natural text repeats the same chunks constantly, so work on each
        # unique chunk once and weight its pair counts by how often it occurs.
        # Special token strings are dropped so no merges form across them.
        chunk_counts = Counter()
        for part in SPECIAL_PATTERN.split(text):
            if part not in SPECIAL_TOKENS:
                chunk_counts.update(re.findall(self.compiled_pattern, part))
        chunk_ids = [list(chunk.encode("utf-8")) for chunk in chunk_counts]
        weights = list(chunk_counts.values())

        for i in range(num_merges):
            stats = {}
            for ids, weight in zip(chunk_ids, weights):
                self._get_stats(ids, stats, weight)
            if not stats:
                break

            pair = max(stats, key=stats.get)
            idx = 256 + i

            chunk_ids = [self._merge(ids, pair, idx) for ids in chunk_ids]
            merges[pair] = idx

        self.merges = merges

    def _encode_chunk(self, chunk_bytes):
        tokens = list(chunk_bytes)

        while len(tokens) > 1:
            stats = self._get_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))

            if pair not in self.merges:
                break

            tokens = self._merge(tokens, pair, self.merges[pair])

        return tokens

    def encode(self, text, allow_special=False):
        """Encode text into token ids.

        Args:
            text: Text to encode
            allow_special: Map special token strings to their ids. Leave off
                for untrusted text so it cannot inject control tokens.

        Returns:
            List of token ids
        """
        special = self.special_tokens if allow_special else {}
        parts = SPECIAL_PATTERN.split(text) if allow_special else [text]
        ids = []
        cache = {}
        for part in parts:
            if part in special:
                ids.append(special[part])
                continue
            for chunk in re.findall(self.compiled_pattern, part):
                if chunk not in cache:
                    cache[chunk] = self._encode_chunk(chunk.encode("utf-8"))
                ids.extend(cache[chunk])
        return ids

    def decode_bytes(self, ids):
        """Decode token ids into raw bytes, which may end mid UTF-8 character.

        Args:
            ids: List of token ids

        Returns:
            Decoded bytes
        """
        vocab = {i: bytes([i]) for i in range(256)}
        for (i, j), idx in self.merges.items():
            vocab[idx] = vocab[i] + vocab[j]
        for tok, idx in self.special_tokens.items():
            vocab[idx] = tok.encode("utf-8")

        return b"".join(vocab[idx] for idx in ids)

    def decode(self, ids):
        """Decode token ids back into text.

        Args:
            ids: List of token ids

        Returns:
            Decoded text
        """
        return self.decode_bytes(ids).decode("utf-8", errors="replace")

    def save(self, path):
        with open(path, "w") as f:
            json.dump([[a, b, idx] for (a, b), idx in self.merges.items()], f)

    @classmethod
    def load(cls, path):
        tokenizer = cls()
        with open(path) as f:
            tokenizer.merges = {(a, b): idx for a, b, idx in json.load(f)}
        return tokenizer


if __name__ == "__main__":
    with open("data/taylorswift.txt", encoding="utf-8") as f:
        content = f.read()

    tokenizer = BPETokenizer()
    tokenizer.train(content, 500)
    ids = tokenizer.encode("Hello World")
    print(ids)
    print(tokenizer.decode(ids))
