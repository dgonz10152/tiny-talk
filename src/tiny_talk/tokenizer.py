from collections import Counter
from itertools import pairwise

import regex as re

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class BPETokenizer:
    """Byte-level BPE tokenizer using the GPT-4 regex split pattern."""

    def __init__(self, pattern=None):
        self.merges = {}
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)

    @property
    def vocab_size(self):
        return 256 + len(self.merges)

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
        if vocab_size < 256:
            raise ValueError(f"vocab_size must be at least 256, got {vocab_size}")
        num_merges = vocab_size - 256

        merges = {}
        # Natural text repeats the same chunks constantly, so work on each
        # unique chunk once and weight its pair counts by how often it occurs.
        chunk_counts = Counter(re.findall(self.compiled_pattern, text))
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

    def encode(self, text):
        """Encode text into token ids.

        Args:
            text: Text to encode

        Returns:
            List of token ids
        """
        ids = []
        cache = {}
        for chunk in re.findall(self.compiled_pattern, text):
            if chunk not in cache:
                cache[chunk] = self._encode_chunk(chunk.encode("utf-8"))
            ids.extend(cache[chunk])
        return ids

    def decode(self, ids):
        """Decode token ids back into text.

        Args:
            ids: List of token ids

        Returns:
            Decoded text
        """
        vocab = {i: bytes([i]) for i in range(256)}
        for (i, j), idx in self.merges.items():
            vocab[idx] = vocab[i] + vocab[j]

        return b"".join(vocab[idx] for idx in ids).decode("utf-8", errors="replace")


if __name__ == "__main__":
    with open("data/taylorswift.txt", encoding="utf-8") as f:
        content = f.read()

    tokenizer = BPETokenizer()
    tokenizer.train(content, 500)
    ids = tokenizer.encode("Hello World")
    print(ids)
    print(tokenizer.decode(ids))
