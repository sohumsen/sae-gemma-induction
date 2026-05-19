"""
tests/test_probe_utils.py
Pure-Python tests for build_probe and _safe_vocab_range.
No model loading required.
"""
import random
import sys
from pathlib import Path

# Ensure the installed src package (or editable install) is importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sae_gemma.induction_probes import SAFE_VOCAB_HI, SAFE_VOCAB_LO, _safe_vocab_range, build_probe


# ── _safe_vocab_range ─────────────────────────────────────────────────────────


def test_safe_vocab_range_large_vocab():
    """When model vocab > SAFE_VOCAB_HI the hi bound is clipped to SAFE_VOCAB_HI."""
    lo, hi = _safe_vocab_range(100_000)
    assert lo == SAFE_VOCAB_LO
    assert hi == SAFE_VOCAB_HI


def test_safe_vocab_range_small_vocab():
    """When model vocab < SAFE_VOCAB_HI the hi bound is clipped to vocab_size - 1."""
    small_vocab = 30_000
    lo, hi = _safe_vocab_range(small_vocab)
    assert lo == SAFE_VOCAB_LO
    assert hi == small_vocab - 1


def test_safe_vocab_range_exactly_at_boundary():
    """Edge: model_vocab_size == SAFE_VOCAB_HI."""
    lo, hi = _safe_vocab_range(SAFE_VOCAB_HI)
    assert hi == SAFE_VOCAB_HI - 1


def test_safe_vocab_range_output_types():
    lo, hi = _safe_vocab_range(256_000)
    assert isinstance(lo, int)
    assert isinstance(hi, int)


# ── build_probe ───────────────────────────────────────────────────────────────


def _make_probe(seed: int, prefix_len: int = 10, gap_len: int = 10, vocab_size: int = 100_000):
    rng = random.Random(seed)
    lo, hi = _safe_vocab_range(vocab_size)
    tokens, A, B = build_probe(rng, lo, hi, prefix_len, gap_len)
    return tokens, A, B, prefix_len, gap_len


def test_build_probe_total_length():
    """Total token count == prefix_len + 2 + gap_len + 1."""
    for seed in range(5):
        tokens, A, B, prefix_len, gap_len = _make_probe(seed, prefix_len=15, gap_len=20)
        expected_len = prefix_len + 2 + gap_len + 1
        assert len(tokens) == expected_len, (
            f"Expected {expected_len}, got {len(tokens)}"
        )


def test_build_probe_ends_with_A():
    """The last token must be A."""
    for seed in range(10):
        tokens, A, B, _, _ = _make_probe(seed)
        assert tokens[-1] == A, f"Last token {tokens[-1]} != A={A}"


def test_build_probe_A_at_prefix_len():
    """A appears at index prefix_len (first occurrence)."""
    for seed in range(10):
        prefix_len = 10
        tokens, A, B, _, _ = _make_probe(seed, prefix_len=prefix_len)
        assert tokens[prefix_len] == A, (
            f"tokens[{prefix_len}]={tokens[prefix_len]}, expected A={A}"
        )


def test_build_probe_B_after_first_A():
    """B appears immediately after the first A (at prefix_len + 1)."""
    for seed in range(10):
        prefix_len = 10
        tokens, A, B, _, _ = _make_probe(seed, prefix_len=prefix_len)
        assert tokens[prefix_len + 1] == B, (
            f"tokens[{prefix_len + 1}]={tokens[prefix_len + 1]}, expected B={B}"
        )


def test_build_probe_A_not_in_gap():
    """A must not appear in the gap region (positions prefix_len+2 through prefix_len+2+gap_len-1)."""
    for seed in range(20):
        prefix_len = 12
        gap_len = 30
        tokens, A, B, _, _ = _make_probe(seed, prefix_len=prefix_len, gap_len=gap_len)
        gap_start = prefix_len + 2
        gap_end = gap_start + gap_len
        gap_tokens = tokens[gap_start:gap_end]
        assert A not in gap_tokens, (
            f"A={A} found in gap: {gap_tokens}"
        )


def test_build_probe_A_and_B_differ():
    """A and B must be distinct tokens."""
    for seed in range(20):
        _, A, B, _, _ = _make_probe(seed)
        assert A != B, f"A == B == {A}"


def test_build_probe_tokens_in_vocab_range():
    """All tokens must be within [vocab_lo, vocab_hi]."""
    vocab_size = 100_000
    lo, hi = _safe_vocab_range(vocab_size)
    for seed in range(10):
        tokens, A, B, _, _ = _make_probe(seed, vocab_size=vocab_size)
        for tok in tokens:
            assert lo <= tok <= hi, f"Token {tok} out of range [{lo}, {hi}]"


def test_build_probe_various_prefix_gap_combos():
    """Structural invariants hold for a range of prefix/gap lengths."""
    params = [
        (10, 10),
        (20, 50),
        (80, 500),
        (30, 100),
        (10, 200),
    ]
    for seed, (prefix_len, gap_len) in enumerate(params):
        tokens, A, B, pl, gl = _make_probe(seed, prefix_len=prefix_len, gap_len=gap_len)
        # Length
        assert len(tokens) == pl + 2 + gl + 1
        # Structural positions
        assert tokens[-1] == A
        assert tokens[pl] == A
        assert tokens[pl + 1] == B
        # No A in gap
        gap = tokens[pl + 2: pl + 2 + gl]
        assert A not in gap


def test_build_probe_deterministic():
    """Same seed produces the same output."""
    tokens1, A1, B1, _, _ = _make_probe(seed=42)
    tokens2, A2, B2, _, _ = _make_probe(seed=42)
    assert tokens1 == tokens2
    assert A1 == A2
    assert B1 == B2


def test_build_probe_different_seeds_differ():
    """Different seeds should produce different results (very high probability)."""
    results = set()
    for seed in range(5):
        tokens, A, B, _, _ = _make_probe(seed)
        results.add(tuple(tokens))
    assert len(results) > 1, "All seeds produced identical sequences — suspiciously unlikely"
