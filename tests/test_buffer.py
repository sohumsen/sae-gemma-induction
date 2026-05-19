"""
tests/test_buffer.py
Tests for TopKBuffer (pure Python / heapq — no torch, no model).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sae_gemma.capture_activations import TopKBuffer


# ── Basic keep-top-k behaviour ────────────────────────────────────────────────


def test_topk_keeps_highest_activations():
    """After inserting 100 items into a top-5 buffer, the 5 highest survive."""
    buf = TopKBuffer(n_features=1, top_k=5)
    for i in range(100):
        buf.update(0, float(i), {"token": f"tok{i}", "token_pos": i, "seq_id": 0, "context": ""})

    # Heap items: (activation, metadata)
    heap = buf.heaps[0]
    surviving_activations = sorted([item[0] for item in heap], reverse=True)
    # Should be [99, 98, 97, 96, 95]
    assert surviving_activations == [99.0, 98.0, 97.0, 96.0, 95.0], surviving_activations


def test_topk_does_not_exceed_k():
    """Buffer never holds more than top_k items per feature."""
    k = 7
    buf = TopKBuffer(n_features=3, top_k=k)
    for fid in range(3):
        for i in range(50):
            buf.update(fid, float(i), {"token": "x", "token_pos": i, "seq_id": 0, "context": ""})
        assert len(buf.heaps[fid]) == k


def test_topk_updates_when_higher():
    """A new item that beats the current minimum replaces it."""
    buf = TopKBuffer(n_features=1, top_k=3)
    buf.update(0, 1.0, {"token": "a", "token_pos": 0, "seq_id": 0, "context": ""})
    buf.update(0, 2.0, {"token": "b", "token_pos": 1, "seq_id": 0, "context": ""})
    buf.update(0, 3.0, {"token": "c", "token_pos": 2, "seq_id": 0, "context": ""})

    # All three slots full; minimum is 1.0
    assert len(buf.heaps[0]) == 3

    # Insert something lower — should be rejected
    buf.update(0, 0.5, {"token": "low", "token_pos": 3, "seq_id": 0, "context": ""})
    acts = {item[0] for item in buf.heaps[0]}
    assert 0.5 not in acts

    # Insert something higher — should replace the minimum (1.0)
    buf.update(0, 10.0, {"token": "high", "token_pos": 4, "seq_id": 0, "context": ""})
    acts = {item[0] for item in buf.heaps[0]}
    assert 10.0 in acts
    assert 1.0 not in acts


def test_topk_does_not_update_when_lower():
    """Items below the current minimum are silently dropped."""
    buf = TopKBuffer(n_features=1, top_k=2)
    buf.update(0, 5.0, {"token": "x", "token_pos": 0, "seq_id": 0, "context": ""})
    buf.update(0, 6.0, {"token": "y", "token_pos": 1, "seq_id": 0, "context": ""})
    # Heap full; current min = 5.0
    buf.update(0, 3.0, {"token": "z", "token_pos": 2, "seq_id": 0, "context": ""})
    assert len(buf.heaps[0]) == 2
    acts = {item[0] for item in buf.heaps[0]}
    assert 3.0 not in acts


# ── to_dataframe ──────────────────────────────────────────────────────────────


def test_to_dataframe_shape():
    """to_dataframe returns one row per (feature, rank) pair."""
    n_features = 4
    top_k = 3
    buf = TopKBuffer(n_features=n_features, top_k=top_k)
    for fid in range(n_features):
        for i in range(top_k):
            buf.update(fid, float(i + 1), {
                "token": f"t{i}", "token_pos": i, "seq_id": fid * 10 + i, "context": "ctx"
            })

    df = buf.to_dataframe()
    assert len(df) == n_features * top_k
    assert set(df.columns) >= {"feature_id", "rank", "activation", "token", "context", "token_pos", "seq_id"}


def test_to_dataframe_dtypes():
    """to_dataframe enforces the expected dtypes."""
    buf = TopKBuffer(n_features=2, top_k=2)
    for fid in range(2):
        for i in range(2):
            buf.update(fid, float(i + 1), {"token": "a", "token_pos": i, "seq_id": i, "context": ""})

    df = buf.to_dataframe()
    assert str(df["feature_id"].dtype) == "int32"
    assert str(df["rank"].dtype) == "int8"
    assert str(df["activation"].dtype) == "float32"
    assert str(df["token_pos"].dtype) == "int32"
    assert str(df["seq_id"].dtype) == "int64"


def test_to_dataframe_ranks_descending():
    """Rank 0 should have the highest activation within each feature."""
    buf = TopKBuffer(n_features=1, top_k=5)
    for v in [10.0, 3.0, 7.0, 1.0, 5.0]:
        buf.update(0, v, {"token": str(v), "token_pos": 0, "seq_id": 0, "context": ""})

    df = buf.to_dataframe()
    feat_df = df[df["feature_id"] == 0].sort_values("rank")
    acts = feat_df["activation"].tolist()
    # Must be strictly descending
    for i in range(len(acts) - 1):
        assert acts[i] >= acts[i + 1], f"Not descending: {acts}"


def test_to_dataframe_empty_buffer():
    """Buffer with no updates produces an empty DataFrame with correct columns."""
    buf = TopKBuffer(n_features=3, top_k=5)
    df = buf.to_dataframe()
    assert len(df) == 0
    assert "feature_id" in df.columns


# ── Edge cases ────────────────────────────────────────────────────────────────


def test_edge_case_n_features_1_top_k_1():
    """n_features=1, top_k=1: only the single highest activation survives."""
    buf = TopKBuffer(n_features=1, top_k=1)
    for v in [0.1, 0.9, 0.5, 0.8]:
        buf.update(0, v, {"token": str(v), "token_pos": 0, "seq_id": 0, "context": ""})

    assert len(buf.heaps[0]) == 1
    assert abs(buf.heaps[0][0][0] - 0.9) < 1e-6

    df = buf.to_dataframe()
    assert len(df) == 1
    assert abs(float(df["activation"].iloc[0]) - 0.9) < 1e-6


def test_edge_case_fewer_inserts_than_k():
    """Buffer only partially filled — should hold exactly what was inserted."""
    buf = TopKBuffer(n_features=1, top_k=10)
    for i in range(3):
        buf.update(0, float(i), {"token": "t", "token_pos": i, "seq_id": i, "context": ""})

    assert len(buf.heaps[0]) == 3
    df = buf.to_dataframe()
    assert len(df) == 3


def test_multiple_features_independent():
    """Updates to feature 0 must not affect feature 1."""
    buf = TopKBuffer(n_features=2, top_k=3)
    for v in [100.0, 200.0, 300.0]:
        buf.update(0, v, {"token": "f0", "token_pos": 0, "seq_id": 0, "context": ""})
    # Feature 1 receives no updates
    assert len(buf.heaps[0]) == 3
    assert len(buf.heaps[1]) == 0


def test_to_dataframe_metadata_preserved():
    """to_dataframe correctly passes through token, context, token_pos, seq_id."""
    buf = TopKBuffer(n_features=1, top_k=1)
    buf.update(0, 42.0, {
        "token": "hello",
        "context": "some context",
        "token_pos": 7,
        "seq_id": 99,
    })
    df = buf.to_dataframe()
    row = df.iloc[0]
    assert row["token"] == "hello"
    assert row["context"] == "some context"
    assert row["token_pos"] == 7
    assert row["seq_id"] == 99
