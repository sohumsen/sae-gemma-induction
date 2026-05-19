"""
tests/test_autointerp.py
Pure-logic tests for autointerp.py helpers.
No model loading, no subprocess calls.
"""
import json
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sae_gemma.autointerp import _append_to_cache, _format_snippets, _load_cache


# ── _format_snippets ──────────────────────────────────────────────────────────


def test_format_snippets_marks_token():
    """Each activating token should be wrapped in <<...>> in the output."""
    rows = [
        {"context": "The cat sat on the mat", "token": "cat"},
        {"context": "Dogs are great pets", "token": "Dogs"},
    ]
    result = _format_snippets(rows)
    assert "<<cat>>" in result
    assert "<<Dogs>>" in result


def test_format_snippets_numbering():
    """Each snippet is prefixed with a 1-based index."""
    rows = [{"context": "hello world", "token": "hello"}]
    result = _format_snippets(rows)
    assert result.startswith("1.")


def test_format_snippets_multiple_rows_numbered():
    rows = [
        {"context": "foo bar baz", "token": "foo"},
        {"context": "alpha beta", "token": "alpha"},
        {"context": "x y z", "token": "x"},
    ]
    result = _format_snippets(rows)
    assert "1." in result
    assert "2." in result
    assert "3." in result


def test_format_snippets_token_not_in_context():
    """If token is not a substring of context, the context is returned unchanged (no crash)."""
    rows = [{"context": "some text here", "token": "absent_token"}]
    result = _format_snippets(rows)
    # Should not raise; the context should still appear
    assert "some text here" in result


def test_format_snippets_empty_token():
    """Empty token string: context is returned without any <<>> wrapping."""
    rows = [{"context": "plain text", "token": ""}]
    result = _format_snippets(rows)
    assert "<<>>" not in result
    assert "plain text" in result


def test_format_snippets_caps_at_20():
    """Only the first 20 rows are used even if more are provided."""
    rows = [{"context": f"context {i}", "token": f"tok{i}"} for i in range(30)]
    result = _format_snippets(rows)
    lines = [l for l in result.strip().split("\n") if l]
    assert len(lines) == 20


def test_format_snippets_empty_list():
    """Empty input produces an empty string (no crash)."""
    result = _format_snippets([])
    assert result == ""


def test_format_snippets_replaces_first_occurrence_only():
    """Only the first occurrence of the token in context is wrapped."""
    rows = [{"context": "cat cat cat", "token": "cat"}]
    result = _format_snippets(rows)
    # Exactly one <<cat>> and two bare 'cat' remaining (total occurrences = 3)
    assert result.count("<<cat>>") == 1


# ── _load_cache ───────────────────────────────────────────────────────────────


def test_load_cache_missing_file(tmp_path):
    """_load_cache returns {} when the file does not exist."""
    missing = tmp_path / "nonexistent.json"
    result = _load_cache(missing)
    assert result == {}


def test_load_cache_reads_correctly(tmp_path):
    """_load_cache reads a valid JSON file and converts keys to int."""
    cache_file = tmp_path / "labels.json"
    data = {"1": "some label", "42": "another label"}
    cache_file.write_text(json.dumps(data), encoding="utf-8")

    result = _load_cache(cache_file)
    assert result == {1: "some label", 42: "another label"}


def test_load_cache_key_types_are_int(tmp_path):
    """Keys returned by _load_cache must be Python ints, not strings."""
    cache_file = tmp_path / "labels.json"
    cache_file.write_text(json.dumps({"7": "label"}), encoding="utf-8")
    result = _load_cache(cache_file)
    for k in result:
        assert isinstance(k, int), f"Expected int key, got {type(k)}"


def test_load_cache_returns_empty_on_corrupt_json(tmp_path):
    """_load_cache returns {} when the file contains invalid JSON."""
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{this is not valid json", encoding="utf-8")
    result = _load_cache(bad_file)
    assert result == {}


def test_load_cache_empty_json_object(tmp_path):
    """_load_cache returns {} for an empty JSON object."""
    cache_file = tmp_path / "empty.json"
    cache_file.write_text("{}", encoding="utf-8")
    result = _load_cache(cache_file)
    assert result == {}


# ── _append_to_cache ──────────────────────────────────────────────────────────


def test_append_to_cache_creates_file(tmp_path):
    """_append_to_cache creates the file if it does not exist."""
    cache_file = tmp_path / "labels.json"
    assert not cache_file.exists()
    _append_to_cache(cache_file, feature_id=5, label="detects cats")
    assert cache_file.exists()


def test_append_to_cache_single_entry(tmp_path):
    """A single append produces a valid JSON file with one entry."""
    cache_file = tmp_path / "labels.json"
    _append_to_cache(cache_file, feature_id=10, label="detects numbers")
    with cache_file.open() as f:
        data = json.load(f)
    assert "10" in data
    assert data["10"] == "detects numbers"


def test_append_to_cache_accumulates(tmp_path):
    """Multiple sequential appends accumulate without overwriting."""
    cache_file = tmp_path / "labels.json"
    _append_to_cache(cache_file, feature_id=1, label="label one")
    _append_to_cache(cache_file, feature_id=2, label="label two")
    _append_to_cache(cache_file, feature_id=3, label="label three")

    with cache_file.open() as f:
        data = json.load(f)
    assert len(data) == 3
    assert data["1"] == "label one"
    assert data["2"] == "label two"
    assert data["3"] == "label three"


def test_append_to_cache_overwrites_existing_key(tmp_path):
    """Writing the same feature_id twice updates the label."""
    cache_file = tmp_path / "labels.json"
    _append_to_cache(cache_file, feature_id=7, label="old label")
    _append_to_cache(cache_file, feature_id=7, label="new label")

    with cache_file.open() as f:
        data = json.load(f)
    assert data["7"] == "new label"
    assert len(data) == 1


def test_append_to_cache_thread_safe(tmp_path):
    """Concurrent appends from multiple threads should not lose any entry."""
    cache_file = tmp_path / "labels.json"
    n_threads = 20
    errors = []

    def worker(fid):
        try:
            _append_to_cache(cache_file, feature_id=fid, label=f"label-{fid}")
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread errors: {errors}"

    with cache_file.open() as f:
        data = json.load(f)

    # All n_threads entries should be present
    assert len(data) == n_threads, f"Expected {n_threads} entries, got {len(data)}"
    for i in range(n_threads):
        assert str(i) in data, f"Missing feature_id {i}"
