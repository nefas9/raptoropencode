"""Tests for ``core.witness.store.WitnessStore``.

Pin the contract: put / get / has / list semantics, dedup by hash,
hash-mismatch rejection, tolerant load on malformed manifests.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


# core/witness/tests/test_store.py → parents[3] = repo root
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from core.witness.store import WitnessStore, WitnessStoreError  # noqa: E402
from core.witness.types import (  # noqa: E402
    Witness,
    WitnessOutcome,
    WitnessSource,
    compute_bytes_hash,
)


def _make_witness(data: bytes, source: WitnessSource = WitnessSource.FUZZ):
    return Witness(
        bytes_hash=compute_bytes_hash(data),
        source=source,
        observed_outcome=WitnessOutcome.EXIT_SIGNAL,
    )


# ----------------------------------------------------------------------
# Put / get round-trip
# ----------------------------------------------------------------------


def test_put_then_get_witness_and_bytes(tmp_path):
    store = WitnessStore(tmp_path)
    data = b"trigger payload"
    w = _make_witness(data)

    store.put(w, data)
    loaded_w = store.get_witness(w.bytes_hash)
    loaded_bytes = store.get_bytes(w.bytes_hash)

    assert loaded_w.bytes_hash == w.bytes_hash
    assert loaded_w.source == w.source
    assert loaded_bytes == data


def test_put_stamps_bytes_len_if_default(tmp_path):
    """Producers that forget to set bytes_len get it filled in
    from the actual data length."""
    store = WitnessStore(tmp_path)
    data = b"A" * 200
    w = _make_witness(data)
    assert w.bytes_len == 0  # default
    store.put(w, data)
    loaded = store.get_witness(w.bytes_hash)
    assert loaded.bytes_len == 200


def test_put_creates_directories_lazily(tmp_path):
    """Constructing a store doesn't create dirs; first put does."""
    root = tmp_path / "new" / "deep" / "path"
    store = WitnessStore(root)
    assert not (root / "manifests").exists()
    assert not (root / "blobs").exists()
    store.put(_make_witness(b"data"), b"data")
    assert (root / "manifests").is_dir()
    assert (root / "blobs").is_dir()


# ----------------------------------------------------------------------
# Hash mismatch rejection
# ----------------------------------------------------------------------


def test_put_rejects_hash_mismatch(tmp_path):
    """Storing a witness whose hash doesn't match the data raises.
    Catches the common producer bug of computing hash on a
    transformed copy of the bytes."""
    store = WitnessStore(tmp_path)
    data = b"actual data"
    w = Witness(
        bytes_hash=compute_bytes_hash(b"different data"),
        source=WitnessSource.FUZZ,
        observed_outcome=WitnessOutcome.EXIT_SIGNAL,
    )
    with pytest.raises(WitnessStoreError, match="does not match"):
        store.put(w, data)


# ----------------------------------------------------------------------
# Existence check + missing-blob handling
# ----------------------------------------------------------------------


def test_has_after_put(tmp_path):
    store = WitnessStore(tmp_path)
    data = b"x"
    w = _make_witness(data)
    assert not store.has(w.bytes_hash)
    store.put(w, data)
    assert store.has(w.bytes_hash)


def test_get_missing_witness_raises(tmp_path):
    store = WitnessStore(tmp_path)
    with pytest.raises(WitnessStoreError, match="manifest not found"):
        store.get_witness("a" * 64)


def test_get_missing_bytes_raises(tmp_path):
    store = WitnessStore(tmp_path)
    with pytest.raises(WitnessStoreError, match="blob not found"):
        store.get_bytes("a" * 64)


def test_blob_path_returns_none_for_missing(tmp_path):
    """``blob_path`` is the soft-lookup variant — returns None on
    miss rather than raising. Useful for the "if we have it, use
    it; otherwise skip" pattern."""
    store = WitnessStore(tmp_path)
    assert store.blob_path("a" * 64) is None
    data = b"present"
    w = _make_witness(data)
    store.put(w, data)
    p = store.blob_path(w.bytes_hash)
    assert p is not None
    assert p.is_file()
    assert p.read_bytes() == data


# ----------------------------------------------------------------------
# Dedup on identical bytes
# ----------------------------------------------------------------------


def test_dedup_blob_across_different_witnesses(tmp_path):
    """Same bytes seen by two pipelines → single blob, two manifests.
    Wait — same hash means single manifest (overwrites). Verify the
    blob isn't rewritten and the most-recent manifest wins."""
    store = WitnessStore(tmp_path)
    data = b"same bytes"

    w1 = Witness(
        bytes_hash=compute_bytes_hash(data),
        source=WitnessSource.FUZZ,
        observed_outcome=WitnessOutcome.EXIT_SIGNAL,
        produced_by="afl++",
    )
    w2 = Witness(
        bytes_hash=compute_bytes_hash(data),
        source=WitnessSource.CRASH_REPLAY,
        observed_outcome=WitnessOutcome.SANITIZER_REPORT,
        produced_by="rr/replay",
    )

    store.put(w1, data)
    blob_path = tmp_path / "blobs" / f"{w1.bytes_hash}.bin"
    mtime_after_w1 = blob_path.stat().st_mtime

    store.put(w2, data)
    # Blob not rewritten (same content); manifest now reflects w2.
    assert blob_path.stat().st_mtime == mtime_after_w1
    loaded = store.get_witness(w1.bytes_hash)
    assert loaded.source == WitnessSource.CRASH_REPLAY  # w2 won
    assert loaded.produced_by == "rr/replay"


# ----------------------------------------------------------------------
# list_witnesses
# ----------------------------------------------------------------------


def test_list_witnesses_on_empty_store(tmp_path):
    store = WitnessStore(tmp_path)
    assert list(store.list_witnesses()) == []


def test_list_witnesses_returns_all(tmp_path):
    store = WitnessStore(tmp_path)
    pairs = [(f"data-{i}".encode(), i) for i in range(5)]
    for data, _i in pairs:
        store.put(_make_witness(data), data)
    listed = list(store.list_witnesses())
    assert len(listed) == 5
    hashes = {w.bytes_hash for w in listed}
    expected = {compute_bytes_hash(d) for d, _ in pairs}
    assert hashes == expected


def test_list_witnesses_skips_malformed_manifest(tmp_path):
    """Malformed JSON shouldn't abort enumeration — log a warning
    and skip. The store's contract is "load all valid records,"
    not "fail fast on the first bad one."""
    store = WitnessStore(tmp_path)
    # Plant one valid + one malformed manifest.
    data = b"valid"
    store.put(_make_witness(data), data)
    malformed = tmp_path / "manifests" / "deadbeef.json"
    malformed.write_text("{ this is not json")

    listed = list(store.list_witnesses())
    assert len(listed) == 1
    assert listed[0].bytes_hash == compute_bytes_hash(data)


# ----------------------------------------------------------------------
# Idempotency / overwrite semantics
# ----------------------------------------------------------------------


def test_put_overwrites_manifest_idempotent_on_blob(tmp_path):
    """Re-putting the same (hash, data, witness) is a no-op for the
    blob and a manifest-rewrite. Useful for retry-safe pipelines."""
    store = WitnessStore(tmp_path)
    data = b"retry-safe"
    w = _make_witness(data)
    store.put(w, data)
    store.put(w, data)  # second put
    loaded = store.get_witness(w.bytes_hash)
    assert loaded.bytes_hash == w.bytes_hash


def test_manifest_is_valid_json(tmp_path):
    """The persisted manifest is human-readable JSON — operators
    can inspect it without a special loader. Worth pinning so a
    future refactor doesn't silently switch to pickle or msgpack."""
    store = WitnessStore(tmp_path)
    data = b"inspectable"
    w = _make_witness(data)
    store.put(w, data)
    manifest_path = tmp_path / "manifests" / f"{w.bytes_hash}.json"
    parsed = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert parsed["bytes_hash"] == w.bytes_hash
    assert parsed["source"] == "fuzz"
