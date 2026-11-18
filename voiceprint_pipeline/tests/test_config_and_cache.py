"""Requirement 6/7: scope fingerprints, cache keys and versioned snapshot semantics."""
from __future__ import annotations

import json
import os

from voiceprint import cache as cache_mod
from voiceprint.config import load_config
from voiceprint.util import atomic_write_json, read_json


def test_scope_hashes_isolate_threshold_change(make_config):
    base = make_config()
    changed = make_config(similarity_threshold=0.9)
    assert base.scope_hash("feature") == changed.scope_hash("feature")
    assert base.scope_hash("stats") == changed.scope_hash("stats")
    assert base.scope_hash("quant") == changed.scope_hash("quant")
    assert base.scope_hash("embedding") == changed.scope_hash("embedding")
    assert base.scope_hash("similarity") != changed.scope_hash("similarity")


def test_scope_hashes_react_to_quant_change(make_config):
    base = make_config()
    changed = make_config(quant_levels=base.quant_levels * 2)
    assert base.scope_hash("quant") != changed.scope_hash("quant")
    assert base.scope_hash("feature") == changed.scope_hash("feature")


def test_cache_key_chains_are_stable_and_width_sensitive(make_config):
    cfg = make_config()
    fk = cache_mod.feature_cache_key("s1", "pcmfp", cfg.scope_hash("feature"))
    ak = cache_mod.aggregation_cache_key(fk, cfg.scope_hash("stats"))
    ek12 = cache_mod.embedding_cache_key(ak, cfg.scope_hash("quant"),
                                         cfg.scope_hash("embedding"), 12, cfg.quant_levels)
    ek13 = cache_mod.embedding_cache_key(ak, cfg.scope_hash("quant"),
                                         cfg.scope_hash("embedding"), 13, cfg.quant_levels)
    assert ek12 != ek13
    # recomputing with identical inputs yields the same key (deterministic)
    assert cache_mod.embedding_cache_key(
        ak, cfg.scope_hash("quant"), cfg.scope_hash("embedding"), 12, cfg.quant_levels
    ) == ek12
    sk = cache_mod.similarity_cache_key(ek12, ek12, cfg.scope_hash("similarity"), 12)
    assert len(sk) == 16 and sk == cache_mod.similarity_cache_key(
        ek12, ek12, cfg.scope_hash("similarity"), 12
    )


def test_snapshot_manifest_roundtrip_and_atomic_write(tmp_path):
    store = cache_mod.SnapshotStore(str(tmp_path / "snap"))
    manifest = {"global_version": 1, "artifact_versions": {"features": {"k": 1}}}
    store.commit(manifest)
    loaded = store.read_manifest()
    assert loaded["global_version"] == 1
    assert "manifest_fp" in loaded
    # no temp files left behind after atomic replace
    leftovers = [n for n in os.listdir(str(tmp_path / "snap")) if n.endswith(".tmp")]
    assert leftovers == []


def test_prune_keeps_referenced_versions(tmp_path):
    store = cache_mod.SnapshotStore(str(tmp_path / "snap"))
    for version in range(1, 6):
        store.write_artifact(version, "features", f"key{version}", {"v": version})
    referenced = {4, 5}
    store.prune_old_versions(keep=1, referenced_versions=referenced)
    remaining = sorted(os.listdir(os.path.join(str(tmp_path / "snap"), "versions")))
    # v4 and v5 referenced; newest also retained
    assert "v00000004" in remaining and "v00000005" in remaining
