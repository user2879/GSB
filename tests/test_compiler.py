from edge_lease.compiler import Compiler
from edge_lease.manifest import load_manifests, load_registry, load_shards


def _load(workspace):
    registry = load_registry(workspace / "nodes.json")
    shards = load_shards(workspace / "shards.json")
    return load_manifests(workspace / "manifest.json", registry, shards)


def test_compile_is_deterministic_and_byte_stable(sample_workspace):
    manifests = _load(sample_workspace)
    compiler = Compiler(content_seed=20261123, scale=1000)
    compiled_a, items_a = compiler.compile_version(manifests[0])
    compiled_b, items_b = compiler.compile_version(manifests[0])
    assert [item.to_dict() for item in items_a] == [
        item.to_dict() for item in items_b
    ]
    assert [cp.artifact_digest for cp in compiled_a.values()] == [
        cp.artifact_digest for cp in compiled_b.values()
    ]
    # items are sorted by priority/path/node
    keys = [(item.priority, item.path, item.node_id) for item in items_a]
    assert keys == sorted(keys)


def test_derived_digest_matches_fixed_point_contract(sample_workspace):
    manifests = _load(sample_workspace)
    compiler = Compiler(content_seed=20261123, scale=1000)
    compiled, _items = compiler.compile_version(manifests[0])
    derived = compiled["media/index.json"]
    intro = compiled["media/intro.bin"]
    body = compiled["media/body.bin"]
    coeff = dict((row[0], row[1]) for row in derived.coeff_q)
    expected = (coeff["media/intro.bin"] * intro.digest + coeff["media/body.bin"] * body.digest) // 1000
    assert derived.digest == expected
    assert derived.derived_from == ("media/intro.bin", "media/body.bin")
    assert set(derived.error) == {"single", "propagated", "bound"}


def test_incremental_compile_keeps_unchanged_artifacts(sample_workspace):
    manifests = _load(sample_workspace)
    compiler = Compiler(content_seed=20261123, scale=1000)
    old_compiled, _old_items = compiler.compile_version(manifests[0])
    merged, _new_items, stable_paths, steps = compiler.incremental_compile(
        manifests[0], manifests[1], old_compiled
    )
    # intro and body did not change in v2
    assert "media/intro.bin" in stable_paths
    assert "media/body.bin" in stable_paths
    assert merged["media/intro.bin"].artifact_digest == old_compiled["media/intro.bin"].artifact_digest
    # index changed: targets + coefficient
    invalidated = {step["path"] for step in steps}
    assert "media/index.json" in invalidated
    reasons = {step["path"]: step["reason"] for step in steps if step["parent"] == "<root>"}
    assert reasons["media/index.json"] in {
        "targets_changed",
        "derived_coeff_changed",
    }


def test_invalidation_propagates_through_dependents(sample_workspace):
    manifests = _load(sample_workspace)
    compiler = Compiler(content_seed=20261123, scale=1000)
    # change an upstream leaf's shards; its dependent must be invalidated
    v2 = manifests[1]
    entry = v2.entry_map()["media/intro.bin"]
    object.__setattr__(
        entry,
        "shards",
        (
            {"start": 0, "end": 64},
            {"start": 64, "end": 128},
        ),
    )
    invalidated, steps = compiler.propagate_invalidation(manifests[0], v2)
    assert "media/intro.bin" in invalidated
    assert "media/index.json" in invalidated
    downstream = [s for s in steps if s["reason"] == "downstream_invalidated"]
    assert any(s["path"] == "media/index.json" and s["parent"] == "media/intro.bin" for s in downstream)
    # topological order preserved in output
    assert invalidated.index("media/intro.bin") < invalidated.index("media/index.json")
