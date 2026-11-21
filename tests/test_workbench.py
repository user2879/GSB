"""Acceptance tests for the verification workbench."""

import json
import shutil
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from workbench import loader
from workbench.config import Config
from workbench.detect import detect_sample
from workbench.edits import apply_edit
from workbench.errors import (
    E_ARBITRATION_DEADLOCK,
    E_BOX_OUT_OF_RANGE,
    E_LABEL_ENUM,
    E_ORDER_GAP,
    E_SAMPLE_INVALID,
    E_SNAPSHOT_STALE,
)
from workbench.inject import run_injections
from workbench.pipeline import run_pipeline
from workbench.samplegen import build_dataset
from workbench.snapshot import SnapshotStore, StaleCommit

CONFIG = ROOT / "config.yaml"
DATA = ROOT / "data"


@pytest.fixture(scope="session", autouse=True)
def dataset():
    if not (DATA / "images").is_dir():
        build_dataset(Config.load(CONFIG), DATA)
    return DATA


def run(config, data, out, **kwargs):
    return run_pipeline(str(config), str(data), str(out), **kwargs)


def comparable(report):
    return {k: v for k, v in report.items() if k not in ("incremental", "concurrency")}


# 1. full run: ledger, arbitration, deadlock, zero entries, merge evidence
def test_full_run_ledger_and_arbitration(tmp_path, dataset):
    report = run(CONFIG, dataset, tmp_path / "out")
    assert report["samples"] == [f"s0{i}" for i in range(1, 9)]
    assert report["failures"] == []

    by_key = {}
    for entry in report["ledger"]:
        by_key.setdefault((entry["sample_id"], entry["category"], entry["color"]), {})[
            entry["source"]
        ] = entry
    # every combo carries all four sources
    for combo, sources in by_key.items():
        assert set(sources) == {"detection", "description", "label_a", "label_b"}
        for entry in sources.values():
            assert isinstance(entry["count"], int)
            assert entry["source_fingerprint"]

    # zero-count entries are kept and explicitly marked
    zero = [e for e in report["ledger"] if e["zero"]]
    assert zero and all(e["count"] == 0 for e in zero)
    assert any(e["sample_id"] == "s05" and e["source"] == "description" for e in zero)

    # s02 occlusion: red square merged from two fragments with evidence
    s02 = report["detection"]["s02"]["instances"]
    red = [i for i in s02 if i["color"] == "red"]
    assert len(red) == 1 and red[0]["merged"] and len(red[0]["fragments"]) == 2
    for frag in red[0]["fragments"]:
        assert len(frag["bbox"]) == 4 and "color_distance" in frag

    # s03 conflict resolved by detection priority, label_b rejected
    arb = {(a["sample_id"], a["category"], a["color"]): a for a in report["arbitration"]}
    s03 = arb[("s03", "triangle", "green")]
    assert s03["status"] == "conflict" and s03["arbitrated_count"] == 2
    assert s03["rule"] == "detection"
    assert s03["rejected"] == [{"source": "label_b", "count": 3, "deviation": 1}]

    # s04 deadlock
    s04 = arb[("s04", "circle", "blue")]
    assert s04["status"] == "deadlock" and s04["code"] == E_ARBITRATION_DEADLOCK
    assert s04["arbitrated_count"] is None
    assert report["deadlocks"] == [
        {"sample_id": "s04", "category": "circle", "color": "blue",
         "code": E_ARBITRATION_DEADLOCK}
    ]

    # s06 duplicate description assertions merged with details
    s06_desc = by_key[("s06", "square", "red")]["description"]
    assert s06_desc["merged"] and s06_desc["count"] == 2
    assert [d["count"] for d in s06_desc["merge_details"]] == [1, 1]

    # detection artifact carries image + params fingerprints
    det = report["detection"]["s01"]
    assert det["image_fingerprint"] and det["params_fingerprint"]


# 2. loader validation business codes
def test_loader_validation_codes(tmp_path):
    config = Config.load(CONFIG)
    data = tmp_path / "data"
    (data / "images").mkdir(parents=True)
    (data / "descriptions").mkdir()
    # valid image for b01, truncated image for b02
    build_dataset(config, tmp_path / "src")
    shutil.copy(tmp_path / "src" / "images" / "s01.png", data / "images" / "b01.png")
    raw = (tmp_path / "src" / "images" / "s02.png").read_bytes()
    (data / "images" / "b02.png").write_bytes(raw[:64])
    # bad enum in description
    (data / "descriptions" / "b01.json").write_text(json.dumps({
        "sample_id": "b01",
        "assertions": [{"category": "hexagon", "color": "red", "count": 1}],
    }))
    # bbox out of range in label_a, order gap in label_b
    (data / "labels_a.jsonl").write_text(json.dumps({
        "sample_id": "b01", "category": "square", "color": "red",
        "bbox": [0, 0, 999, 10], "order": 1,
    }) + "\n")
    (data / "labels_b.jsonl").write_text(
        json.dumps({"sample_id": "b01", "category": "square", "color": "red",
                    "bbox": [0, 0, 10, 10], "order": 1}) + "\n"
        + json.dumps({"sample_id": "b01", "category": "square", "color": "red",
                      "bbox": [10, 10, 20, 20], "order": 4}) + "\n"
    )
    samples, failures = loader.load_inputs(data, config)
    codes = {(f["sample_id"], f["source"]): f["code"] for f in failures}
    assert codes[("b02", "image")] == E_SAMPLE_INVALID
    assert codes[("b01", "description")] == E_LABEL_ENUM
    assert codes[("b01", "label_a")] == E_BOX_OUT_OF_RANGE
    assert codes[("b01", "label_b")] == E_ORDER_GAP
    # failure list ordering is stable
    keys = [(f["sample_id"], f["source"]) for f in sorted(failures, key=lambda f: (f["sample_id"], f["source"]))]
    assert keys == sorted(keys)
    # offending sources dropped, others kept
    assert samples["b01"]["label_a"] == []
    assert len(samples["b01"]["label_b"]) == 1


# 3. detection determinism
def test_detection_determinism(dataset):
    config = Config.load(CONFIG)
    samples, _ = loader.load_inputs(dataset, config)
    arr = samples["s02"]["image"]
    first = detect_sample("s02", arr, config)
    second = detect_sample("s02", arr, config)
    assert first == second
    ids = [i["instance_id"] for i in first["instances"]]
    assert ids == sorted(ids)


# 4. incremental equivalence across the full edit sequence
def test_incremental_equivalence_per_edit(tmp_path, dataset):
    work = tmp_path / "work"
    shutil.copytree(dataset, work)
    config_work = tmp_path / "config.yaml"
    shutil.copy(CONFIG, config_work)
    edits = json.loads((dataset / "edits.json").read_text())

    cache = tmp_path / "cache.json"
    base = run(CONFIG, dataset, tmp_path / "base", cache_path=cache)
    assert base["failures"] == []

    for step, edit in enumerate(edits, start=1):
        apply_edit(edit, work, config_work)
        inc = run(config_work, work, tmp_path / f"inc{step}",
                  cache_path=cache, incremental=True)
        full = run(config_work, work, tmp_path / f"full{step}")
        assert comparable(inc) == comparable(full), f"step {step}: {edit['op']}"
        # reuse + recompute universe aligns with the full run
        inc_cats = inc["incremental"]["categories"]
        universe = {(i["sample_id"], i["category"])
                    for i in inc_cats["reused"] + inc_cats["recomputed"]}
        full_set = {(i["sample_id"], i["category"])
                    for i in full["incremental"]["categories"]["recomputed"]}
        assert universe == full_set
        det = inc["incremental"]["detection"]
        assert set(det["reused"]) | set(det["recomputed"]) == set(inc["samples"])

    # precise invalidation spot checks (re-run selected steps fresh)
    work2 = tmp_path / "work2"
    shutil.copytree(dataset, work2)
    config2 = tmp_path / "config2.yaml"
    shutil.copy(CONFIG, config2)
    cache2 = tmp_path / "cache2.json"
    run(config2, work2, tmp_path / "seed", cache_path=cache2)

    apply_edit(edits[0], work2, config2)  # desc_modify s01 square count
    inc = run(config2, work2, tmp_path / "p1", cache_path=cache2, incremental=True)
    assert inc["incremental"]["categories"]["recomputed"] == [
        {"sample_id": "s01", "category": "square"}
    ]
    assert inc["incremental"]["detection"]["recomputed"] == []

    apply_edit(edits[5], work2, config2)  # replace_image s07
    inc = run(config2, work2, tmp_path / "p2", cache_path=cache2, incremental=True)
    assert inc["incremental"]["detection"]["recomputed"] == ["s07"]
    assert {(i["sample_id"], i["category"]) for i in inc["incremental"]["categories"]["recomputed"]} == {
        ("s07", "square"), ("s07", "circle"), ("s07", "triangle")
    }

    apply_edit(edits[6], work2, config2)  # set_tolerance
    inc = run(config2, work2, tmp_path / "p3", cache_path=cache2, incremental=True)
    assert inc["incremental"]["detection"]["recomputed"] == []
    assert len(inc["incremental"]["categories"]["recomputed"]) == 8 * 3


# 5. concurrency equals serial
def test_concurrency_consistency(tmp_path, dataset):
    serial = run(CONFIG, dataset, tmp_path / "s1", concurrency=1, verify_serial=True)
    parallel = run(CONFIG, dataset, tmp_path / "s4", concurrency=4, verify_serial=True)
    assert serial["concurrency"]["serial_consistent"] is True
    assert parallel["concurrency"]["serial_consistent"] is True
    assert comparable(serial) == comparable(parallel)


# 6. snapshot stale detection, resume, commit-time version check
def test_snapshot_stale_and_resume(tmp_path, dataset):
    snap = tmp_path / "snap"
    first = run(CONFIG, dataset, tmp_path / "o1", snapshot_dir=snap)
    assert first["snapshot"]["stale"] is False
    manifest = json.loads((snap / "manifest.json").read_text())
    assert manifest["completed_stages"] == ["load", "detect", "ledger", "arbitrate"]
    assert manifest["cursor"] == "s08"
    assert manifest["config_hash"] == first["config_hash"]
    assert "images/s01.png" in manifest["source_hashes"]

    # resume with matching version/config: no stale, identical results
    second = run(CONFIG, dataset, tmp_path / "o2", snapshot_dir=snap)
    assert second["snapshot"]["stale"] is False
    assert comparable(second) == comparable(first)

    # tamper version -> stale -> recompute, identical results
    manifest["version"] = 999
    (snap / "manifest.json").write_text(json.dumps(manifest))
    third = run(CONFIG, dataset, tmp_path / "o3", snapshot_dir=snap)
    assert third["snapshot"]["stale"] is True
    assert third["snapshot"]["events"][0]["code"] == E_SNAPSHOT_STALE
    strip = lambda r: {k: v for k, v in comparable(r).items() if k != "snapshot"}
    assert strip(third) == strip(first)


def test_snapshot_commit_detects_version_change(tmp_path):
    store = SnapshotStore(tmp_path / "snap")
    observed = store.read_manifest()
    store.commit("load", {"x": 1}, observed, version=1, config_hash="h",
                 source_hashes={}, cursor="s01")
    stale_observed = observed  # manifest moved on since this was read
    with pytest.raises(StaleCommit):
        store.commit("detect", {"y": 2}, stale_observed, version=1, config_hash="h",
                     source_hashes={}, cursor="s01")


# 7. fault injections map to business codes
def test_injections(tmp_path, dataset):
    report = run_injections(CONFIG, dataset, dataset / "injections.json", tmp_path / "inj")
    hits = {h["id"]: h for h in report["injections"]}
    assert set(hits) == {"INJ-001", "INJ-002", "INJ-003", "INJ-004"}
    assert all(h["hit"] for h in hits.values())
    assert hits["INJ-001"]["expected_code"] == E_SAMPLE_INVALID
    assert hits["INJ-002"]["expected_code"] == E_LABEL_ENUM
    assert hits["INJ-003"]["expected_code"] == E_ORDER_GAP
    assert hits["INJ-004"]["expected_code"] == E_BOX_OUT_OF_RANGE
    keys = [(f["sample_id"], f["source"], f["code"]) for f in report["failures"]]
    assert keys == sorted(keys)


# 8. arbitration priority change is honored (label majority)
def test_arbitration_priority_change(tmp_path, dataset):
    config_raw = yaml.safe_load(CONFIG.read_text())
    config_raw["arbitration_priority"] = ["label_majority", "detection", "description", "label_a"]
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config_raw, sort_keys=False))
    report = run(config_path, dataset, tmp_path / "out")
    arb = {(a["sample_id"], a["category"], a["color"]): a for a in report["arbitration"]}
    s08 = arb[("s08", "circle", "red")]
    assert s08["status"] == "conflict"
    assert s08["rule"] == "label_majority"
    assert s08["arbitrated_count"] == 2
    assert s08["rejected"] == [{"source": "description", "count": 3, "deviation": 1}]
