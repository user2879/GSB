import json

import pytest

from gpu_diag.errors import (E_CYCLE, E_DECL, E_MEMORY, E_QUOTA, DiagError)
from gpu_diag.loader import load_declaration


def _write(directory, pools, tasks, params):
    (directory / "pools.json").write_text(json.dumps(pools), encoding="utf-8")
    (directory / "tasks.json").write_text(json.dumps(tasks), encoding="utf-8")
    (directory / "params.json").write_text(json.dumps(params), encoding="utf-8")


def test_valid_declaration_loads(sample_dir):
    decl = load_declaration(sample_dir / "pools.json",
                            sample_dir / "tasks.json",
                            sample_dir / "params.json")
    assert len(decl.pools) == 3
    assert len(decl.tasks) == 5
    assert "T1#0" in decl.shards


def test_non_positive_pool_field_rejected(sample_dir):
    pools = json.loads((sample_dir / "pools.json").read_text())
    pools["pools"][0]["gpu_count"] = 0
    tasks = json.loads((sample_dir / "tasks.json").read_text())
    params = json.loads((sample_dir / "params.json").read_text())
    _write(sample_dir, pools, tasks, params)
    with pytest.raises(DiagError) as exc:
        load_declaration(sample_dir / "pools.json", sample_dir / "tasks.json",
                         sample_dir / "params.json")
    assert exc.value.code == E_DECL
    assert any(v["field"] == "gpu_count" for v in exc.value.details)


def test_duplicate_identifiers_rejected(sample_dir, tmp_path):
    pools = json.loads((sample_dir / "pools.json").read_text())
    pools["pools"].append(dict(pools["pools"][0]))
    tasks = json.loads((sample_dir / "tasks.json").read_text())
    params = json.loads((sample_dir / "params.json").read_text())
    _write(sample_dir, pools, tasks, params)
    with pytest.raises(DiagError) as exc:
        load_declaration(sample_dir / "pools.json", sample_dir / "tasks.json",
                         sample_dir / "params.json")
    assert exc.value.code == E_DECL
    assert any("unique" in v["expected"] for v in exc.value.details)


def test_memory_overflow_uses_e_memory(sample_dir):
    tasks = json.loads((sample_dir / "tasks.json").read_text())
    tasks["tasks"][0]["memory_per_gpu_mib"] = 99999
    (sample_dir / "tasks.json").write_text(json.dumps(tasks), encoding="utf-8")
    with pytest.raises(DiagError) as exc:
        load_declaration(sample_dir / "pools.json", sample_dir / "tasks.json",
                         sample_dir / "params.json")
    assert exc.value.code == E_MEMORY


def test_quota_share_overflow_uses_e_quota(sample_dir):
    pools = json.loads((sample_dir / "pools.json").read_text())
    pools["pools"][0]["quota_share"] = 90
    (sample_dir / "pools.json").write_text(json.dumps(pools), encoding="utf-8")
    with pytest.raises(DiagError) as exc:
        load_declaration(sample_dir / "pools.json", sample_dir / "tasks.json",
                         sample_dir / "params.json")
    assert exc.value.code == E_QUOTA


def test_dependency_cycle_uses_e_cycle(sample_dir):
    tasks = json.loads((sample_dir / "tasks.json").read_text())
    for task in tasks["tasks"]:
        if task["task_id"] == "T1":
            task["deps"] = ["T4"]
        if task["task_id"] == "T4":
            task["deps"] = ["T1"]
    (sample_dir / "tasks.json").write_text(json.dumps(tasks), encoding="utf-8")
    with pytest.raises(DiagError) as exc:
        load_declaration(sample_dir / "pools.json", sample_dir / "tasks.json",
                         sample_dir / "params.json")
    assert exc.value.code == E_CYCLE


def test_missing_dependency_ref_rejected(sample_dir):
    tasks = json.loads((sample_dir / "tasks.json").read_text())
    tasks["shards"][0]["deps"] = ["NOPE#9"]
    (sample_dir / "tasks.json").write_text(json.dumps(tasks), encoding="utf-8")
    with pytest.raises(DiagError) as exc:
        load_declaration(sample_dir / "pools.json", sample_dir / "tasks.json",
                         sample_dir / "params.json")
    assert exc.value.code == E_DECL


def test_atomic_rejection_lists_all_violations(sample_dir):
    pools = json.loads((sample_dir / "pools.json").read_text())
    pools["pools"][0]["gpu_count"] = -1
    pools["pools"][1]["compute_units"] = 0
    (sample_dir / "pools.json").write_text(json.dumps(pools), encoding="utf-8")
    with pytest.raises(DiagError) as exc:
        load_declaration(sample_dir / "pools.json", sample_dir / "tasks.json",
                         sample_dir / "params.json")
    fields = {v["field"] for v in exc.value.details}
    assert {"gpu_count", "compute_units"}.issubset(fields)
