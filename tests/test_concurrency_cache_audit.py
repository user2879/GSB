from __future__ import annotations

import json

from corpus_workbench.demo_data import generate_demo_workspace
from corpus_workbench.pipeline import Workbench


def _workbench(
    tmp_path,
    output_name: str,
    run_id: str,
    scheduler_name: str,
    dedup_name: str = "dedup.json",
):
    return Workbench(
        data_path=tmp_path / "data" / "corpus",
        balance_path=tmp_path / "configs" / "balance.json",
        dedup_path=tmp_path / "configs" / dedup_name,
        scheduler_path=tmp_path / "configs" / scheduler_name,
        injection_path=tmp_path / "configs" / "injection.json",
        output_root=tmp_path / output_name,
        run_id=run_id,
    )


def test_serial_and_parallel_are_field_identical(tmp_path):
    generate_demo_workspace(tmp_path)
    serial = _workbench(tmp_path, "out-serial", "serial", "scheduler_serial.json").run(reset=True)
    parallel = _workbench(tmp_path, "out-parallel", "parallel", "scheduler_parallel.json").run(reset=True)

    assert serial["deterministic_result_digest"] == parallel["deterministic_result_digest"]
    assert serial["deterministic_core"] == parallel["deterministic_core"]
    assert serial["budget"]["deduction_sequence"] == parallel["budget"]["deduction_sequence"]
    assert [item["stable_key"] for item in serial["budget"]["deduction_sequence"]] == sorted(
        item["stable_key"] for item in serial["budget"]["deduction_sequence"]
    )
    assert serial["dedup"]["groups"] == parallel["dedup"]["groups"]
    assert serial["failures"] == parallel["failures"]


def test_threshold_reuses_signatures_and_parameter_change_invalidates(tmp_path):
    generate_demo_workspace(tmp_path)
    baseline = _workbench(tmp_path, "out-cache", "base", "scheduler.json").run(reset=True)
    unique_count = baseline["documents"]["sampled_unique_before_dedup"]
    assert baseline["cache_audit"]["signatures"]["misses"] == unique_count

    threshold = _workbench(
        tmp_path,
        "out-cache",
        "threshold",
        "scheduler.json",
        dedup_name="dedup_threshold_090.json",
    ).run(reset=True)
    signature_audit = threshold["cache_audit"]["signatures"]
    assert signature_audit["hits"] == unique_count
    assert signature_audit["misses"] == 0
    assert threshold["budget"]["consumed_points"] == 0
    assert signature_audit["reused_sources"] == ["books", "papers", "web"]

    changed = _workbench(
        tmp_path,
        "out-cache",
        "shingle",
        "scheduler.json",
        dedup_name="dedup_shingle_4_dim_32.json",
    ).run(reset=True)
    changed_audit = changed["cache_audit"]["signatures"]
    assert changed_audit["hits"] == 0
    assert changed_audit["misses"] == unique_count
    assert changed_audit["invalidated_by_params"] == unique_count


def test_content_change_only_invalidates_affected_document(tmp_path):
    generate_demo_workspace(tmp_path)
    base = _workbench(tmp_path, "out-content", "base", "scheduler.json").run(reset=True)
    source, doc_id = base["sampling"]["samples"][0]["source"], base["sampling"]["samples"][0]["doc_id"]
    path = tmp_path / "data" / "corpus" / f"{source}.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    changed = False
    for index, line in enumerate(lines):
        record = json.loads(line)
        if record["doc_id"] == doc_id:
            record["text"] += " altered-token"
            lines[index] = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            changed = True
    assert changed
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    rerun = _workbench(tmp_path, "out-content", "content-change", "scheduler.json").run(reset=True)
    audit = rerun["cache_audit"]["signatures"]
    unique_count = base["documents"]["sampled_unique_before_dedup"]
    assert audit["hits"] == unique_count - 1
    assert audit["misses"] == 1
    assert audit["invalidated_by_content"] == 1
    assert audit["miss_sources"] == {source: 1}


def test_content_change_can_resume_and_reclaims_only_changed_signature(tmp_path):
    generate_demo_workspace(tmp_path)
    workbench = _workbench(tmp_path, "out-resume-content", "resume-content", "scheduler.json")
    baseline = workbench.run(reset=True)
    source, doc_id = baseline["sampling"]["samples"][0]["source"], baseline["sampling"]["samples"][0]["doc_id"]
    old_deductions = [
        item for item in baseline["budget"]["deduction_sequence"]
        if item["source"] == source and item["doc_id"] == doc_id
    ]
    assert len(old_deductions) == 1
    old_points = old_deductions[0]["points"]

    path = tmp_path / "data" / "corpus" / f"{source}.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        record = json.loads(line)
        if record["doc_id"] == doc_id:
            record["text"] += " resume-altered-token"
            lines[index] = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            break
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    resumed = workbench.run(resume=True)
    assert resumed["status"] == "complete"
    assert resumed["budget"]["reclaimed_points"] == old_points
    new_deductions = [
        item for item in resumed["budget"]["deduction_sequence"]
        if item["source"] == source and item["doc_id"] == doc_id
    ]
    assert len(new_deductions) == 1
    assert new_deductions[0]["sequence"] == old_deductions[0]["sequence"]
    assert [item["stable_key"] for item in resumed["budget"]["deduction_sequence"]] == sorted(
        item["stable_key"] for item in resumed["budget"]["deduction_sequence"]
    )
    assert resumed["cache_audit"]["signatures"]["misses"] == 1
    assert resumed["cache_audit"]["signatures"]["miss_sources"] == {source: 1}
