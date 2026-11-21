from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

from news_diag.config import WorkbenchConfig
from news_diag.errors import ConfigInvalid
from news_diag.loader import load_documents
from news_diag.pipeline import Workbench
from news_diag.util import canonical_json


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
DOCS = SAMPLES / "news.jsonl"
CONFIG = SAMPLES / "config.yaml"
COMMANDS = SAMPLES / "commands.jsonl"
INJECTIONS = SAMPLES / "injections.yaml"


def _run(tmp_path: Path, name: str, *, rebuild: bool = False,
         apply_through: int | None = None, injections: Path | None = None,
         config: WorkbenchConfig | None = None, commands: Path | None = COMMANDS,
         **kwargs):
    config = config or WorkbenchConfig.load(CONFIG)
    workbench = Workbench(DOCS, config, commands, tmp_path / name / "snapshots",
                          injections_path=injections)
    return workbench.run(rebuild=rebuild, apply_through=apply_through, **kwargs)


def test_full_stage_contract_and_offsets(tmp_path: Path) -> None:
    run = _run(tmp_path, "full", rebuild=True)
    canonical = run.canonical
    assert len(canonical["documents"]) == 10
    assert run.report["failure_count"] == 1
    assert run.report["failures"][0]["code"] == "E_COMMAND_INVALID"
    assert run.report["snapshot_version"] >= 1
    assert set(run.report["stage_timings_ms"]) == {"split", "scoring", "selection", "rewrite", "summary"}
    assert run.report["summary"]["truncation_count"] > 0

    for document in canonical["documents"]:
        split = document["split"]
        source = split["source_text"]
        for sentence in split["sentences"]:
            assert source[sentence["start"]:sentence["end"]] == sentence["text"]
            assert sentence["end"] > sentence["start"]
        scored = document["scoring"]["sentences"]
        assert [item["score"] for item in scored] == [item["score"] for item in scored]
        selected = document["selection"]["selected_sentences"]
        assert selected == sorted(
            selected,
            key=lambda item: (-item["score"], item["start"], item["end"], item["sentence_id"]),
        )
        assert len(selected) <= 2  # the command stream's rollback leaves max_sentences=2
        for sentence in scored:
            if sentence["entity_coverage_zero"]:
                assert "entity_coverage" not in sentence["score_sources"]
                assert sentence["entity_coverage_scaled"] == 0
        for record in document["rewrite"]["replacement_records"]:
            assert record["candidate_index"] == 0
        for truncation in document["rewrite"]["truncations"]:
            assert truncation["final_length"] == truncation["truncation_offset"]
            assert truncation["original_length"] > truncation["final_length"]


def test_incremental_matches_full_after_every_command(tmp_path: Path) -> None:
    config = WorkbenchConfig.load(CONFIG)
    incremental_dirs = tmp_path / "incremental"
    full_dirs = tmp_path / "full"
    for step in range(12):
        incremental = Workbench(DOCS, config, COMMANDS, incremental_dirs / "snapshots").run(
            apply_through=step,
        )
        full = Workbench(DOCS, config, COMMANDS, full_dirs / "snapshots").run(
            rebuild=True,
            apply_through=step,
        )
        assert canonical_json(incremental.canonical) == canonical_json(full.canonical), step


def test_precise_stage_level_invalidation(tmp_path: Path) -> None:
    config = WorkbenchConfig.load(CONFIG)
    snapshot_dir = tmp_path / "snapshots"

    baseline = Workbench(DOCS, config, COMMANDS, snapshot_dir).run(apply_through=0)
    assert baseline.report["reuse_counts"] == {
        "checked": 41, "reused": 0, "recomputed": 41, "failed": 0,
    }

    unchanged = Workbench(DOCS, config, COMMANDS, snapshot_dir).run(apply_through=0)
    assert unchanged.report["reuse_counts"] == {
        "checked": 41, "reused": 41, "recomputed": 0, "failed": 0,
    }

    add_paragraph = Workbench(DOCS, config, COMMANDS, snapshot_dir).run(apply_through=1)
    assert add_paragraph.report["reuse_counts"] == {
        "checked": 41, "reused": 36, "recomputed": 5, "failed": 0,
    }

    summary_length = Workbench(DOCS, config, COMMANDS, snapshot_dir).run(apply_through=2)
    assert summary_length.report["reuse_counts"] == {
        "checked": 41, "reused": 20, "recomputed": 21, "failed": 0,
    }

    style = Workbench(DOCS, config, COMMANDS, snapshot_dir).run(apply_through=3)
    assert style.report["reuse_counts"] == {
        "checked": 41, "reused": 30, "recomputed": 11, "failed": 0,
    }

    replacement_version = Workbench(DOCS, config, COMMANDS, snapshot_dir).run(apply_through=4)
    assert replacement_version.report["reuse_counts"] == {
        "checked": 41, "reused": 30, "recomputed": 11, "failed": 0,
    }


def test_injection_isolation_and_business_codes(tmp_path: Path) -> None:
    run = _run(tmp_path, "injection", rebuild=True, injections=INJECTIONS)
    assert run.report["injection_hit_count"] == 5
    assert all(item["hit"] and item["code_match"] for item in run.report["injections"])
    failures = run.report["failures"]
    injected_failures = [item for item in failures if item.get("injection_id")]
    assert {item["document_id"] for item in injected_failures} == {
        "news-0002", "news-0003", "news-0005", "news-0007", "news-0009",
    }
    assert {item["code"] for item in injected_failures} == {
        "E_DOC_INVALID", "E_CONFIG_INVALID",
    }
    summarized = {document["document_id"] for document in run.canonical["documents"]}
    assert summarized.isdisjoint({item["document_id"] for item in injected_failures})
    assert run.report["summary"]["document_count"] == 5


def test_stale_snapshot_is_refused(tmp_path: Path) -> None:
    config = WorkbenchConfig.load(CONFIG)
    snapshot_dir = tmp_path / "snapshots"
    Workbench(DOCS, config, COMMANDS, snapshot_dir).run()
    stale = Workbench(DOCS, config, COMMANDS, snapshot_dir).run(expect_version=999)
    assert stale.report["stale"] is True
    assert stale.report["reuse_counts"]["reused"] == 0
    assert any(item["code"] == "E_SNAPSHOT_STALE" for item in stale.report["failures"])
    assert stale.report["stale_error"]["details"]["reasons"] == ["global_version_mismatch"]


def test_corrupt_current_is_stale_and_rebuild_version_is_monotonic(tmp_path: Path) -> None:
    config = WorkbenchConfig.load(CONFIG)
    snapshot_dir = tmp_path / "snapshots"
    baseline = Workbench(DOCS, config, COMMANDS, snapshot_dir).run(apply_through=2)
    rebuilt = Workbench(DOCS, config, COMMANDS, snapshot_dir).run(
        rebuild=True,
        apply_through=2,
    )
    assert rebuilt.report["snapshot_version"] > baseline.report["snapshot_version"]

    current = (snapshot_dir / "CURRENT").read_text(encoding="utf-8").strip()
    (snapshot_dir / current / "state.json").write_text("{broken", encoding="utf-8")
    recovered = Workbench(DOCS, config, COMMANDS, snapshot_dir).run(apply_through=2)
    assert recovered.report["stale"] is True
    assert recovered.report["reuse_counts"]["reused"] == 0
    assert any(item["code"] == "E_SNAPSHOT_STALE" for item in recovered.report["failures"])
    assert recovered.report["summary"]["document_count"] == 10


def test_thread_count_does_not_change_canonical_result(tmp_path: Path) -> None:
    config = WorkbenchConfig.load(CONFIG)
    serial = _run(tmp_path, "serial", rebuild=True, config=replace(config, thread_pool_size=1))
    parallel = _run(tmp_path, "parallel", rebuild=True, config=replace(config, thread_pool_size=8))
    assert canonical_json(serial.canonical) == canonical_json(parallel.canonical)
    assert [d["document_id"] for d in serial.canonical["documents"]] == sorted(
        d["document_id"] for d in parallel.canonical["documents"]
    )


def test_graceful_signal_resume_equals_full(tmp_path: Path) -> None:
    config = WorkbenchConfig.load(CONFIG)
    snapshot_dir = tmp_path / "snapshots"
    partial = Workbench(DOCS, config, COMMANDS, snapshot_dir).run(stop_after_docs=2)
    assert partial.interrupted is True
    assert partial.report["completed_document_ids"]
    resumed = Workbench(DOCS, config, COMMANDS, snapshot_dir).run()
    full = Workbench(DOCS, config, COMMANDS, tmp_path / "full").run(rebuild=True)
    assert resumed.report["mode"] == "incremental"
    assert canonical_json(resumed.canonical) == canonical_json(full.canonical)


def test_orphan_generation_is_ignored(tmp_path: Path) -> None:
    config = WorkbenchConfig.load(CONFIG)
    snapshot_dir = tmp_path / "snapshots"
    baseline = Workbench(DOCS, config, COMMANDS, snapshot_dir).run()
    orphan = snapshot_dir / "gen-999999999999"
    orphan.mkdir()
    (orphan / "state.json").write_text('{"schema_version": 1, "global_version": 999}', encoding="utf-8")
    temp_orphan = snapshot_dir / ".tmp-dead-process"
    temp_orphan.mkdir()
    (temp_orphan / "state.json").write_text('{"partial": true}', encoding="utf-8")
    resumed = Workbench(DOCS, config, COMMANDS, snapshot_dir).run()
    assert resumed.report["mode"] == "incremental"
    assert canonical_json(resumed.canonical) == canonical_json(baseline.canonical)


def test_invalid_document_and_command_codes(tmp_path: Path) -> None:
    loaded = load_documents(SAMPLES / "invalid_documents.jsonl")
    assert len(loaded.failures) == 4
    assert {failure["code"] for failure in loaded.failures} == {"E_DOC_INVALID"}
    assert list(loaded.documents) == ["dup"]

    run = _run(tmp_path, "invalid-commands", rebuild=True,
               commands=SAMPLES / "invalid_commands.jsonl")
    command_failures = [failure for failure in run.report["failures"] if failure["stage"] == "command"]
    assert len(command_failures) == 4
    assert {failure["code"] for failure in command_failures} == {"E_COMMAND_INVALID"}

    with pytest.raises(ConfigInvalid) as exc_info:
        WorkbenchConfig.load(SAMPLES / "invalid_config.yaml")
    assert exc_info.value.code == "E_CONFIG_INVALID"


def test_forced_kill_restart_uses_only_committed_generation(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    report = tmp_path / "partial_report.json"
    output = tmp_path / "partial_output.json"
    cmd = [
        sys.executable, "-m", "news_diag", "run",
        "--docs", str(DOCS), "--config", str(CONFIG), "--commands", str(COMMANDS),
        "--snapshot-dir", str(snapshot_dir), "--report", str(report),
        "--output", str(output), "--doc-delay-ms", "200", "--clean-snapshots",
    ]
    process = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True)
    current = snapshot_dir / "CURRENT"
    deadline = time.time() + 10
    while time.time() < deadline and not current.exists():
        time.sleep(0.02)
    assert current.exists()
    committed_name = current.read_text(encoding="utf-8").strip()
    os.kill(process.pid, signal.SIGKILL)
    process.communicate(timeout=10)
    assert process.returncode == -signal.SIGKILL
    assert (snapshot_dir / committed_name / "state.json").is_file()

    config = WorkbenchConfig.load(CONFIG)
    resumed = Workbench(DOCS, config, COMMANDS, snapshot_dir).run()
    full = Workbench(DOCS, config, COMMANDS, tmp_path / "full").run(rebuild=True)
    assert resumed.report["mode"] == "incremental"
    assert canonical_json(resumed.canonical) == canonical_json(full.canonical)
