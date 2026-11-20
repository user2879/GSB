"""Requirement 8: atomic snapshots, recovery and per-chunk failure handling."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from asr_stream_editor.engine import StreamingEngine
from asr_stream_editor.runner import load_resources
from asr_stream_editor.snapshot import SnapshotStore, STAGES

ROOT = Path(__file__).resolve().parents[1]


def _engine(samples_dir, work_dir, fault=None, budget=100000):
    pm, ac, sc, clips, _ = load_resources(samples_dir)
    clip = clips[0]
    store = SnapshotStore(os.path.join(work_dir, "snap", clip.source))
    return StreamingEngine(clip, pm, ac, sc, store, fault or {}), pm, ac, sc, clip, store


def _drain(engine):
    response = engine.process_command({"command": "initial_decode"})
    while response.code is not None:
        response = engine.process_command({"command": "resume"})
    return response


def test_snapshot_commit_is_versioned_and_complete(samples_dir, work_dir):
    engine, pm, ac, sc, clip, store = _engine(samples_dir, work_dir)
    _drain(engine)
    manifest = store.load_manifest()
    assert manifest["global_version"] >= 1
    assert manifest["audio_fingerprint"]
    assert manifest["config_hash"]
    assert sorted(manifest["stages"].keys()) == sorted(STAGES)
    for stage in STAGES:
        assert manifest["stages"][stage]["version"] == manifest["global_version"]
        assert manifest["stages"][stage]["files"]
    # active directory contains only complete swapped snapshots
    assert os.path.isdir(store.active_path)
    assert not os.listdir(store.tmp_root)


def test_recovery_reports_resume_point(samples_dir, work_dir):
    engine, pm, ac, sc, clip, store = _engine(samples_dir, work_dir)
    _drain(engine)
    fresh_store = SnapshotStore(os.path.join(work_dir, "snap", clip.source))
    point = fresh_store.recovery_point()
    assert point is not None
    assert point["recovery_chunk_index"] == len(engine.chunks) - 1
    assert point["global_version"] == engine.version
    assert point["audio_fingerprint"] == clip.fingerprint


def test_chunk_failure_is_recorded_and_other_chunks_continue(samples_dir, work_dir):
    faults = {"fail_chunks": {"3": {"stage": "decode",
                                    "message": "injected decode failure"}}}
    engine, pm, ac, sc, clip, store = _engine(samples_dir, work_dir, faults)
    response = _drain(engine)
    payload = response.to_dict()
    assert payload["failures"]
    record = payload["failures"][0]
    assert record["chunk_index"] == 3
    assert record["stage"] == "decode"
    assert record["code"] == "E_DECODE_FAILED"
    assert record["exception_type"] == "RuntimeError"
    # remaining chunks still produced tokens
    assert len(engine.chunk_results) == len(engine.chunks)
    assert engine.chunk_results[3].failed
    assert not engine.chunk_results[0].failed


def test_feature_failure_blocks_decode_without_killing_run(samples_dir, work_dir):
    faults = {"fail_chunks": {"2": {"stage": "features",
                                    "message": "injected feature failure"}}}
    engine, *_ = _engine(samples_dir, work_dir, faults)
    response = _drain(engine)
    codes = {item["code"] for item in response.to_dict()["failures"]}
    assert "E_FEATURE_FAILED" in codes
    assert response.to_dict()["transcript"]


def _crash_subprocess(samples_dir, work_dir, crash_chunk):
    script = f"""
import sys
sys.path.insert(0, {str(ROOT)!r})
import dataclasses
from asr_stream_editor.runner import load_resources
from asr_stream_editor.snapshot import SnapshotStore
from asr_stream_editor.engine import StreamingEngine
import os
pm, ac, sc, clips, _ = load_resources({samples_dir!r})
sc = dataclasses.replace(sc, redecode_budget_ms=2)
clip = clips[0]
store = SnapshotStore(os.path.join({work_dir!r}, "snap", clip.source))
engine = StreamingEngine(clip, pm, ac, sc, store,
    {{"crash_chunks": {{"{crash_chunk}": {{"stage": "feature", "exit_code": 9}}}}}})
response = engine.process_command({{"command": "initial_decode"}})
while response.code is not None:
    response = engine.process_command({{"command": "resume"}})
"""
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    return proc


def test_hard_kill_recovery_uses_only_complete_snapshot(samples_dir, work_dir):
    crash_work = os.path.join(work_dir, "crash")
    proc = _crash_subprocess(samples_dir, crash_work, crash_chunk=6)
    assert proc.returncode == 9
    pm, ac, sc, clips, _ = load_resources(samples_dir)
    store = SnapshotStore(os.path.join(crash_work, "snap", clips[0].source))
    point = store.recovery_point()
    assert point is not None
    assert point["recovery_chunk_index"] < 6
    # no half-built temp directories survive the swap
    assert os.listdir(store.tmp_root) == []


def test_graceful_exit_signal_finishes_chunk_and_exits_zero(samples_dir, work_dir):
    script = f"""
import sys, time, signal, os
sys.path.insert(0, {str(ROOT)!r})
from asr_stream_editor.runner import load_resources
from asr_stream_editor.snapshot import SnapshotStore
from asr_stream_editor.engine import StreamingEngine
pm, ac, sc, clips, _ = load_resources({samples_dir!r})
clip = clips[0]
store = SnapshotStore(os.path.join({work_dir!r}, "snap", clip.source))
engine = StreamingEngine(clip, pm, ac, sc, store,
    {{"exit_after_chunk": 4}})
response = engine.process_command({{"command": "initial_decode"}})
while response.code is not None:
    response = engine.process_command({{"command": "resume"}})
"""
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert proc.returncode == 0


def test_sigterm_finishes_commit_and_exits_zero(samples_dir, work_dir):
    import shutil
    case_dir = os.path.join(work_dir, "sigterm")
    os.makedirs(case_dir, exist_ok=True)
    for name in os.listdir(samples_dir):
        source = os.path.join(samples_dir, name)
        if os.path.isfile(source):
            shutil.copy(source, os.path.join(case_dir, name))
    shutil.copytree(os.path.join(samples_dir, "replacements"),
                    os.path.join(case_dir, "replacements"), dirs_exist_ok=True)
    config_path = os.path.join(case_dir, "stream_config.json")
    config = json.load(open(config_path))
    config["redecode_budget_ms"] = 1
    json.dump(config, open(config_path, "w"), sort_keys=True)
    script = f"""
import sys, signal
sys.path.insert(0, {str(ROOT)!r})
from asr_stream_editor.runner import load_resources
from asr_stream_editor.snapshot import SnapshotStore
from asr_stream_editor.engine import StreamingEngine
import dataclasses, os
pm, ac, sc, clips, _ = load_resources({case_dir!r})
sc = dataclasses.replace(sc, redecode_budget_ms=1)
gate = {{"requested": False}}
signal.signal(signal.SIGTERM, lambda signum, frame: gate.__setitem__("requested", True))
clip = clips[0]
store = SnapshotStore(os.path.join({case_dir!r}, "out", "snap", clip.source))
engine = StreamingEngine(clip, pm, ac, sc, store, {{}},
                         should_exit=lambda: gate["requested"])
for index in range(60):
    engine.process_command({{"command": "initial_decode" if index == 0 else "resume"}})
"""
    proc = subprocess.Popen([sys.executable, "-c", script],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(0.4)
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=10)
    assert proc.returncode == 0
    pm, ac, sc, clips, _ = load_resources(case_dir)
    store = SnapshotStore(os.path.join(case_dir, "out", "snap", clips[0].source))
    point = store.recovery_point()
    assert point is not None
    assert sorted(point["stages"]) == sorted(STAGES)
