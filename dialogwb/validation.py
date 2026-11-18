from __future__ import annotations

import copy
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .config import load_config, normalize_config
from .edits import apply_batch, load_edit_batches
from .engine import Engine, core_view
from .lexicon import load_lexicon
from .loader import load_jsonl, normalize_records
from .samplegen import generate_sample
from .snapshot import SnapshotManager
from .util import atomic_write_json
from .workbench import ConfigMismatchError, Workbench, build_report


def run_validation(
    *, samples_dir: str | Path, out_dir: str | Path, cancellation_rounds: int = 3
) -> dict[str, Any]:
    samples = Path(samples_dir)
    output = Path(out_dir)
    if not (samples / "events.jsonl").exists():
        generate_sample(samples)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    _copy_samples(samples, output / "samples")

    events_path = samples / "events.jsonl"
    lexicon_path = samples / "lexicon.json"
    config_path = samples / "config.json"
    edits_path = samples / "edits.json"
    lexicon = load_lexicon(lexicon_path)
    base_config = load_config(config_path)
    base_loaded = load_jsonl(events_path, lexicon)
    batches = load_edit_batches(edits_path)
    states = _materialize_states(base_loaded, base_config, batches)

    full_workbench = Workbench(
        events_path=events_path,
        lexicon_path=lexicon_path,
        config_path=config_path,
        edits_path=edits_path,
        out_dir=output / "incremental_run",
        snapshot_dir=output / "incremental_snapshots",
    )
    incremental_final = full_workbench.run()
    history = full_workbench.stage_history
    if len(history) != len(states):
        raise RuntimeError("stage history does not cover baseline and all batches")

    batch_equivalence: list[dict[str, Any]] = []
    for index, (state, history_item) in enumerate(zip(states, history)):
        incremental_result = history_item["result"]
        full_result = _full_rebuild(lexicon, base_loaded, state)
        equal = core_view(incremental_result) == core_view(full_result)
        batch_equivalence.append(
            {
                "batch": index,
                "batch_id": "baseline" if index == 0 else batches[index - 1]["batch_id"],
                "equal_to_full_rebuild": equal,
                "first_difference": _first_difference(
                    core_view(full_result), core_view(incremental_result)
                ),
                "incremental_snapshot": incremental_result["recovery"][
                    "snapshot_version"
                ],
                "full_snapshot": full_result["recovery"]["snapshot_version"],
                "summary_round_count": len(incremental_result["summary_lengths"]),
                "reuse": {
                    session_id: session["incremental"]
                    for session_id, session in incremental_result["sessions"].items()
                },
            }
        )

    final_full = _full_rebuild(lexicon, base_loaded, states[-1])
    final_equal = core_view(incremental_final) == core_view(final_full)
    _write_result(output / "final_incremental", incremental_final)
    _write_result(output / "final_full_rebuild", final_full)

    concurrency = _validate_concurrency(
        lexicon, base_loaded, states[-1], output / "concurrency"
    )
    exception_isolation = _validate_exception_isolation(
        lexicon, base_loaded, states[-1]["config"], output / "exception_isolation.json"
    )
    graceful = _validate_graceful_signal(
        events_path,
        lexicon_path,
        config_path,
        edits_path,
        output,
        final_full,
    )
    forced = _validate_forced_kill(
        events_path,
        lexicon_path,
        config_path,
        edits_path,
        output,
        final_full,
    )
    explicit_cancel = _validate_explicit_cancel(
        events_path,
        lexicon_path,
        config_path,
        edits_path,
        output,
        final_full,
        cancellation_rounds,
    )

    failure_codes = [item["code"] for item in incremental_final["failures"]]
    required_codes = {
        "E_BAD_JSON",
        "E_EMPTY_UTTERANCE",
        "E_UNKNOWN_TOKEN",
        "E_TURN_GAP",
    }
    report = {
        "schema_version": 1,
        "baseline": {
            "status": history[0]["result"]["status"],
            "snapshot_version": history[0]["result"]["recovery"]["snapshot_version"],
            "summary_length_rounds": len(history[0]["result"]["summary_lengths"]),
            "counts": history[0]["result"]["counts"],
            "failure_codes": [item["code"] for item in history[0]["result"]["failures"]],
            "all_required_business_codes": required_codes.issubset(
                {item["code"] for item in history[0]["result"]["failures"]}
            ),
        },
        "final": {
            "incremental_snapshot": incremental_final["recovery"]["snapshot_version"],
            "full_snapshot": final_full["recovery"]["snapshot_version"],
            "equal_to_full_rebuild": final_equal,
            "first_difference": _first_difference(
                core_view(final_full), core_view(incremental_final)
            ),
            "counts": incremental_final["counts"],
            "failure_count": len(incremental_final["failures"]),
            "failure_codes": failure_codes,
            "rounds": len(incremental_final["summary_lengths"]),
        },
        "batch_equivalence": batch_equivalence,
        "concurrency": concurrency,
        "exception_isolation": exception_isolation,
        "graceful_signal": graceful,
        "explicit_cancellation": explicit_cancel,
        "forced_termination": forced,
        "config_mismatch": forced.get("config_mismatch"),
    }
    report["all_checks_passed"] = all(
        [
            report["baseline"]["all_required_business_codes"],
            final_equal,
            all(item["equal_to_full_rebuild"] for item in batch_equivalence),
            concurrency["equal_to_serial"],
            exception_isolation["no_later_reference"],
            graceful["resume_equal_to_uninterrupted"],
            explicit_cancel["resume_equal_to_uninterrupted"],
            forced["resume_equal_to_uninterrupted"],
            forced.get("forced_exit_code") == 137,
            bool(forced.get("config_mismatch", {}).get("rejected")),
            bool(forced.get("config_mismatch", {}).get("pointer_unchanged")),
        ]
    )
    atomic_write_json(output / "validation_report.json", report)
    _write_validation_readme(output, report)
    return report


def _materialize_states(
    base_loaded: Any,
    base_config: dict[str, Any],
    batches: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = [
        {
            "label": "baseline",
            "records": copy.deepcopy(base_loaded.records),
            "config": copy.deepcopy(base_config),
            "edit_failures": [],
            "order_counter": max(
                (int(item.get("_order", 0)) for item in base_loaded.records),
                default=0,
            ),
        }
    ]
    records = copy.deepcopy(base_loaded.records)
    config = copy.deepcopy(base_config)
    order_counter = states[0]["order_counter"]
    for batch in batches:
        records, config, failures, order_counter = apply_batch(
            records, config, batch, order_counter
        )
        states.append(
            {
                "label": batch["batch_id"],
                "records": records,
                "config": config,
                "edit_failures": failures,
                "order_counter": order_counter,
            }
        )
    return states


def _full_rebuild(lexicon: Any, base_loaded: Any, state: dict[str, Any]) -> dict[str, Any]:
    loaded = normalize_records(
        copy.deepcopy(state["records"]),
        lexicon,
        parse_failures=base_loaded.parse_failures,
    )
    return Engine(
        config=state["config"],
        lexicon=lexicon,
        loaded=loaded,
        mode="full",
        initial_failures=state["edit_failures"],
    ).run()


def _validate_concurrency(
    lexicon: Any, base_loaded: Any, state: dict[str, Any], directory: Path
) -> dict[str, Any]:
    loaded = normalize_records(
        copy.deepcopy(state["records"]),
        lexicon,
        parse_failures=base_loaded.parse_failures,
    )
    results: dict[int, dict[str, Any]] = {}
    for concurrency in (1, 4):
        config = normalize_config({**state["config"], "concurrency": concurrency})
        results[concurrency] = Engine(
            config=config,
            lexicon=lexicon,
            loaded=loaded,
            mode="full",
            initial_failures=state["edit_failures"],
        ).run()
    equal = core_view(
        results[1], include_stage_timing=True
    ) == core_view(results[4], include_stage_timing=True)
    directory.mkdir(parents=True, exist_ok=True)
    atomic_write_json(directory / "serial.json", results[1])
    atomic_write_json(directory / "concurrent.json", results[4])
    return {
        "equal_to_serial": equal,
        "first_difference": _first_difference(
            core_view(results[1], include_stage_timing=True),
            core_view(results[4], include_stage_timing=True),
        ),
        "serial_stage_timing": results[1]["stage_timing"],
        "concurrent_stage_timing": results[4]["stage_timing"],
        "serial_observed_wall_clock_ms": results[1]["runtime_metrics"][
            "observed_wall_clock_ms"
        ],
        "concurrent_observed_wall_clock_ms": results[4]["runtime_metrics"][
            "observed_wall_clock_ms"
        ],
    }


def _validate_exception_isolation(
    lexicon: Any, base_loaded: Any, config: dict[str, Any], report_path: Path
) -> dict[str, Any]:
    def fault(stage: str, session_id: str, turn: int) -> None:
        if stage == "merge" and session_id == "s01_plan" and turn == 1:
            raise RuntimeError("injected merge failure")

    result = Engine(
        config=config,
        lexicon=lexicon,
        loaded=base_loaded,
        mode="full",
        fault_hook=fault,
    ).run()
    session = result["sessions"]["s01_plan"]
    stage_failures = [
        item
        for item in result["failures"]
        if item["code"] == "E_STAGE_EXCEPTION"
        and item["session_id"] == "s01_plan"
        and item["turn"] == 1
        and item["stage"] == "merge"
    ]
    later_turns = [item for item in session["rounds"] if item["turn"] > 1]
    later_references_turn_one = any(
        source["turn"] == 1
        for round_item in later_turns
        for source in round_item["source_map"]
        for source in source["sources"]
    ) or any(
        event["turn"] == 1
        for round_item in later_turns
        for event in round_item["extraction"].get("considered_events", [])
    )
    payload = {
        "stage_failure": stage_failures[0] if stage_failures else None,
        "later_reference_to_failed_turn": later_references_turn_one,
        "no_later_reference": bool(stage_failures) and not later_references_turn_one,
        "result_status": result["status"],
    }
    atomic_write_json(report_path, payload)
    return payload


def _validate_graceful_signal(
    events_path: Path,
    lexicon_path: Path,
    config_path: Path,
    edits_path: Path,
    output: Path,
    uninterrupted: dict[str, Any],
) -> dict[str, Any]:
    out_dir = output / "graceful_signal" / "out"
    snapshot_dir = output / "graceful_signal" / "snapshot"
    command = _cli_command(
        events_path,
        lexicon_path,
        config_path,
        edits_path,
        out_dir,
        snapshot_dir,
        extra=["--fresh"],
    )
    process = subprocess.Popen(
        command,
        cwd=str(Path.cwd()),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    pointer = snapshot_dir / "CURRENT"
    deadline = time.time() + 10
    while time.time() < deadline and process.poll() is None and not pointer.exists():
        time.sleep(0.01)
    if pointer.exists() and process.poll() is None:
        process.send_signal(signal.SIGTERM)
    stdout, stderr = process.communicate(timeout=60)
    interrupted_result = None
    result_path = out_dir / "result.json"
    if result_path.exists():
        interrupted_result = json.loads(result_path.read_text(encoding="utf-8"))

    resumed = Workbench(
        events_path=events_path,
        lexicon_path=lexicon_path,
        config_path=config_path,
        edits_path=edits_path,
        out_dir=output / "graceful_signal" / "resumed_out",
        snapshot_dir=snapshot_dir,
    ).run(resume=True)
    equal = core_view(resumed) == core_view(uninterrupted)
    payload = {
        "exit_code": process.returncode,
        "interrupted_status": interrupted_result.get("status")
        if interrupted_result
        else None,
        "interrupted_recovery": interrupted_result.get("recovery")
        if interrupted_result
        else None,
        "resumed_status": resumed["status"],
        "resumed_from": resumed.get("resumed_from"),
        "resume_equal_to_uninterrupted": equal,
        "stdout": stdout.strip().splitlines()[-5:],
        "stderr": stderr.strip().splitlines()[-5:],
    }
    atomic_write_json(output / "graceful_signal" / "validation.json", payload)
    return payload


def _validate_explicit_cancel(
    events_path: Path,
    lexicon_path: Path,
    config_path: Path,
    edits_path: Path,
    output: Path,
    uninterrupted: dict[str, Any],
    cancellation_rounds: int,
) -> dict[str, Any]:
    directory = output / "explicit_cancellation"
    workbench = Workbench(
        events_path=events_path,
        lexicon_path=lexicon_path,
        config_path=config_path,
        edits_path=edits_path,
        out_dir=directory / "out",
        snapshot_dir=directory / "snapshot",
    )
    interrupted = workbench.run(stop_after_rounds=cancellation_rounds)
    resumed = Workbench(
        events_path=events_path,
        lexicon_path=lexicon_path,
        config_path=config_path,
        edits_path=edits_path,
        out_dir=directory / "resumed_out",
        snapshot_dir=directory / "snapshot",
    ).run(resume=True)
    payload = {
        "requested_rounds": cancellation_rounds,
        "interrupted_status": interrupted["status"],
        "interrupted_recovery": interrupted["recovery"],
        "resumed_status": resumed["status"],
        "resumed_from": resumed.get("resumed_from"),
        "resume_equal_to_uninterrupted": core_view(resumed)
        == core_view(uninterrupted),
    }
    atomic_write_json(directory / "validation.json", payload)
    return payload


def _validate_forced_kill(
    events_path: Path,
    lexicon_path: Path,
    config_path: Path,
    edits_path: Path,
    output: Path,
    uninterrupted: dict[str, Any],
) -> dict[str, Any]:
    directory = output / "forced_termination"
    out_dir = directory / "out"
    snapshot_dir = directory / "snapshot"
    command = _cli_command(
        events_path,
        lexicon_path,
        config_path,
        edits_path,
        out_dir,
        snapshot_dir,
        extra=["--fresh", "--simulate-kill-after-snapshots", "5"],
    )
    process = subprocess.run(
        command,
        cwd=str(Path.cwd()),
        capture_output=True,
        text=True,
        timeout=60,
    )
    manager = SnapshotManager(snapshot_dir)
    pointer = manager.load_current()
    pointer_before = pointer["manifest"] if pointer else None

    mismatch_config = directory / "config_mismatch.json"
    changed_config = json.loads(config_path.read_text(encoding="utf-8"))
    changed_config["compression_window"] = int(
        changed_config.get("compression_window", 3)
    ) + 1
    mismatch_config.write_text(
        json.dumps(changed_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    mismatch_rejected = False
    mismatch_error = None
    try:
        Workbench(
            events_path=events_path,
            lexicon_path=lexicon_path,
            config_path=mismatch_config,
            edits_path=edits_path,
            out_dir=directory / "mismatch_out",
            snapshot_dir=snapshot_dir,
        ).run(resume=True)
    except ConfigMismatchError as exc:
        mismatch_rejected = True
        mismatch_error = {"code": exc.code, "message": str(exc)}
    pointer_after = SnapshotManager(snapshot_dir).load_current()
    pointer_unchanged = (
        pointer_before is not None
        and pointer_after is not None
        and pointer_before["snapshot_version"]
        == pointer_after["manifest"]["snapshot_version"]
    )

    resumed = Workbench(
        events_path=events_path,
        lexicon_path=lexicon_path,
        config_path=config_path,
        edits_path=edits_path,
        out_dir=directory / "resumed_out",
        snapshot_dir=snapshot_dir,
    ).run(resume=True)
    payload = {
        "forced_exit_code": process.returncode,
        "snapshot_before_resume": pointer_before,
        "resumed_status": resumed["status"],
        "resumed_from": resumed.get("resumed_from"),
        "resume_equal_to_uninterrupted": core_view(resumed)
        == core_view(uninterrupted),
        "config_mismatch": {
            "rejected": mismatch_rejected,
            "error": mismatch_error,
            "pointer_unchanged": pointer_unchanged,
        },
    }
    atomic_write_json(directory / "validation.json", payload)
    return payload


def _cli_command(
    events_path: Path,
    lexicon_path: Path,
    config_path: Path,
    edits_path: Path,
    out_dir: Path,
    snapshot_dir: Path,
    *,
    extra: list[str],
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "dialogwb",
        "run",
        "--events",
        str(events_path),
        "--lexicon",
        str(lexicon_path),
        "--config",
        str(config_path),
        "--edits",
        str(edits_path),
        "--out",
        str(out_dir),
        "--snapshot",
        str(snapshot_dir),
        *extra,
    ]


def _write_result(directory: Path, result: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    atomic_write_json(directory / "result.json", result)
    atomic_write_json(
        directory / "report.json",
        build_report(result, snapshot_info=None),
    )


def _copy_samples(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("events.jsonl", "lexicon.json", "config.json", "edits.json"):
        shutil.copy2(source / name, destination / name)


def _first_difference(left: Any, right: Any, path: str = "$") -> dict[str, Any] | None:
    if type(left) is not type(right):
        return {"path": path, "left_type": type(left).__name__, "right_type": type(right).__name__}
    if isinstance(left, dict):
        for key in sorted(set(left) | set(right)):
            if key not in left:
                return {"path": f"{path}.{key}", "left": "<missing>", "right": right[key]}
            if key not in right:
                return {"path": f"{path}.{key}", "left": left[key], "right": "<missing>"}
            difference = _first_difference(left[key], right[key], f"{path}.{key}")
            if difference is not None:
                return difference
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return {"path": f"{path}.length", "left": len(left), "right": len(right)}
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            difference = _first_difference(
                left_item, right_item, f"{path}[{index}]"
            )
            if difference is not None:
                return difference
        return None
    if left != right:
        return {"path": path, "left": left, "right": right}
    return None


def _write_validation_readme(output: Path, report: dict[str, Any]) -> None:
    text = f"""# 实测产物

- `validation_report.json`：完整验收矩阵，`all_checks_passed` = `{report['all_checks_passed']}`。
- `final_incremental/`：三批编辑后的增量最终结果。
- `final_full_rebuild/`：相同最终输入与配置的全量重建结果。
- `concurrency/`：并发 1 与并发 4 的独立结果。
- `graceful_signal/`：SIGTERM 优雅退出与续跑验证。
- `explicit_cancellation/`：可处理轮次取消与续跑验证。
- `forced_termination/`：子进程强制退出、配置拒绝与恢复验证。
- `exception_isolation.json`：阶段异常隔离结果。
- `samples/`：固定种子输入副本。
- `unittest.log`：回归测试日志。

复现：

```bash
python3 -m dialogwb validate --samples samples --out artifacts/validation
```
"""
    (output / "README.md").write_text(text, encoding="utf-8")
