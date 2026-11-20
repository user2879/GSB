from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import SchedulerParams
from .scheduler import Engine, compare_run_outcomes
from .scripts import load_script
from .snapshot import replay_snapshot, restore_snapshot, save_snapshot
from .storage import SnapshotStore, Storage


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLES = PROJECT_ROOT / "samples"


def _value_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def render_result(result: Any, *, include_commits: bool = True) -> list[str]:
    lines = [
        f"  script={result.script_id} source_version={result.source_version} normalized_version={result.normalized_version}",
        (
            "  params="
            f"global_limit:{result.params.global_concurrency_limit},"
            f"quantum:{result.params.clock_quantum_ms},"
            f"parameter_version:{result.params.parameter_version}"
        ),
        f"  run_id={result.run_id}",
        f"  data_version={result.global_data_version}",
        "  sessions:",
    ]
    for outcome in result.outcomes:
        lines.append(
            "    "
            f"{outcome.start_seq}. {outcome.session_id} outcome={outcome.outcome} "
            f"code={outcome.business_code or '-'} isolation={outcome.isolation_level} "
            f"wait={outcome.wait_used_ms}ms commit_version={outcome.committed_data_version}"
        )
        lines.append(
            "       ops="
            + ",".join(
                f"{operation.op_no}:{operation.kind}:{operation.status}"
                + (f":v{operation.data_version}" if operation.data_version is not None else "")
                + (f":wait{operation.wait_ms}" if operation.wait_ms else "")
                for operation in outcome.operations
            )
        )
    if include_commits:
        lines.append("  commits:")
        if not result.commits:
            lines.append("    (none)")
        for commit in result.commits:
            rows = ",".join(
                f"{write['row_key']}={_value_text(write['value'])}"
                for write in commit.writes
            )
            lines.append(
                f"    commit_seq={commit.commit_seq} data_version={commit.data_version} "
                f"session={commit.session_id} virtual_ms={commit.virtual_ms} writes={rows}"
            )
    lines.append("  final_rows:")
    for row_key in sorted(result.final_rows):
        row = result.final_rows[row_key]
        lines.append(
            f"    {row_key}={_value_text(row['value'])} version={row['version']}"
        )
    lines.append(f"  remaining_lock_entries={len(result.lock_state.get('rows', []))}")
    return lines


def _run_file(
    path: Path,
    db_path: Path,
    params: SchedulerParams,
) -> tuple[Any, Storage]:
    script = load_script(path)
    storage = Storage(db_path)
    result = Engine(storage, params).run(script)
    return result, storage


def _scenario_snapshot_checks(
    label: str,
    script_path: Path,
    workdir: Path,
    params: SchedulerParams,
) -> list[str]:
    filename = label.replace("-", "_")
    db_path = workdir / f"{filename}.sqlite"
    snapshot_path = workdir / f"{filename}-snapshots.sqlite"
    restore_db = workdir / f"{filename}-restore.sqlite"
    result, storage = _run_file(script_path, db_path, params)
    script = load_script(script_path)
    with SnapshotStore(snapshot_path) as store:
        save_snapshot(store, f"{label}-final", script, result)
        replay_report, _replay_result = replay_snapshot(store, f"{label}-final")
        with Storage(restore_db) as target:
            restore_report = restore_snapshot(
                store,
                f"{label}-final",
                target_storage=target,
            )
    lines = render_result(result)
    lines.append(
        f"  snapshot={label}-final saved; replay="
        f"{'REPRODUCIBLE' if replay_report.reproducible else 'NON_REPRODUCIBLE'}"
    )
    lines.append(
        "  restore="
        f"mode:{restore_report.mode},"
        f"data_version:{restore_report.data_version},"
        f"changed_sessions:{','.join(restore_report.changed_sessions) or '-'},"
        f"changed_params:{','.join(restore_report.changed_parameter_items) or '-'},"
        f"cleared_lock_entries:{restore_report.cleared_lock_entries}"
    )
    storage.close()
    return lines


def _legacy_upgrade_lines(
    label: str,
    source_name: str,
    workdir: Path,
    params: SchedulerParams,
) -> list[str]:
    source_path = SAMPLES / source_name
    source_script = load_script(source_path)
    source_result, source_storage = _run_file(
        source_path,
        workdir / f"{label}-source.sqlite",
        params,
    )
    upgraded = source_script.to_dict(include_source=True)
    upgraded_path = workdir / f"{label}-upgraded.json"
    upgraded_path.write_text(
        json.dumps(upgraded, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    upgraded_result, upgraded_storage = _run_file(
        upgraded_path,
        workdir / f"{label}-upgraded.sqlite",
        params,
    )
    differences = compare_run_outcomes(source_result, upgraded_result)
    lines = [
        f"  {label}: source_format={source_script.source_version} upgraded_format=3",
        "  source_run:",
        *render_result(source_result, include_commits=False),
        "  upgraded_run:",
        *render_result(upgraded_result, include_commits=False),
        "  upgrade_equivalence=" + ("OK" if not differences else "DIFF"),
    ]
    if differences:
        for difference in differences:
            lines.append(
                f"    diff field={difference['field']} "
                f"left={_value_text(difference['left'])} right={_value_text(difference['right'])}"
            )
    source_storage.close()
    upgraded_storage.close()
    return lines


def run_demo(workdir: str | Path | None = None) -> str:
    workdir_path = Path(workdir) if workdir is not None else PROJECT_ROOT / "runtime" / "demo"
    workdir_path.mkdir(parents=True, exist_ok=True)
    # Recreate only this demo's database artifacts while preserving unrelated files.
    for path in sorted(workdir_path.glob("*.sqlite*")):
        path.unlink(missing_ok=True)

    lines: list[str] = [
        "TRANSACTION ISOLATION AND LOCK LAB - DETERMINISTIC DEMO",
        f"workdir={workdir_path}",
        "order=normal,lock_timeout,deadlock,legacy_upgrade",
    ]
    params = SchedulerParams(global_concurrency_limit=2, clock_quantum_ms=1)

    lines.append("")
    lines.append("[1] normal concurrent editing")
    lines.extend(
        _scenario_snapshot_checks(
            "01-normal",
            SAMPLES / "current-v3.json",
            workdir_path,
            params,
        )
    )

    lines.append("")
    lines.append("[2] lock-wait timeout")
    lines.extend(
        _scenario_snapshot_checks(
            "02-timeout",
            SAMPLES / "lock-timeout-v3.json",
            workdir_path,
            params,
        )
    )

    lines.append("")
    lines.append("[3] deadlock arbitration")
    lines.extend(
        _scenario_snapshot_checks(
            "03-deadlock",
            SAMPLES / "deadlock-v3.json",
            workdir_path,
            params,
        )
    )

    lines.append("")
    lines.append("[4] legacy v1/v2 upgrade then scheduling")
    lines.extend(_legacy_upgrade_lines("v1", "legacy-v1.json", workdir_path, params))
    lines.append("")
    lines.extend(_legacy_upgrade_lines("v2", "legacy-v2.json", workdir_path, params))

    report = "\n".join(lines) + "\n"
    (workdir_path / "demo_report.txt").write_text(report, encoding="utf-8")
    return report
