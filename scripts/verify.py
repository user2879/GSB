from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rope_audit.cli import build_fixture_config  # noqa: E402
from rope_audit.config import parse_config  # noqa: E402
from rope_audit.engine import AuditEngine  # noqa: E402
from rope_audit.samples import generate_samples  # noqa: E402
from rope_audit.snapshot import SnapshotStore  # noqa: E402


def run_command(args: list[str], stdout_path: Path | None = None, stderr_path: Path | None = None) -> dict:
    process = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=False)
    if stdout_path:
        stdout_path.write_text(process.stdout, encoding="utf-8")
    if stderr_path:
        stderr_path.write_text(process.stderr, encoding="utf-8")
    return {
        "args": args,
        "returncode": process.returncode,
        "stdout": process.stdout,
        "stderr": process.stderr,
    }


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    runs = ROOT / "artifacts" / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    summary: dict = {"environment": {"python": sys.version.split()[0], "platform": platform.platform()}}
    try:
        import numpy as np

        summary["environment"]["numpy"] = np.__version__
    except ImportError:
        summary["environment"]["numpy"] = "missing"

    pytest_result = run_command([sys.executable, "-m", "pytest", "-q"])
    summary["pytest"] = {
        "returncode": pytest_result["returncode"],
        "tail": "\n".join(pytest_result["stdout"].splitlines()[-5:]),
    }

    generate = run_command([sys.executable, "-m", "rope_audit", "generate", "--out-dir", "examples"])
    summary["generate"] = json.loads(generate["stdout"]) if generate["returncode"] == 0 else generate

    validate = run_command(
        [
            sys.executable,
            "-m",
            "rope_audit",
            "validate",
            "--config",
            "examples/config.json",
            "--samples",
            "examples/samples.jsonl",
            "--commands",
            "examples/commands.jsonl",
        ]
    )
    summary["validate"] = json.loads(validate["stdout"]) if validate["returncode"] == 0 else validate

    replay_path = runs / "replay-report.json"
    replay = run_command(
        [
            sys.executable,
            "-m",
            "rope_audit",
            "replay",
            "--config",
            "examples/config.json",
            "--samples",
            "examples/samples.jsonl",
            "--commands",
            "examples/commands.jsonl",
            "--output",
            str(replay_path.relative_to(ROOT)),
            "--workers",
            "2",
            "--no-snapshot",
        ]
    )
    summary["replay_cli"] = json.loads(replay["stdout"]) if replay["returncode"] == 0 else replay
    replay_report = load_json(replay_path) if replay_path.exists() else {}
    final_report = replay_report.get("incremental_final", {}).get("report", {})

    budget_path = runs / "budget-pending.json"
    budget = run_command(
        [
            sys.executable,
            "-m",
            "rope_audit",
            "run",
            "--config",
            "examples/config.json",
            "--samples",
            "examples/samples.jsonl",
            "--no-snapshot",
            "--batch-budget",
            "96",
            "--output",
            str(budget_path.relative_to(ROOT)),
        ]
    )
    budget_report = load_json(budget_path)["final_report"]
    summary["budget_first_batch"] = {
        "cli": json.loads(budget["stdout"]),
        "pending": budget_report["budget"]["pending_sample_ids"],
        "consumed": budget_report["budget"]["last_batch_consumed_points"],
        "codes": [item["code"] for item in budget_report["budget"]["plan"]["pending"]],
    }

    drain_path = runs / "budget-drain.json"
    drain = run_command(
        [
            sys.executable,
            "-m",
            "rope_audit",
            "run",
            "--config",
            "examples/config.json",
            "--samples",
            "examples/samples.jsonl",
            "--no-snapshot",
            "--drain",
            "--batch-budget",
            "96",
            "--output",
            str(drain_path.relative_to(ROOT)),
        ]
    )
    drain_output = load_json(drain_path)
    summary["budget_drain"] = {
        "cli": json.loads(drain["stdout"]),
        "batch_count": drain_output["batch_count"],
        "pending": drain_output["final_report"]["budget"]["pending_sample_ids"],
    }

    quota_path = runs / "quota-exceeded.json"
    run_command(
        [
            sys.executable,
            "-m",
            "rope_audit",
            "run",
            "--config",
            "examples/config.json",
            "--samples",
            "examples/samples.jsonl",
            "--no-snapshot",
            "--batch-budget",
            "10000",
            "--quota",
            "40",
            "--output",
            str(quota_path.relative_to(ROOT)),
        ]
    )
    quota_report = load_json(quota_path)["final_report"]
    summary["quota"] = {
        "failure_count": len(quota_report["failures"]),
        "codes": sorted({item["code"] for item in quota_report["failures"]}),
    }

    injection_stdout = runs / "injection-summary.json"
    injection_stderr = runs / "injection.stderr"
    injection_output_path = runs / "injection-run.json"
    injection = run_command(
        [
            sys.executable,
            "-m",
            "rope_audit",
            "run",
            "--config",
            "examples/config.json",
            "--samples",
            "examples/samples.jsonl",
            "--injections",
            "examples/injections.json",
            "--no-snapshot",
            "--drain",
            "--batch-budget",
            "10000",
            "--quota",
            "1000",
            "--output",
            str(injection_output_path.relative_to(ROOT)),
        ],
        stderr_path=injection_stderr,
    )
    injection_cli = json.loads(injection["stdout"])
    injection_report = load_json(injection_output_path)["final_report"]
    summary["injection"] = {
        "cli": injection_cli,
        "stderr": injection_stderr.read_text(encoding="utf-8").strip(),
        "failures": injection_report["failures"],
    }

    raw_config = build_fixture_config()
    raw_config["budget"] = {"batch_point_budget": 10_000, "sample_quota_points": 10_000}
    config = parse_config(raw_config)
    samples = generate_samples(config, count=2)
    shutil.rmtree(ROOT / "artifacts" / "snapshot-recovery-demo", ignore_errors=True)
    engine = AuditEngine(config, samples)
    engine.run_batch(workers=1)
    store = SnapshotStore(ROOT / "artifacts" / "snapshot-recovery-demo")
    store.commit(engine, engine.last_report)
    prepared = store.prepare_transaction(engine, engine.last_report)
    prepared_existed_before_recovery = prepared.exists()
    recovery_after_crash = store.recover(engine.config_hash)
    pointer_removed = False
    if store.current_pointer.exists():
        store.current_pointer.unlink()
        pointer_removed = True
    recovery_after_pointer_loss = store.recover(engine.config_hash)

    shutil.rmtree(ROOT / "artifacts" / "snapshot-kill-demo", ignore_errors=True)
    killed = run_command([sys.executable, str(ROOT / "scripts" / "crash_demo.py")])
    killed_temp = Path(killed["stdout"].strip().splitlines()[-1]) if killed["stdout"].strip() else None
    survived_sigkill = bool(killed_temp and killed_temp.exists())
    kill_store = SnapshotStore(ROOT / "artifacts" / "snapshot-kill-demo")
    kill_recovery = kill_store.recover()
    summary["snapshot"] = {
        "prepared_uncommitted_dir": str(prepared.relative_to(ROOT)),
        "prepared_existed": prepared_existed_before_recovery,
        "after_recover_status": recovery_after_crash["status"],
        "orphans_removed": len(recovery_after_crash["orphaned_temp_dirs_removed"]),
        "pointer_removed": pointer_removed,
        "pointer_loss_status": recovery_after_pointer_loss["status"],
        "global_version": recovery_after_pointer_loss["global_version"],
        "sigkill_returncode": killed["returncode"],
        "sigkill_temp_dir": str(killed_temp.relative_to(ROOT)) if killed_temp else None,
        "temp_survived_sigkill": survived_sigkill,
        "sigkill_recovery_status": kill_recovery["status"],
        "sigkill_orphans_removed": len(kill_recovery["orphaned_temp_dirs_removed"]),
        "temp_cleaned_after_recover": bool(killed_temp and not killed_temp.exists()),
    }

    summary_path = runs / "verification-summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    lines: list[str] = []
    lines.append("# 实际验证结果")
    lines.append("")
    lines.append(f"- Python: `{summary['environment']['python']}` on {summary['environment']['platform']}")
    lines.append(f"- NumPy: `{summary['environment'].get('numpy')}`")
    lines.append(f"- pytest: returncode `{summary['pytest']['returncode']}`")
    lines.append("```text")
    lines.append(summary["pytest"]["tail"])
    lines.append("```")
    lines.append("")
    lines.append("## 命令流一致性")
    eq = replay_report.get("equivalence", {})
    lines.append(f"- incremental vs blank full serial: `{eq.get('incremental_vs_blank_full_serial')}`")
    lines.append(f"- serial vs threaded: `{eq.get('serial_vs_threaded')}`")
    lines.append("")
    lines.append("| step | op | batches | position hit/miss | rotation hit/miss | dot hit/miss | norm hit/miss | summary hit/miss |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")
    for step in replay_report.get("steps", []):
        counts = step["ledger_counts"]

        def hm(stage: str) -> str:
            item = counts.get(stage, {})
            return f"{item.get('hit', 0)}/{item.get('miss', 0)}"

        lines.append(
            f"| {step['step']} | `{step['command'].get('op')}` | {step['batch_count']} | {hm('position')} | "
            f"{hm('rotation')} | {hm('dot')} | {hm('normalization')} | {hm('summary')} |"
        )
    lines.append("")
    lines.append("## 终态分层统计")
    lines.append("")
    lines.append("| scheme | stratum | samples | keys | score mean | max abs score error | breach stages |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for scheme_name, stats in final_report.get("statistics", {}).items():
        breach_total = sum(stats.get("first_breach_stage_counts", {}).values())
        for stratum_name in ("training", "extrapolation"):
            bucket = stats["strata"][stratum_name]
            mean = bucket["score_mean"]
            err = bucket["score_max_abs_error_vs_truth"]
            lines.append(
                f"| `{scheme_name}` | {stratum_name} | {bucket['sample_count']} | {bucket['key_count']} | "
                f"{mean:.8f} | {err:.3e} | {breach_total} |"
                if mean is not None and err is not None
                else f"| `{scheme_name}` | {stratum_name} | 0 | 0 | - | - | {breach_total} |"
            )
    lines.append("")
    lines.append(f"- 外推位置交叉对照行数: `{len(final_report.get('cross_scheme_extrapolation_positions', []))}`")
    coarse_step = next((step for step in replay_report.get("steps", []) if step["step"] == 9), None)
    if coarse_step:
        lines.append("")
        lines.append("## 粗量化档位首次超差（命令流第 9 步）")
        lines.append("")
        lines.append("| scheme | sample | first stage | value | tolerance |")
        lines.append("|---|---|---|---:|---:|")
        for scheme_name, breaches in coarse_step.get("first_breaches", {}).items():
            for breach in breaches[:2]:
                lines.append(
                    f"| `{scheme_name}` | `{breach.get('sample_id')}` | `{breach.get('stage')}` | "
                    f"{breach.get('value'):.6e} | {breach.get('tolerance'):.6e} |"
                )
    cross_rows = final_report.get("cross_scheme_extrapolation_positions", [])
    if cross_rows:
        row = cross_rows[0]
        lines.append("")
        lines.append(f"## 同一外推位置对照示例：`{row['sample_id']}` / key_index `{row['key_index']}` / distance `{row['relative_distance']}`")
        lines.append("")
        lines.append("| scheme | position err | rotation err | dot err | norm err | fixed score | fixed prob |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for scheme_name, values in row["schemes"].items():
            lines.append(
                f"| `{scheme_name}` | {values['position']['abs_error']:.3e} | "
                f"{values['rotation']['abs_error']:.3e} | {values['dot']['abs_error']:.3e} | "
                f"{values['normalization']['abs_error']:.3e} | {values['fixed_score']:.6f} | "
                f"{values['fixed_probability']:.6f} |"
            )
    lines.append("")
    lines.append("## 预算与故障隔离")
    lines.append(f"- 首批待办: `{summary['budget_first_batch']['pending']}`; consumed=`{summary['budget_first_batch']['consumed']}`; codes=`{summary['budget_first_batch']['codes']}`")
    lines.append(f"- 排空批次: `{summary['budget_drain']['batch_count']}`, pending=`{summary['budget_drain']['pending']}`")
    lines.append(f"- 配额超限失败数: `{summary['quota']['failure_count']}`, codes=`{summary['quota']['codes']}`")
    lines.append("- 注入 stderr:")
    lines.append("```text")
    lines.append(summary["injection"]["stderr"])
    lines.append("```")
    for failure in summary["injection"]["failures"]:
        lines.append(f"- failure: `{failure.get('code')}` sample=`{failure.get('sample_id')}` scheme=`{failure.get('scheme_id')}` stage=`{failure.get('stage')}`")
    lines.append("")
    lines.append("## 快照恢复")
    snap = summary["snapshot"]
    lines.append(f"- 未提交临时目录: `{snap['prepared_uncommitted_dir']}`; recover 前存在=`{snap['prepared_existed']}`")
    lines.append(f"- 崩溃恢复: `{snap['after_recover_status']}`; 清理 orphan=`{snap['orphans_removed']}`")
    lines.append(f"- 删除 current pointer 后: `{snap['pointer_loss_status']}`; global_version=`{snap['global_version']}`")
    lines.append(
        f"- 真实 SIGKILL: returncode=`{snap['sigkill_returncode']}`, 临时目录 kill 后存在=`{snap['temp_survived_sigkill']}`, "
        f"恢复状态=`{snap['sigkill_recovery_status']}`, 清理=`{snap['sigkill_orphans_removed']}`, "
        f"recover 后目录消失=`{snap['temp_cleaned_after_recover']}`"
    )

    report_path = ROOT / "docs" / "verification-report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(summary_path.relative_to(ROOT)), "report": str(report_path.relative_to(ROOT))}, ensure_ascii=False))
    return 0 if pytest_result["returncode"] == 0 and eq.get("incremental_vs_full_serial") and eq.get("serial_vs_threaded") else 1


if __name__ == "__main__":
    raise SystemExit(main())
