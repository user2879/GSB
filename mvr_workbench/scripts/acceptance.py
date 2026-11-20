"""一键验收：全量重建、增量=全量、快照恢复、注入处理、并发/串行一致性。

用法：python scripts/acceptance.py
输出：acceptance_runs/ 下的 JSON 诊断报告与逐步骤结论。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from mvrwb.editor import apply_edit_stream, load_config  # noqa: E402
from mvrwb.pipeline import ReconstructionPipeline  # noqa: E402
from mvrwb.sampler import generate_sample  # noqa: E402
from mvrwb.util import read_json, write_json_atomic  # noqa: E402

from tests.conftest import reconstruction_artifacts  # noqa: E402

OUT = os.path.join(ROOT, "acceptance_runs")


def fresh_dirs(base: str, name: str) -> dict:
    path = os.path.join(base, name)
    return {
        "work": os.path.join(path, "work"),
        "snapshot": os.path.join(path, "snapshot"),
        "report": os.path.join(path, "report.json"),
    }


def run(input_dir, dirs, *, injections=None, env=None, edits=None):
    apply_edit_stream(input_dir, dirs["work"], edits)
    pipeline = ReconstructionPipeline(
        dirs["work"], dirs["snapshot"],
        injection_config=injections, crash_env=env or {},
    )
    pipeline.stop_flag = {"stop": False}
    outcome = pipeline.run()
    write_json_atomic(dirs["report"], outcome.report)
    return outcome.report


def main() -> int:
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(OUT, exist_ok=True)
    sample = os.path.join(OUT, "sample")
    generate_sample(sample, view_count=6)
    results = []

    # 1) 从空白全量重建（串行 workers=1 与并发 workers=8 各一次）
    for workers in (1, 8):
        dirs = fresh_dirs(OUT, f"full_w{workers}")
        apply_edit_stream(sample, dirs["work"], None)
        cfg_path = os.path.join(dirs["work"], "reconstruction_config.json")
        cfg = read_json(cfg_path); cfg["thread_workers"] = workers
        write_json_atomic(cfg_path, cfg)
        report = run(sample, dirs)
        results.append((f"full-rebuild workers={workers}", report))

    serial_artifacts = reconstruction_artifacts(results[0][1])
    parallel_artifacts = reconstruction_artifacts(results[1][1])
    concurrency_ok = serial_artifacts == parallel_artifacts

    # 2) 增量编辑序列：同快照反复运行
    edit_stream = os.path.join(sample, "edit_stream.json")
    stream = read_json(edit_stream)["commands"]
    inc_snapshot = os.path.join(OUT, "incremental", "snapshot")
    incremental_report = None
    for index in range(1, len(stream) + 1):
        part = os.path.join(OUT, f"stream_{index}.json")
        write_json_atomic(part, {"commands": stream[:index]})
        dirs = {
            "work": os.path.join(OUT, "incremental", "work"),
            "snapshot": inc_snapshot,
            "report": os.path.join(OUT, "incremental", f"report_{index}.json"),
        }
        incremental_report = run(sample, dirs, edits=part)

    # 3) 对最终状态从空白全量重建，对照增量结果
    final_state = os.path.join(OUT, "final_state")
    shutil.copytree(sample, final_state)
    shutil.copyfile(os.path.join(sample, "alt_view_0002.png"),
                    os.path.join(final_state, "views", "view_0002.png"))
    shutil.copyfile(os.path.join(sample, "alt_view_0003.cam.json"),
                    os.path.join(final_state, "views", "view_0003.cam.json"))
    final_cfg = read_json(os.path.join(final_state, "reconstruction_config.json"))
    final_cfg.update({"feature_threshold": 170.0, "voxel_size": 0.28})
    write_json_atomic(os.path.join(final_state, "reconstruction_config.json"), final_cfg)
    full_final = run(final_state, fresh_dirs(OUT, "full_final"))
    incremental_ok = (
        reconstruction_artifacts(incremental_report)
        == reconstruction_artifacts(full_final)
    )

    # 4) 注入故障处理
    injection_report = run(
        sample, fresh_dirs(OUT, "injection"),
        injections=os.path.join(sample, "injection_config.json"),
    )
    injection_codes = sorted({hit["code"] for hit in injection_report["injections"]})
    injection_ids = sorted(hit["injection_id"] for hit in injection_report["injections"])

    # 5) 强制终止后恢复（子进程模拟 SIGKILL 式 os._exit）
    kill_snapshot = os.path.join(OUT, "kill", "snapshot")
    kill_work = os.path.join(OUT, "kill", "work")
    kill_report = os.path.join(OUT, "kill", "report.json")
    base_cmd = [
        sys.executable, "-m", "mvrwb.cli", "run",
        "--input", sample, "--work", kill_work,
        "--snapshot", kill_snapshot, "--report", kill_report,
    ]
    env = dict(os.environ)
    env.update({"MVR_CRASH_AFTER_STAGE": "depth", "MVR_CRASH_AFTER_VIEW": "2"})
    killed = subprocess.run(base_cmd, env=env, cwd=ROOT, capture_output=True, text=True)
    no_report_on_kill = not os.path.exists(kill_report)
    resume_cmd = list(base_cmd)
    resume_report = os.path.join(OUT, "kill", "resume_report.json")
    resume_cmd[-1] = resume_report
    resumed_proc = subprocess.run(resume_cmd, env=dict(os.environ), cwd=ROOT,
                                  capture_output=True, text=True)
    resumed = read_json(resume_report)
    clean_reference = run(sample, fresh_dirs(OUT, "kill_reference"))
    recovery_ok = (
        killed.returncode != 0
        and no_report_on_kill
        and resumed_proc.returncode == 0
        and reconstruction_artifacts(resumed)
        == reconstruction_artifacts(clean_reference)
    )

    summary = {
        "concurrency_serial_equals_parallel": concurrency_ok,
        "incremental_equals_clean_full_rebuild": incremental_ok,
        "injection_hit_ids": injection_ids,
        "injection_codes": injection_codes,
        "forced_kill_exit_code": killed.returncode,
        "no_report_on_forced_kill": no_report_on_kill,
        "recovery_matches_clean_rebuild": recovery_ok,
        "baseline_cloud_point_count": results[0][1]["cloud"]["point_count"],
        "final_cloud_point_count": full_final["cloud"]["point_count"],
        "baseline_budget_consumed": results[0][1]["budget_ledger"]["consumed"],
        "failures_under_injection": injection_report["failures"],
    }
    write_json_atomic(os.path.join(OUT, "acceptance_summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    all_ok = all([
        concurrency_ok, incremental_ok, recovery_ok,
        len(injection_ids) == 4, resumed_proc.returncode == 0,
    ])
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
