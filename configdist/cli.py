"""命令行入口：样本生成、归一、下发、权威重算、恢复与验收演示。

Windows 终端用法（Linux/macOS 相同）：
  python -m configdist.cli gen-samples --workdir demo_ws
  python -m configdist.cli normalize   --workdir demo_ws
  python -m configdist.cli run         --workdir demo_ws --manifest app_v3.json
  python -m configdist.cli recompute   --workdir demo_ws
  python -m configdist.cli recover     --workdir demo_ws
  python -m configdist.cli demo        --workdir demo_ws   # 全量验收
  python -m configdist.gui             --workdir demo_ws   # Tkinter 界面
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time

from .errors import PlatformError
from .manifest import encode_manifest
from .platform import Platform


def _print(title, obj):
    print(f"\n=== {title} ===")
    print(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False,
                     default=str))


def cmd_gen_samples(args):
    platform = Platform(args.workdir)
    paths = platform.gen_samples()
    _print("样本已生成", paths)


def cmd_normalize(args):
    platform = Platform(args.workdir)
    _print("归一报告", platform.normalize())


def cmd_run(args):
    platform = Platform(args.workdir)
    platform.open()
    try:
        platform.start_nodes()
        manifest = platform.load_manifest(args.manifest)
        _print("下发报告", platform.dispatch(manifest))
    finally:
        platform.close()


def cmd_recompute(args):
    platform = Platform(args.workdir)
    platform.open()
    try:
        _print("权威重算", platform.recompute_authoritative())
    finally:
        platform.close()


def cmd_recover(args):
    platform = Platform(args.workdir)
    platform.open()
    try:
        _print("恢复报告", platform.recover())
    finally:
        platform.close()


def cmd_crash_run(args):
    """内部命令：提交一个批次后在第二个批次运行中强制退出（os._exit）。"""
    platform = Platform(args.workdir)
    platform.gen_samples()
    platform.normalize()
    platform.open()
    platform.start_nodes()
    manifest = platform.load_manifest("app_v3.json")
    platform.dispatch(manifest)  # 批次 1：完整提交
    _tasks, records, _kv = platform._build_tasks(manifest)
    platform.ledger.begin_batch(manifest.manifest_version,
                                "crash-run", records)  # 批次 2：running 未提交
    platform.nodes.stop_all()
    platform.ledger.conn.close()
    os._exit(9)  # 模拟强制终止：不提交批次 2，不生成退出报告


def _normkill_check(workdir):
    """归一中断续做：子进程升级中被强杀，重启后依据检查点续做。"""
    scratch = os.path.join(workdir, "normkill")
    mdir = os.path.join(scratch, "samples", "manifests")
    os.makedirs(mdir, exist_ok=True)
    entries = [{"key": f"k{i:03d}", "value": i * 7 - 100} for i in range(30)]
    with open(os.path.join(mdir, "big_v1.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"format": "v1", "entries": entries}, fh)
    env = dict(os.environ, CONFIGDIST_UPGRADE_DELAY_MS="120")
    proc = subprocess.Popen(
        [sys.executable, "-m", "configdist.cli", "normalize",
         "--workdir", scratch],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    time.sleep(0.9)  # 让子进程处理若干条后强杀
    proc.kill()
    proc.wait()
    ckpt = os.path.join(mdir, "big_v1.json.upgchk.json")
    with open(ckpt, "r", encoding="utf-8") as fh:
        done_before_kill = json.load(fh)["next_index"]
    platform = Platform(scratch)
    reports = platform.normalize()
    report = reports[0]
    ok = (0 < done_before_kill < 30
          and report["resumed_from"] == done_before_kill
          and report["records"] == 30
          and report["source_version"] == "v1"
          and report["target_version"] == "v3"
          and not os.path.exists(ckpt))
    backup = os.path.join(mdir, "big_v1.json.v1.bak")
    ok = ok and os.path.exists(backup) and not os.access(backup, os.W_OK)
    return {"killed_after_records": done_before_kill,
            "resume_report": report, "ok": ok}


def cmd_demo(args):
    workdir = args.workdir
    if os.path.exists(workdir):
        shutil.rmtree(workdir)
    summary = {}

    # 0. 归一中断续做（独立子进程强杀）
    summary["normalize_resume"] = _normkill_check(workdir)

    platform = Platform(workdir)
    summary["samples"] = platform.gen_samples()
    summary["normalize"] = platform.normalize()
    summary["recover_on_open"] = platform.open()
    try:
        platform.start_nodes()

        # 1. 优先级并发下发（后台线程），主线程模拟界面轮询保持可交互
        manifest = platform.load_manifest("app_v3.json")
        dispatch_box = {}

        def _run():
            dispatch_box["report"] = platform.dispatch(manifest)

        worker = threading.Thread(target=_run)
        started = time.monotonic()
        worker.start()
        polls = 0
        in_flight_seen = 0
        while worker.is_alive():
            view = platform.ui_query()
            polls += 1
            if view["progress"]:
                in_flight_seen = max(in_flight_seen,
                                     view["progress"]["in_flight_global"])
            time.sleep(0.01)
        worker.join()
        report = dispatch_box["report"]
        summary["dispatch_v5"] = report
        summary["interactivity"] = {
            "ui_polls_during_dispatch": polls,
            "max_in_flight_observed": in_flight_seen,
            "dispatch_wall_ms": int((time.monotonic() - started) * 1000),
            "responsive": polls > 0,
        }

        # 2. 权威重算与快速结论逐字节比较
        summary["recompute_v5"] = platform.recompute_authoritative()

        # 3. 节点强杀后按持久化状态恢复
        ack = platform.restart_node("node-a")
        summary["node_restart"] = {
            "node": "node-a",
            "recovered_state": ack["state"],
            "ok": ack["state"]["applied"] > 0,
        }

        # 4. 编辑演进：版本单调递增、陈旧防护与整体替换
        old_version = manifest.manifest_version
        new_obj = json.loads(encode_manifest(manifest))
        for entry in new_obj["entries"]:
            if entry["key"] == "alpha":
                entry["value"] = 2000
        new_obj["entries"].append(
            {"key": "eta", "value": 99, "targets": ["node-a"], "priority": 50})
        edit = platform.edit_manifest("app_v3.json", json.dumps(new_obj))
        summary["edit"] = edit
        stale_count = sum(1 for e in platform.preview.values() if e["stale"])
        summary["staleness"] = {"stale_entries_after_edit": stale_count,
                                "excluded_from_view": stale_count > 0}
        report2 = platform.dispatch(platform.current_manifest)
        summary["dispatch_v6"] = report2
        summary["recompute_v6"] = platform.recompute_authoritative()
        summary["replacement"] = platform.replacement_report(
            old_version, edit["manifest_version"])
        summary["latency"] = platform.latency_report()
    finally:
        platform.close()

    # 5. 强制终止后的账本恢复（独立子进程 os._exit）
    crash_dir = os.path.join(workdir, "crash_ws")
    proc = subprocess.run(
        [sys.executable, "-m", "configdist.cli", "crash-run",
         "--workdir", crash_dir],
        capture_output=True, text=True)
    crash_platform = Platform(crash_dir)
    recovery = crash_platform.open()
    crash_platform.start_nodes()
    manifest = crash_platform.load_manifest("app_v3.json")
    rerun = crash_platform.dispatch(manifest)
    recompute = crash_platform.recompute_authoritative()
    crash_platform.close()
    summary["kill_recovery"] = {
        "crash_exit_code": proc.returncode,
        "recovery": recovery,
        "rerun_batch": rerun["batch_id"],
        "rerun_ok": rerun["ok"] == rerun["slices"] and not rerun["deviations"],
        "recompute_match": recompute["match"],
    }

    checks = {
        "normalize_three_versions": all(
            r["status"] in ("upgraded", "already-current")
            for r in summary["normalize"]) and len(summary["normalize"]) == 3,
        "normalize_resume": summary["normalize_resume"]["ok"],
        "priority_dispatch": summary["dispatch_v5"]["start_order"][0][0]
            == "rho" and not summary["dispatch_v5"]["deviations"],
        "quota_respected":
            summary["dispatch_v5"]["quota"]["global_peak"] <= 3 and
            max(summary["dispatch_v5"]["quota"]["node_peak"].values()) <= 2,
        "converged": not summary["dispatch_v5"]["deviations"],
        "authoritative_match": summary["recompute_v5"]["match"]
            and summary["recompute_v6"]["match"],
        "interactive": summary["interactivity"]["responsive"],
        "node_restart": summary["node_restart"]["ok"],
        "staleness": summary["staleness"]["excluded_from_view"],
        "kill_recovery": summary["kill_recovery"]["recovery"]
            ["unknown_batches"] != []
            and summary["kill_recovery"]["recompute_match"],
    }
    summary["acceptance"] = checks
    summary["pass"] = all(checks.values())
    _print("验收演示汇总", summary)
    print("\nACCEPTANCE:", "PASS" if summary["pass"] else "FAIL")
    return 0 if summary["pass"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(prog="configdist.cli")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add(name, fn):
        p = sub.add_parser(name)
        p.add_argument("--workdir", default="demo_ws")
        if name == "run":
            p.add_argument("--manifest", default="app_v3.json")
        p.set_defaults(fn=fn)

    add("gen-samples", cmd_gen_samples)
    add("normalize", cmd_normalize)
    add("run", cmd_run)
    add("recompute", cmd_recompute)
    add("recover", cmd_recover)
    add("crash-run", cmd_crash_run)
    add("demo", cmd_demo)
    args = parser.parse_args(argv)
    try:
        return args.fn(args) or 0
    except PlatformError as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
