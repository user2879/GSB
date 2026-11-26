"""可重复演示脚本：端到端跑通验收路径并打印实际结果。

用法: python3 -m src.demo

步骤：生成样本 -> 首次运行（真实子进程 + 回环套接字）-> 修改工作进程种子做
失效子集重算 -> 增量/全量比对 -> 重放校验 -> kill -9 恢复 -> 各业务码异常演示。
"""
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import time

from . import receipts as receipts_mod, samplegen
from .core import load_config
from .errors import BizError
from .ledger import Ledger
from .pipeline import run_pipeline
from .replay import replay
from .verify import compare

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(ROOT, "config", "pipeline.json")
PORTS = os.path.join(ROOT, "config", "ports.json")
STATE = os.path.join(ROOT, "state")
LEDGER = os.path.join(STATE, "ledger.db")
SAMPLES = os.path.join(ROOT, "samples")
RECEIPT_FILES = [os.path.join(SAMPLES, n)
                 for n in ("receipts_r1.txt", "receipts_r2.txt", "receipts_r3.txt")]


def step(title):
    print(f"\n=== {title} ===")


def show(label, value):
    print(f"  {label}: {value}")


def main():
    os.chdir(ROOT)
    shutil.rmtree(STATE, ignore_errors=True)
    os.makedirs(STATE, exist_ok=True)

    step("1. 生成确定性样本（三格式历史回执 + 异常样本）")
    files = samplegen.generate(CONFIG, SAMPLES)
    show("samples", files)

    step("2. 首次运行：真实子进程 + 回环套接字派发，三格式回执归一采纳")
    summary = run_pipeline(CONFIG, PORTS, LEDGER, receipt_files=RECEIPT_FILES, snapshot=True)
    show("param_versions", summary["param_versions"])
    show("committed", summary["committed"])
    show("snapshot", os.path.basename(summary["snapshot"]))
    ledger = Ledger(LEDGER)
    adopted = ledger.query("SELECT src_format, COUNT(*) c FROM receipts"
                           " WHERE adopted=1 GROUP BY src_format ORDER BY src_format")
    rejected = ledger.query("SELECT COUNT(*) c FROM receipts WHERE adopted=0")
    show("adopted_by_format", adopted)
    show("rejected_total", rejected[0]["c"])
    digest_before = {r["batch_id"]: r["adopted_digest"]
                     for r in ledger.query("SELECT batch_id, adopted_digest FROM batches")}
    ledger.close()

    step("3. 修改 w1 工作进程种子 -> 仅失效子集重算")
    cfg = load_config(CONFIG)
    alt = dict(cfg)
    alt.pop("workers_by_id", None)
    alt["workers"] = [dict(w, seed=w["seed"] + 1) if w["worker_id"] == "w1" else w
                      for w in cfg["workers"]]
    alt_path = os.path.join(STATE, "pipeline.alt.json")
    with open(alt_path, "w", encoding="utf-8") as fh:
        json.dump(alt, fh, ensure_ascii=False, indent=2)
    summary2 = run_pipeline(alt_path, PORTS, LEDGER, receipt_files=RECEIPT_FILES)
    show("invalidated_batches", summary2["invalidated_batches"])
    show("invalidated_requests", summary2["invalidated_requests"])
    show("recomputed", summary2["committed"])
    show("unaffected", summary2["unaffected_batches"])
    ledger = Ledger(LEDGER)
    digest_after = {r["batch_id"]: r["adopted_digest"]
                    for r in ledger.query("SELECT batch_id, adopted_digest FROM batches")}
    ledger.close()
    untouched_ok = all(digest_before[b] == digest_after[b]
                       for b in summary2["unaffected_batches"])
    changed = [b for b in summary2["invalidated_batches"]
               if digest_before[b] != digest_after[b]]
    show("untouched_byte_identical", untouched_ok)
    show("invalidated_digests_changed", changed)

    step("4. 校验：增量重算 vs 从零全量重建")
    result = compare(LEDGER, alt_path, PORTS, RECEIPT_FILES)
    show("result", result["result"])
    show("compared_batches", len(result["compared_batches"]))

    step("5. 恢复后一致性重放校验")
    show("replay", replay(LEDGER))

    step("6. kill -9 强制终止 -> 仅已提交记录可恢复，未提交批标记未知并重跑")
    kill_ledger = os.path.join(STATE, "kill.db")
    proc = subprocess.Popen(
        [sys.executable, "-m", "src.pipeline", "run",
         "--config", CONFIG, "--ports", PORTS, "--ledger", kill_ledger,
         "--compute-delay-ms", "1200"],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + 30
    done_before_kill = 0
    while time.monotonic() < deadline:
        if os.path.exists(kill_ledger) and os.path.getsize(kill_ledger) > 0:
            try:
                conn = sqlite3.connect(kill_ledger)
                done_before_kill = conn.execute(
                    "SELECT COUNT(*) FROM batches WHERE status='done'").fetchone()[0]
                conn.close()
            except sqlite3.Error:
                done_before_kill = 0
            if done_before_kill >= 1:
                break
        time.sleep(0.1)
    proc.send_signal(signal.SIGKILL)
    proc.wait()
    show("committed_before_kill", done_before_kill)
    _wait_ports_released(json.load(open(PORTS)))
    ledger = Ledger(kill_ledger)
    unknown = [r["batch_id"] for r in
               ledger.query("SELECT batch_id FROM batches WHERE status='unknown'"
                            " ORDER BY batch_id")]
    show("unknown_after_recovery", unknown)
    ledger.close()
    summary3 = run_pipeline(CONFIG, PORTS, kill_ledger)
    show("rerun_committed", summary3["committed"])
    show("replay_after_recovery", replay(kill_ledger))

    step("7. 业务码异常演示")
    for name in ("bad_field.txt", "bad_ref.txt", "bad_ref_dangling.txt"):
        try:
            receipts_mod.parse_file(os.path.join(SAMPLES, name))
        except BizError as exc:
            show(name, str(exc))
    from .scheduler import Scheduler, BatchTask
    ports = json.load(open(PORTS))
    try:
        cfg_to = load_config(CONFIG)
        cfg_to["__config_path__"] = CONFIG
        cfg_to["scheduler"] = dict(cfg_to["scheduler"], timeout_ms=200)
        with Scheduler(cfg_to, ports, compute_delay_ms=1500) as sched:
            res = sched.run([BatchTask(batch_id="batch-0000", worker_id="w0",
                                       param_version=1,
                                       request_ids=("req-0000",))])[0]
            show("E_TIMEOUT", str(res.error))
    except BizError as exc:
        show("E_TIMEOUT", str(exc))
    import socket as _socket
    hold = _socket.socket()
    hold.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
    hold.bind(("127.0.0.1", ports["workers"]["w0"]))
    hold.listen(1)
    try:
        cfg_ep = load_config(CONFIG)
        cfg_ep["__config_path__"] = CONFIG
        Scheduler(cfg_ep, ports)
    except BizError as exc:
        show("E_PORT", str(exc))
    finally:
        hold.close()

    step("8. 账本与快照")
    ledger = Ledger(LEDGER)
    show("format_version", ledger.get_meta("format_version"))
    show("content_digest", ledger.get_meta("content_digest")[:16] + "...")
    show("progress_cursor", ledger.progress_cursor())
    show("snapshots", Ledger.list_snapshots(os.path.join(STATE, "snapshots")))
    ledger.close()
    print("\n演示完成。")


def _wait_ports_released(ports, timeout_s=10):
    """kill -9 后等待孤儿工作进程经 stdin EOF 看门狗自行退出并释放端口。"""
    import socket
    deadline = time.monotonic() + timeout_s
    for port in ports["workers"].values():
        while time.monotonic() < deadline:
            probe = socket.socket()
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind((ports["host"], port))
                probe.close()
                break
            except OSError:
                probe.close()
                time.sleep(0.1)


if __name__ == "__main__":
    main()
