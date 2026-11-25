"""强制终止模拟：在部分批已提交后杀死进程，留下未提交的在途批。"""

import os
import signal

from .clock import MonotonicClock
from .ledger import Ledger
from .leases import LeaseManager
from .plan import build_plan, plan_fingerprint, stable_hash
from .processes import WorkerPool
from .scheduler import Scheduler
from .service import make_receipt_records


def run_until_kill(spec, spec_dict, db_path, commit_before_kill=2):
    """提交前 N 批后 SIGKILL 自身；由上层重新启动恢复流程。"""
    clock = MonotonicClock()
    plans = build_plan(spec)
    delivery_index = {d.request_id: d for d in spec.deliveries}
    ledger = Ledger(db_path)
    snapshot_version = ledger.begin_snapshot(
        stable_hash(spec_dict), clock.now_ms(),
        plan_fingerprint(plans), "crash_demo")
    lease_manager = LeaseManager(spec, clock)
    lease_manager.start_reaper()
    committed = 0
    with WorkerPool(spec, spec_dict) as pool:
        scheduler = Scheduler(spec, pool, lease_manager, clock)
        for plan in plans:
            result = scheduler._dispatch(plan, delivery_index)
            records, _ = make_receipt_records(spec, plan, result)
            ledger.commit_batch(snapshot_version, result,
                                lease_manager.snapshot_leases(), records)
            committed += 1
            print(f"[crash-sim] 已提交 {committed} 批: {plan.batch_id}", flush=True)
            if committed >= commit_before_kill:
                print("[crash-sim] 模拟强制终止：发送 SIGKILL，后续批无提交记录",
                      flush=True)
                os.kill(os.getpid(), signal.SIGKILL)
    raise RuntimeError("不应到达此处")
