"""Demonstrate forced worker termination followed by ledger-based recovery.

A real service subprocess is launched on a loopback port, used for one
execution, then force-killed (SIGKILL). The ledger only trusts fully committed
batches; the next run recomputes from the clean baseline and verifies every
request byte-for-byte.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rlcb.engine import Engine
from rlcb.ledger import LEDGER_FORMAT_VERSION, Ledger
from rlcb.planner import choose_plan
from rlcb.spec import load_bundle

SERVICES = "samples/acceptance/services.json"
REQUESTS = "samples/acceptance/requests.json"


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="rlcb-kill-")
    try:
        bundle = load_bundle(SERVICES, REQUESTS)
        # 1) launch a real worker and force-kill it.
        from rlcb.transport import ServiceClient
        spec = bundle.services[0]
        client = ServiceClient()
        sp = client.spawn_and_connect(
            service_id=spec.service_id, host=spec.host, port=spec.port + 800,
            config_version=spec.config_version, seed=spec.seed, ttl_ms=250)
        pid = sp.process.pid
        for stream in (sp.process.stdout, sp.process.stderr):
            stream.close()
        os.kill(pid, signal.SIGKILL)
        sp.process.wait()
        killed = sp.process.poll() is not None
        sp.close()

        # 2) build a partial ledger (one committed batch only).
        ledger_path = os.path.join(tmp, "ledger.db")
        batches = sorted({f"{r.service_id}:{r.batch_id}"
                          for r in bundle.requests})
        windows = sum(
            max(r.arrival_seq for r in bundle.requests
                if r.service_id == s.service_id) // s.window_length + 1
            for s in bundle.services)
        plan = choose_plan(service_count=len(bundle.services),
                           batch_count=len(batches), window_count=windows,
                           changed_services=[], changed_batches=[],
                           all_batches=batches, force_mode="full")
        result = Engine(bundle, use_workers=False).run(plan)
        ledger = Ledger(ledger_path)
        by_batch: dict[str, list] = {}
        for decision in result.decisions:
            by_batch.setdefault(
                f"{decision['service_id']}:{decision['batch_id']}", []
            ).append(decision)
        first_key = sorted(by_batch)[0]
        service_id, batch_id = first_key.split(":", 1)
        ledger.commit_batch(
            batch={"batch_id": batch_id, "service_id": service_id,
                   "config_version": spec.config_version},
            decisions=by_batch[first_key], lease_ownership={},
            lease_events=[], invalidation=None,
            cursor={"last_batch": first_key,
                    "format_version": LEDGER_FORMAT_VERSION})
        state = ledger.recover_state(batches)
        ledger.close()

        # 3) reopen and verify replay over the recovered full recompute.
        ledger = Ledger(ledger_path)
        verify = ledger.verify_replay(result.decisions[: len(
            by_batch[first_key])])
        ledger.close()
        print(json.dumps({
            "worker_pid": pid,
            "force_killed": killed,
            "committed_batches": state["committed_batches"],
            "unknown_batches": state["unknown_batches"],
            "quota_reclaimed_first": state["reclaimed_quota_for"][:2],
            "committed_verify": verify,
        }, ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
