"""Repeatable command-line demonstration for the delay queue component.

Subcommands
-----------
run                 Normal run: retries, dead-letter, dedup, replay verify
crash               Hard-kill the process mid-delivery (os._exit, no cleanup)
recover             Reopen the crashed ledger, recover, finish and replay
ui                  Launch the Tkinter diagnostic dashboard
inject <scenario>   E_SPEC / E_SNAPSHOT / E_TIMEOUT fault injection

Everything is deterministic and uses only the local filesystem.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

from delayqueue import (
    DelayQueueError,
    ReplayError,
    SnapshotMismatchError,
    SpecError,
    TimeoutConfigError,
    launch_dashboard,
    load_declaration,
    open_queue,
    replay_ledger,
)
from delayqueue.policy import should_fail
from delayqueue.replay import check_reproducibility

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "samples"
RUNTIME = ROOT / "runtime"
SPEC_PATH = SAMPLES / "topics.json"
MESSAGES_PATH = SAMPLES / "messages.ndjson"
LEDGER_PATH = RUNTIME / "queue.db"


def _fresh_runtime() -> None:
    if RUNTIME.exists():
        for path in RUNTIME.iterdir():
            if path.is_file():
                path.chmod(0o644)
        shutil.rmtree(RUNTIME)
    RUNTIME.mkdir(parents=True, exist_ok=True)


def _make_handler(queue, slow_message: str | None = None,
                  slow_ms: int = 0):
    def handler(delivery, q):
        if slow_message == delivery.message_id:
            time.sleep(slow_ms / 1000.0)
        if should_fail(delivery.attempt, delivery.payload):
            queue.nack(delivery, f"simulated-failure-{delivery.attempt + 1}")
        else:
            result = queue.ack(delivery)
            if result["suppressed"]:
                print(f"  suppress idem={delivery.idempotency_key} "
                      f"count={result['suppressed_count']}")
    return handler


def _print_final_state(queue) -> None:
    snapshot = queue.snapshot()
    print("\nstate counts      :", snapshot["state_counts"])
    print("attempt dist      :", snapshot["attempt_distribution"])
    print("global inflight   :",
          f"{snapshot['global_inflight']}/{snapshot['global_concurrency']}")
    for topic, view in snapshot["topics"].items():
        print(f"  {topic:<14} ready={view['ready']} delayed={view['delayed']}"
              f" inflight={view['inflight']} quota={view['quota']}")
    for record in snapshot["dead"]:
        print(f"  DEAD {record['message_id']} attempts={record['attempts']}"
              f" first={record['first_error']!r} last={record['last_error']!r}")
    for message in queue.all_messages():
        print(f"  {message.message_id} state={message.state:<8}"
              f" attempts={message.attempts} suppressed={message.suppressed}"
              f" migrations={queue.ledger.migration_count(message.message_id)}")


def cmd_run() -> int:
    _fresh_runtime()
    print("== normal run ==")
    queue, inputs = open_queue(
        str(SPEC_PATH), str(MESSAGES_PATH), str(LEDGER_PATH))
    print(f"loaded {len(inputs)} messages, ledger={LEDGER_PATH}")
    try:
        queue.run_workers(_make_handler(queue), worker_count=4)
        drained = queue.wait_drained(timeout=15)
        print("drained           :", drained)
        _print_final_state(queue)
        result = replay_ledger(queue)
        print("\nreplay verify     :", result)
    finally:
        queue.close()
    return 0


def cmd_crash() -> int:
    _fresh_runtime()
    print("== crash run: hard kill while deliveries are in flight ==")
    queue, _ = open_queue(
        str(SPEC_PATH), str(MESSAGES_PATH), str(LEDGER_PATH))
    deliveries = []
    for _ in range(3):
        delivery = queue.claim(timeout=2)
        if delivery is not None:
            deliveries.append(delivery)
            print(f"  claimed {delivery.message_id} attempt={delivery.attempt}"
                  f" lease_mono={delivery.lease_monotonic_ms}")
    print(f"{len(deliveries)} leases committed; forcing os._exit(42)...")
    sys.stdout.flush()
    os._exit(42)  # hard termination: no finally, no graceful close
    return 0  # unreachable


def cmd_recover() -> int:
    print("== recovery run ==")
    queue, report = DelayQueue_reopen()
    report_dict = report.as_dict()
    print("recovery report   :", json.dumps(report_dict, ensure_ascii=False))
    (RUNTIME / "recovery_report.json").write_text(
        json.dumps(report_dict, indent=2, ensure_ascii=False),
        encoding="utf-8")
    try:
        queue.run_workers(_make_handler(queue), worker_count=4)
        queue.wait_drained(timeout=15)
        _print_final_state(queue)
        result = replay_ledger(queue)
        print("\nreplay verify     :", result)
    finally:
        queue.close()
    return 0


def DelayQueue_reopen():
    from delayqueue import recover_queue
    return recover_queue(str(LEDGER_PATH))


def cmd_ui() -> int:
    print("== launching Tkinter dashboard (needs a desktop session) ==")
    if not LEDGER_PATH.exists():
        queue, _ = open_queue(
            str(SPEC_PATH), str(MESSAGES_PATH), str(LEDGER_PATH))
        queue.run_workers(_make_handler(queue), worker_count=4)
        queue.wait_drained(timeout=15)
    else:
        queue, _ = DelayQueue_reopen()
    launch_dashboard(queue)
    return 0


def cmd_inject(scenario: str) -> int:
    print(f"== fault injection: {scenario} ==")
    if scenario == "spec":
        bad_path = RUNTIME / "bad_messages.ndjson"
        _fresh_runtime()
        bad_path.write_text(
            '{"message_id": "x1", "topic": "ghost", "deliver_at": 0, '
            '"payload": 1, "idempotency_key": "k1"}\n',
            encoding="utf-8")
        try:
            spec = load_declaration(str(SPEC_PATH))
            from delayqueue import load_stream
            load_stream(str(bad_path), spec)
        except SpecError as exc:
            print(f"caught {exc.code}: {exc.message} "
                  f"line={exc.line} field={exc.field}")
            return 0
    if scenario == "snapshot":
        _fresh_runtime()
        queue, _ = open_queue(
            str(SPEC_PATH), str(MESSAGES_PATH), str(LEDGER_PATH))
        queue.run_workers(_make_handler(queue), worker_count=4)
        queue.wait_drained(timeout=15)
        queue.ledger.tamper_digest()
        queue.close()
        try:
            from delayqueue import recover_queue
            recover_queue(str(LEDGER_PATH))
        except SnapshotMismatchError as exc:
            report = exc.report
            print(f"caught {exc.code}: {exc.message}")
            print("replacement report:")
            print(json.dumps(report, indent=2, ensure_ascii=False))
            replacement = _replace_tampered_ledger()
            (RUNTIME / "replacement_report.json").write_text(
                json.dumps(replacement, indent=2, ensure_ascii=False),
                encoding="utf-8")
            print("full replacement :",
                  json.dumps(replacement, ensure_ascii=False))
            return 0
    if scenario == "timeout":
        _fresh_runtime()
        try:
            from delayqueue import DelayQueue
            DelayQueue.validate_timeout(0)
        except TimeoutConfigError as exc:
            print(f"caught {exc.code}: {exc.message} details={exc.details}")
            return 0
    print("scenario not handled")
    return 2


def _replace_tampered_ledger() -> dict:
    from delayqueue import Ledger, load_stream
    spec = load_declaration(str(SPEC_PATH))
    inputs = load_stream(str(MESSAGES_PATH), spec)
    backup = RUNTIME / "queue.backup.db"
    return Ledger.replace(
        str(LEDGER_PATH), str(backup), spec, inputs,
        epoch_ms=int(time.time() * 1000))


def main(argv: list[str]) -> int:
    command = argv[1] if len(argv) > 1 else "run"
    try:
        if command == "run":
            return cmd_run()
        if command == "crash":
            return cmd_crash()
        if command == "recover":
            return cmd_recover()
        if command == "ui":
            return cmd_ui()
        if command == "inject":
            if len(argv) < 3:
                print("usage: demo.py inject {spec|snapshot|timeout}")
                return 2
            return cmd_inject(argv[2])
        print(f"unknown command: {command}")
        return 2
    except DelayQueueError as exc:
        print(f"uncaught {exc.code}: {exc}")
        return 1
    except ReplayError as exc:  # pragma: no cover
        print(f"uncaught {exc.code}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
