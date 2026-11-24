"""Command line entry point for the repeatable local demo."""

import argparse
import copy
import json
import os
import sys

from .diagnostics import render_headless
from .engine import Engine
from .fault_inject import inject_corrupt_manifest, inject_cycle_manifest, inject_snapshot_mismatch
from .ledger import LEDGER_FORMAT_VERSION, Ledger, canonical
from .logger import StructuredLogger
from .protocol import send_batch, start_all, stop_all
from .clock import now_ms
from .spec import load_events, load_manifest


def _paths(workdir):
    return {
        "ledger": os.path.join(workdir, "ledger.sqlite"),
        "log": os.path.join(workdir, "events.log.jsonl"),
        "report": os.path.join(workdir, "report.json"),
    }


def _write_report(path, report):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, ensure_ascii=False)


def run(manifest_path, events_path, workdir, crash_after=None, ui=False, crash_dispatch=None):
    paths = _paths(workdir)
    os.makedirs(workdir, exist_ok=True)
    logger = StructuredLogger(paths["log"])
    try:
        manifest = load_manifest(manifest_path)
        events = load_events(events_path, manifest)
        ledger = Ledger(paths["ledger"])
        engine = Engine(manifest, ledger, logger)
        engine.crash_on_dispatch = crash_dispatch
        processes = start_all(manifest_path, manifest.nodes)
        report = {"mode": "run", "format_version": LEDGER_FORMAT_VERSION}
        try:
            engine.start()
            for index, event in enumerate(events):
                ledger.append_event(event.__dict__)
                if event.kind == "submit":
                    engine.submit(event.shard_id, event.seq)
                elif event.kind == "cancel":
                    engine.cancel(event.shard_id, event.seq)
                elif event.kind == "quota":
                    engine.change_quota(event.node_id, event.concurrency, event.lease_limit, event.seq)
                elif event.kind == "revise":
                    engine.revise(event.revision, event.seq)
                engine._pump_until_blocked()
                ledger.write_snapshot(event.seq)
                if crash_after is not None and index == crash_after:
                    logger.event("crash_injection", event_seq=event.seq)
                    report["crash_injected_at_event"] = event.seq
                    _write_report(paths["report"], report)
                    logger.close()
                    os._exit(9)
            engine.wait_idle()
            report["outcomes"] = _summarize_outcomes(ledger)
            report["invalidations"] = engine.invalidations
            report["unknown"] = ledger.unknown_shards()
            report["snapshots"] = ledger.snapshots()
            report["head_digest"] = ledger.verify_digest_chain()
            _write_report(paths["report"], report)
            if ui:
                from .diagnostics import open_dashboard

                open_dashboard(engine, ledger)
            else:
                print(render_headless(engine, ledger))
        finally:
            engine.shutdown()
            stop_all(processes)
            ledger.close()
            logger.close()
        return report
    except Exception as exc:
        if hasattr(exc, "to_dict"):
            logger.error(exc)
        raise
    finally:
        logger.close()


def recover(manifest_path, events_path, workdir):
    paths = _paths(workdir)
    logger = StructuredLogger(paths["log"].replace(".jsonl", ".recover.jsonl"))
    try:
        manifest = load_manifest(manifest_path)
        ledger = Ledger(paths["ledger"])
        engine = Engine(manifest, ledger, logger)
        engine.recover()
        unknown = list(ledger.unknown_shards())
        report = {
            "mode": "recover",
            "format_version": LEDGER_FORMAT_VERSION,
            "head_digest": ledger.verify_digest_chain(),
            "unknown": unknown,
            "outcomes": _summarize_outcomes(ledger),
        }
        if unknown:
            events = load_events(events_path, manifest)
            processes = start_all(manifest_path, manifest.nodes)
            try:
                engine.start()
                for sid, generation, reason in sorted(unknown):
                    engine.submitted.add(sid)
                    engine.unknown.discard(sid)
                    engine.state[sid] = "queued"
                    engine.attempts[sid] = 0
                    engine.queue_start[sid] = now_ms()
                engine._pump_until_blocked()
                engine.wait_idle()
            finally:
                engine.shutdown()
                stop_all(processes)
            report["rerun_unknown"] = [row[0] for row in unknown]
            report["outcomes_after_rerun"] = _summarize_outcomes(ledger)
        _write_report(paths["report"].replace(".json", ".recover.json"), report)
        print(render_headless(engine, ledger))
        ledger.close()
        return report
    finally:
        logger.close()


def verify(manifest_path, events_path, workdir):
    paths = _paths(workdir)
    logger = StructuredLogger(paths["log"].replace(".jsonl", ".verify.jsonl"))
    try:
        manifest = load_manifest(manifest_path)
        events = load_events(events_path, manifest)
        ledger = Ledger(paths["ledger"])
        engine = Engine(manifest, ledger, logger)
        result = engine.replay_verify(events)
        report = {
            "mode": "verify",
            "format_version": LEDGER_FORMAT_VERSION,
            "head_digest": ledger.verify_digest_chain(),
            **result,
        }
        _write_report(paths["report"].replace(".json", ".verify.json"), report)
        print(json.dumps(report, indent=2, sort_keys=True))
        ledger.close()
        return report
    except Exception as exc:
        if hasattr(exc, "to_dict"):
            logger.error(exc)
        raise
    finally:
        logger.close()


def inject(kind, manifest_path, events_path, workdir):
    paths = _paths(workdir)
    os.makedirs(workdir, exist_ok=True)
    logger = StructuredLogger(paths["log"].replace(".jsonl", f".{kind}.jsonl"))
    try:
        if kind == "corrupt-spec":
            target = inject_corrupt_manifest(manifest_path, workdir)
            load_manifest(target)
        elif kind == "cycle":
            target = inject_cycle_manifest(manifest_path, workdir)
            load_manifest(target)
        elif kind == "snapshot":
            inject_snapshot_mismatch(paths["ledger"])
            Ledger(paths["ledger"])
        raise AssertionError(f"fault injection {kind} did not raise")
    except Exception as exc:
        if hasattr(exc, "to_dict"):
            logger.error(exc)
            print(json.dumps(exc.to_dict(), indent=2, sort_keys=True, ensure_ascii=False))
            return exc.code
        raise
    finally:
        logger.close()


def _summarize_outcomes(ledger: Ledger):
    rows = []
    for sid, generation, status, version, summary, units, digest in ledger.all_outcomes():
        rows.append(
            {
                "shard_id": sid,
                "generation": generation,
                "status": status,
                "config_version": version,
                "units": units,
                "summary_digest": digest,
                "summary": json.loads(summary) if summary else None,
            }
        )
    return rows


def build_parser():
    parser = argparse.ArgumentParser(description="Cross-node shard quota ledger demo")
    parser.add_argument("mode", choices=["run", "recover", "verify", "inject"])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--crash-after", type=int, default=None)
    parser.add_argument("--crash-dispatch", default=None)
    parser.add_argument("--inject", choices=["corrupt-spec", "cycle", "snapshot"], default=None)
    parser.add_argument("--ui", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.mode == "run":
        run(args.manifest, args.events, args.workdir, crash_after=args.crash_after, ui=args.ui, crash_dispatch=args.crash_dispatch)
    elif args.mode == "recover":
        recover(args.manifest, args.events, args.workdir)
    elif args.mode == "verify":
        verify(args.manifest, args.events, args.workdir)
    elif args.mode == "inject":
        code = inject(args.inject, args.manifest, args.events, args.workdir)
        return 0 if code else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
