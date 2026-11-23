#!/usr/bin/env python3
"""CCObs 命令行入口: 本地回环确定性 TCP 拥塞控制观测组件。

用法:
  python run_session.py --config C.json --script S.json --db D.db run
  python run_session.py --config C.json --script S.json --db D.db recover
  python run_session.py --config C.json --script S.json --db D.db replay
  python run_session.py --config C.json --script S.json --db D.db gui

退出码: 0 正常; 2 E_CONFIG; 3 E_SNAPSHOT; 4 E_REPLAY。
"""
import argparse
import json
import sys

from ccobs.config import load_peer_script, load_session_config
from ccobs.errors import E_CONFIG, E_SNAPSHOT, E_REPLAY, BizError
from ccobs.ledger import canonical
from ccobs.session import build_ledger, run_session
from ccobs.snapshot import RecoveryResult, SnapshotStore, diff_windows
from ccobs.replay import replay_verify

EXIT_CODES = {E_CONFIG: 2, E_SNAPSHOT: 3, E_REPLAY: 4}


def _load(args):
    config = load_session_config(args.config)
    script = load_peer_script(args.script, config)
    return config, script


def _print_ledger(ledger):
    for batch in ledger.batches:
        print("BATCH|%d|%d..%d|clock_ms=%d|v%d|%s" % (
            batch.index, batch.event_lo, batch.event_hi, batch.clock_ms,
            batch.config_version, batch.digest[:16]))
    for window in ledger.all_windows():
        state = "SEALED" if window.sealed else "OPEN"
        print("WINDOW|%d|batches %d..%d|%s|v%d|%s" % (
            window.index, window.batch_lo, window.batch_hi, state,
            window.version, window.digest[:16] if window.digest else "-"))
        for name in sorted(window.items):
            item = window.items[name]
            print("ITEM|%s|w%d|value=%d|err<=%d|batches=%d..%d|affected=%s|"
                  "halted=%s|v%d" % (
                      item.name, window.index, item.value, item.error_bound,
                      item.batch_lo, item.batch_hi, item.affected,
                      item.halted, item.version))
    for anomaly in ledger.anomalies:
        print(anomaly.render())


def cmd_run(args):
    config, script = _load(args)
    store = SnapshotStore(args.db)
    try:
        version, changed, diff = store.negotiate_version(config, script)
        stale_archive = json.loads(store.load_meta().get("stale_archive", "[]"))
        old_windows = store.load_windows() if changed else []
        if changed:
            stale_archive += [{"window": w.index, "version": w.version,
                               "digest": w.digest}
                              for w in old_windows if w.sealed]
            print("VERSION|%d|config/script changed|%s" % (version, canonical(diff)))

        events = run_session(config, script)
        if args.crash_after_batches is not None:
            keep = args.crash_after_batches * config.batch_size
            events = events[:keep]
        ledger = build_ledger(config, version, events)
        finished = args.crash_after_batches is None and ledger.finished
        store.save_session(ledger, events, finished,
                           script_digest=script.digest,
                           stale_archive=stale_archive)
        _print_ledger(ledger)
        if changed:
            print("REPLACE|" + canonical(diff_windows(old_windows,
                                                      ledger.all_windows())))
        print("RUN_OK|events=%d|batches=%d|sealed=%d|finished=%s" % (
            len(events), len(ledger.batches), len(ledger.sealed_windows),
            finished))
    finally:
        store.close()
    return 0


def cmd_recover(args):
    config, script = _load(args)
    store = SnapshotStore(args.db)
    try:
        result = store.recover(config, script)
    except BizError as exc:
        print(exc.render())
        return EXIT_CODES[exc.code]
    finally:
        store.close()
    print("RECOVER|events=%d|unknown_batches=%s|exit_report=%s" % (
        result.recovered_events,
        ",".join(str(i) for i in result.unknown_batches) or "none",
        result.exit_report_available))
    if result.unknown_batches:
        print("UNKNOWN|batches=%s|require rerun; 不得声称能生成退出报告" % (
            ",".join(str(i) for i in result.unknown_batches)))
    return 0


def cmd_replay(args):
    config, script = _load(args)
    store = SnapshotStore(args.db)
    try:
        problem = store.verify()
        if problem is not None:
            print(problem.render())
            return EXIT_CODES[problem.code]
        meta = store.load_meta()
        if not meta:
            print("REPLAY_SKIP|no snapshot")
            return 0
        events = store.load_events()
        trace = store.load_trace()
        result = replay_verify(config, int(meta["config_version"]), events, trace)
    finally:
        store.close()
    print(str(result))
    return 0 if result.ok else EXIT_CODES[E_REPLAY]


def cmd_gui(args):
    config, script = _load(args)
    store = SnapshotStore(args.db)
    try:
        result = store.recover(config, script)
        meta = store.load_meta()
        stale_archive = json.loads(meta.get("stale_archive", "[]"))
        ledger = result.ledger
        if ledger is None:
            print("GUI_SKIP|no snapshot")
            return 0
        from ccobs.gui import build_view_model, run_gui
        model = build_view_model(
            ledger,
            sealed_stale={e["window"] for e in stale_archive},
            snapshots=["%s v%s" % (args.db, meta.get("config_version", "?"))],
        )

        def on_replay():
            replay_result = replay_verify(config, ledger.config_version,
                                          store.load_events(),
                                          store.load_trace())
            return str(replay_result)

        run_gui(model, on_replay=on_replay)
    except BizError as exc:
        print(exc.render())
        return EXIT_CODES[exc.code]
    finally:
        store.close()
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="CCObs 拥塞控制观测组件")
    parser.add_argument("--config", required=True)
    parser.add_argument("--script", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--crash-after-batches", type=int, default=None,
                        help="模拟强制终止: 只落盘前 N 个完整批次")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("run")
    sub.add_parser("recover")
    sub.add_parser("replay")
    sub.add_parser("gui")
    args = parser.parse_args(argv)
    try:
        handler = {"run": cmd_run, "recover": cmd_recover,
                   "replay": cmd_replay, "gui": cmd_gui}[args.command]
        return handler(args)
    except BizError as exc:
        print(exc.render())
        return EXIT_CODES.get(exc.code, 1)


if __name__ == "__main__":
    sys.exit(main())
