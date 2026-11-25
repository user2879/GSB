"""Command-line entry point producing structured acceptance results.

Scenarios:

* ``demo``     - full multi-environment arbitration + incremental update
* ``large``    - scale-adaptive path selection on a large deterministic set
* ``recovery`` - forced termination, snapshot replacement and replay validation
* ``envcheck`` - interpreter / SQLite / tkinter / loopback port preflight
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import socket
from pathlib import Path
from typing import Any

from .determinism import canonical
from .db import sqlite_version
from .errors import ArbError
from .ledger import Ledger
from .platform import ArbitrationPlatform
from .samplegen import read_ndjson
from .spec import load_manifest

ROOT = Path(__file__).resolve().parent.parent


def _write(name: str, payload: Any) -> Path:
    out = ROOT / "results" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
                   encoding="utf-8")
    return out


def _free_port(preferred: int) -> int:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.settimeout(0.5)
        if probe.connect_ex(("127.0.0.1", preferred)) != 0:
            return preferred
    finally:
        probe.close()
    # Find the next free loopback port deterministically.
    candidate = preferred + 1
    while True:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            if probe.connect_ex(("127.0.0.1", candidate)) != 0:
                return candidate
        finally:
            probe.close()
        candidate += 1


def cmd_envcheck(_args: argparse.Namespace) -> dict[str, Any]:
    import sys

    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "sqlite_runtime": sqlite_version(),
        "sqlite_meets_minimum": tuple(
            int(x) for x in sqlite_version().split(".")[:2]) >= (3, 40),
    }
    tk_spec = importlib.util.find_spec("tkinter")
    if tk_spec is None:
        info["tkinter"] = "unavailable: module not found"
    else:
        try:
            importlib.import_module("tkinter")
            info["tkinter"] = "available"
        except Exception as exc:  # pragma: no cover - environment dependent
            info["tkinter"] = f"unavailable: {type(exc).__name__}"
    try:
        import numpy
        info["numpy"] = numpy.__version__
    except Exception as exc:  # pragma: no cover
        info["numpy"] = f"unavailable: {exc}"
    info["loopback"] = "127.0.0.1"
    info["sample_ports_free"] = {
        str(port): _is_free(port) for port in (49101, 49102, 49103,
                                               49211, 49212, 49213, 49214)
    }
    _write("envcheck.json", info)
    return info


def _is_free(port: int) -> bool:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.settimeout(0.5)
        return probe.connect_ex(("127.0.0.1", port)) != 0
    finally:
        probe.close()


def _load(manifest_path: str, events_path: str, db_path: str,
          shift: int = 0) -> tuple[Any, ArbitrationPlatform, list[dict[str, Any]]]:
    manifest = load_manifest(ROOT / manifest_path)
    if shift:
        manifest = dataclasses.replace(
            manifest,
            environments=tuple(
                dataclasses.replace(env, port=env.port + shift)
                for env in manifest.environments),
        )
    platform = ArbitrationPlatform(manifest, ROOT / db_path)
    events = read_ndjson(ROOT / events_path)
    return manifest, platform, events


def cmd_demo(args: argparse.Namespace) -> dict[str, Any]:
    manifest, platform, events = _load(
        "config/manifest.json", "samples/events.ndjson",
        "results/demo_ledger.sqlite")
    snapshot = platform.open_snapshot()
    platform.start_services()
    ingest = platform.ingest(events, batch_size=5)
    full = platform.run_view("view-all", use_scheduler=False)

    target = next(
        oid for oid, obs in platform.store.items()
        if obs["env_id"] == "env-alpha" and obs["key"] == "cache.ttl"
        and obs["seq"] == 6
    )
    before = {k: canonical(v.to_record())
              for k, v in full.result.verdicts.items()}
    change = platform.replace_observation(target, 999)
    incremental = platform.run_view(
        "view-all", changes=[change], use_scheduler=True,
        force_path="incremental")
    after = {k: canonical(v.to_record())
             for k, v in incremental.result.verdicts.items()}
    preserved = {k: before[k] == after[k]
                 for k in sorted(before)}
    second_view = platform.run_view("view-alpha-bravo", use_scheduler=False)
    replay = platform.replay_ledger()
    platform.shutdown()

    payload = {
        "snapshot_open": replay_outcome(snapshot),
        "ingest": ingest,
        "full_arbitration": full.as_dict(),
        "incremental_arbitration": incremental.as_dict(),
        "unaffected_keys_preserved": preserved,
        "second_view": {
            "view_id": second_view.view_id,
            "digest": second_view.result.content_digest,
        },
        "replay": replay_outcome(replay),
    }
    _write("demo_result.json", payload)
    return payload


def cmd_large(args: argparse.Namespace) -> dict[str, Any]:
    manifest, platform, events = _load(
        "config/manifest_large.json", "samples/events_large.ndjson",
        "results/large_ledger.sqlite", shift=2000)
    platform.open_snapshot()
    platform.start_services()
    platform.ingest(events, batch_size=16)
    full = platform.run_view("view-large", use_scheduler=False)

    target = next(
        oid for oid, obs in platform.store.items()
        if obs["env_id"] == "env-bravo" and obs["key"] == "pool.size"
        and obs["seq"] == 40
    )
    change = platform.replace_observation(target, 777)
    incremental = platform.run_view(
        "view-large", changes=[change], use_scheduler=True)
    platform.shutdown()

    payload = {
        "scale_parameters_full": full.path_decision["parameters"],
        "full_decision": full.path_decision,
        "incremental_decision": incremental.path_decision,
        "incremental_scheduling": incremental.scheduling,
        "authority_digests": {
            "full": full.result.content_digest,
            "incremental": incremental.result.content_digest,
        },
    }
    _write("large_result.json", payload)
    return payload


def cmd_recovery(args: argparse.Namespace) -> dict[str, Any]:
    manifest, platform, events = _load(
        "config/manifest.json", "samples/events.ndjson",
        "results/recovery_ledger.sqlite", shift=3000)
    platform.open_snapshot()
    platform.start_services()
    platform.ingest(events, batch_size=7)
    committed = platform.run_view("view-all", use_scheduler=False)

    # Begin an uncommitted view then force-kill an environment service.
    platform.ledger.begin_view("view-alpha-bravo", 1, 0)
    exit_code = platform.force_kill_service("env-bravo")
    recovery = platform.recover_after_kill()
    restart = platform.restart_service("env-bravo")
    rerun = platform.run_view("view-alpha-bravo", use_scheduler=False)
    replay = platform.replay_ledger()

    # Demonstrate whole-snapshot replacement on a digest mismatch: first
    # persist a snapshot under the current declaration, then reopen the same
    # file with an incompatible declaration (different scale power / digest).
    replacement_path = ROOT / "results/replacement_ledger.sqlite"
    seed_ledger = Ledger(replacement_path, manifest)
    seed_ledger.open_or_replace()
    seed_ledger.close()
    changed = dataclasses.replace(manifest, scale_power=3)
    replacement_ledger = Ledger(replacement_path, changed)
    replacement = replacement_ledger.open_or_replace()
    replacement_ledger.close()
    platform.shutdown()

    payload = {
        "committed_view_digest": committed.result.content_digest,
        "force_kill_exit_code": exit_code,
        "recovery": recovery,
        "restart": {"env_id": restart["env_id"],
                    "repushed": restart["repushed"]},
        "rerun_digest": rerun.result.content_digest,
        "replay": replay_outcome(replay),
        "snapshot_replacement": replay_outcome(replacement),
    }
    _write("recovery_result.json", payload)
    return payload


def replay_outcome(outcome: Any) -> dict[str, Any]:
    return {
        "matched": getattr(outcome, "matched", 0),
        "unknown_views": getattr(outcome, "unknown_views", []),
        "replaced": getattr(outcome, "replaced", False),
        "reason": getattr(outcome, "reason", None),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multi-environment arbitration")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("envcheck").set_defaults(func=cmd_envcheck)
    sub.add_parser("demo").set_defaults(func=cmd_demo)
    sub.add_parser("large").set_defaults(func=cmd_large)
    sub.add_parser("recovery").set_defaults(func=cmd_recovery)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = args.func(args)
    except ArbError as exc:
        error = exc.to_dict()
        _write("error.json", error)
        print(json.dumps(error, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
