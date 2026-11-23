"""Command line interface for the multi-cluster diagnostics console."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .engine import Engine, NodeKey
from .errors import DiagError, canonical_json
from .inject import (inject_corrupt_spec, inject_dangling_derives,
                     inject_snapshot_digest)
from .ledger import Ledger
from .model import Revision, Spec, load_spec
from .orchestrator import Orchestrator
from .snapshot import build_snapshot, load_snapshot, save_snapshot

DEFAULT_SAMPLES = Path("samples")
DEFAULT_DATA = Path("data")


def _paths(args) -> dict:
    base = Path(args.samples)
    return {
        "clusters": base / "clusters.json",
        "tasks": base / "tasks.json",
        "revisions": Path(args.revisions) if args.revisions
        else base / "revisions.json",
    }


def _print(payload) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


def _progress(label):
    def callback(index, total, key: NodeKey):
        if index == total or index % max(1, total // 6) == 0:
            print(f"[{label}] {index}/{total} {key.task_id} "
                  f"{key.cluster_id} r{key.version}")
    return callback


def cmd_validate(args) -> int:
    paths = _paths(args)
    spec = load_spec(paths["clusters"], paths["tasks"], paths["revisions"])
    _print({"status": "ok", "clusters": len(spec.clusters),
            "tasks": len(spec.tasks), "revisions": len(spec.revisions),
            "head": spec.revisions[-1].version})
    return 0


def cmd_gen_samples(args) -> int:
    from scripts.generate_samples import main as gen_main  # type: ignore
    return gen_main(["--out", args.samples])


def _open_engine(args, start_services: bool = True):
    paths = _paths(args)
    spec = load_spec(paths["clusters"], paths["tasks"], paths["revisions"])
    data = Path(args.data)
    data.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(data / "ledger.db")
    orchestrator = Orchestrator(spec) if start_services and not args.local \
        else None
    if orchestrator is not None:
        orchestrator.start_all()
    return spec, ledger, orchestrator, Engine(spec, ledger, orchestrator)


def _revision_records(spec) -> list[dict]:
    return [{"version": r.version, "values": r.values,
             "strategy": r.strategy, "derives": list(r.derives)}
            for r in spec.revisions]


def cmd_apply(args) -> int:
    spec, ledger, orch, engine = _open_engine(args)
    try:
        state_path = Path(args.data) / "last_chain.json"
        previous = json.loads(state_path.read_text("utf-8")) \
            if state_path.exists() else None
        result = engine.apply_chain(
            previous, batch_id=f"apply-{time.time_ns()}",
            progress=_progress("apply"))
        state_path.write_text(canonical_json(_revision_records(spec)),
                              encoding="utf-8")
        snapshot = build_snapshot(
            spec, spec.revisions[-1].version,
            [n for n in ledger.list_nodes("committed")
             if n["version"] in set(spec.ancestors_including(
                 spec.revisions[-1].version))],
            {"stage": "apply", "head": spec.revisions[-1].version})
        save_snapshot(Path(args.data) / "snapshot.json", snapshot)
        _print(result)
    finally:
        if orch is not None:
            orch.stop_all()
        ledger.close()
    return 0


def cmd_rebuild(args) -> int:
    spec, ledger, orch, engine = _open_engine(args)
    try:
        root = args.root if args.root is not None else spec.revisions[-1].version
        result = engine.rebuild(root, use_services=not args.local,
                                progress=_progress("rebuild"))
        _print(result)
    finally:
        if orch is not None:
            orch.stop_all()
        ledger.close()
    return 0


def cmd_rollback(args) -> int:
    spec, ledger, orch, engine = _open_engine(args)
    try:
        result = engine.rollback(args.target, progress=_progress("rollback"))
        chain_versions = set(spec.ancestors_including(args.target))
        snapshot = build_snapshot(
            spec, args.target,
            [n for n in ledger.list_nodes("committed")
             if n["version"] in chain_versions],
            {"stage": "rollback", "root": args.target})
        save_snapshot(Path(args.data) / "snapshot.json", snapshot)
        _print(result)
    finally:
        if orch is not None:
            orch.stop_all()
        ledger.close()
    return 0


def cmd_view(args) -> int:
    spec, ledger, _, engine = _open_engine(args, start_services=False)
    try:
        _print(engine.active_view())
    finally:
        ledger.close()
    return 0


def cmd_snapshot_save(args) -> int:
    spec, ledger, _, engine = _open_engine(args, start_services=False)
    try:
        root = ledger.get_root() or spec.revisions[-1].version
        snapshot = build_snapshot(
            spec, root, ledger.list_nodes("committed"),
            {"stage": "manual", "root": root})
        save_snapshot(Path(args.data) / args.name, snapshot)
        _print({"saved": str(Path(args.data) / args.name), "root": root,
                "nodes": len(snapshot["nodes"])})
    finally:
        ledger.close()
    return 0


def cmd_recover(args) -> int:
    spec, ledger, _, engine = _open_engine(args, start_services=False)
    try:
        report = engine.recover()
        _print(report)
    finally:
        ledger.close()
    return 0


def cmd_rerun(args) -> int:
    spec, ledger, orch, engine = _open_engine(args)
    try:
        _print(engine.rerun_unknown())
    finally:
        if orch is not None:
            orch.stop_all()
        ledger.close()
    return 0


def cmd_replay(args) -> int:
    spec, ledger, orch, engine = _open_engine(args)
    try:
        snapshot_path = Path(args.data) / "snapshot.json"
        snapshot = load_snapshot(snapshot_path, spec)
        result = engine.replay(snapshot, use_services=not args.local,
                               progress=_progress("replay"))
        _print(result)
    finally:
        if orch is not None:
            orch.stop_all()
        ledger.close()
    return 0


def cmd_inject(args) -> int:
    paths = _paths(args)
    if args.kind == "corrupt-spec":
        target = inject_corrupt_spec(paths["revisions"], args.data)
        args.revisions = target
        return _run_injected_spec(args)
    if args.kind == "dangling-derives":
        target = inject_dangling_derives(paths["revisions"], args.data)
        args.revisions = target
        return _run_injected_ref(args)
    if args.kind == "snapshot-digest":
        snapshot_path = Path(args.data) / "snapshot.json"
        target = inject_snapshot_digest(snapshot_path, args.data)
        spec, ledger, _, _ = _open_engine(args, start_services=False)
        try:
            load_snapshot(target, spec)
        except DiagError as exc:
            _print(exc.as_dict())
            return 0 if exc.code == "E_SNAPSHOT" else 2
        finally:
            ledger.close()
    return 2


def _run_injected_spec(args) -> int:
    try:
        paths = _paths(args)
        load_spec(paths["clusters"], paths["tasks"], paths["revisions"])
    except DiagError as exc:
        _print(exc.as_dict())
        return 0 if exc.code == "E_SPEC" else 2
    return 2


def _run_injected_ref(args) -> int:
    # The dangling reference is created deliberately after load-time
    # validation, so build the model directly and let the engine's reference
    # guard classify it as E_REF (an internal integrity violation).
    import json as _json
    from .model import Cluster, Revision, Task
    paths = _paths(args)
    clusters_raw = _json.loads(Path(paths["clusters"]).read_text("utf-8"))
    tasks_raw = _json.loads(Path(paths["tasks"]).read_text("utf-8"))
    revisions_raw = _json.loads(Path(paths["revisions"]).read_text("utf-8"))
    base = load_spec(paths["clusters"], paths["tasks"],
                     _paths_safe_revisions(args))
    revisions = []
    for item in revisions_raw["revisions"]:
        revisions.append(Revision(item["version"], item.get("parent"),
                                  item.get("values", {}),
                                  item.get("strategy", "rolling"),
                                  tuple(item.get("derives", []))))
    spec = Spec(clusters=base.clusters, tasks=base.tasks,
                revisions=tuple(revisions))
    ledger = Ledger(Path(args.data) / "ledger_ref_inject.db")
    engine = Engine(spec, ledger)
    try:
        engine.validate_references()
    except DiagError as exc:
        _print(exc.as_dict())
        return 0 if exc.code == "E_REF" else 2
    finally:
        ledger.close()
    return 2


def _paths_safe_revisions(args):
    return Path(args.samples) / "revisions.json"


def cmd_demo(args) -> int:
    """Repeatable end-to-end demonstration used for acceptance."""
    from .demo import run_demo
    report = run_demo(Path(args.samples), Path(args.data),
                      local=args.local, crash=args.crash)
    _print(report)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="diagconsole",
        description="Multi-cluster config revision diagnostics console")
    parser.add_argument("--samples", default=str(DEFAULT_SAMPLES))
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--revisions", default=None)
    parser.add_argument("--local", action="store_true",
                        help="compute in-process without spawning services")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("gen-samples").set_defaults(func=cmd_gen_samples)
    sub.add_parser("validate").set_defaults(func=cmd_validate)
    sub.add_parser("apply").set_defaults(func=cmd_apply)
    rebuild = sub.add_parser("rebuild")
    rebuild.add_argument("--root", type=int, default=None)
    rebuild.set_defaults(func=cmd_rebuild)
    rollback = sub.add_parser("rollback")
    rollback.add_argument("--target", type=int, required=True)
    rollback.set_defaults(func=cmd_rollback)
    sub.add_parser("view").set_defaults(func=cmd_view)
    snap = sub.add_parser("snapshot-save")
    snap.add_argument("--name", default="snapshot.json")
    snap.set_defaults(func=cmd_snapshot_save)
    sub.add_parser("recover").set_defaults(func=cmd_recover)
    sub.add_parser("rerun").set_defaults(func=cmd_rerun)
    sub.add_parser("replay").set_defaults(func=cmd_replay)
    inject = sub.add_parser("inject")
    inject.add_argument("kind", choices=["corrupt-spec", "dangling-derives",
                                         "snapshot-digest"])
    inject.set_defaults(func=cmd_inject)
    demo = sub.add_parser("demo")
    demo.add_argument("--crash", action="store_true",
                      help="simulate a forced termination mid-batch")
    demo.set_defaults(func=cmd_demo)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except DiagError as exc:
        print(json.dumps(exc.as_dict(), indent=2, sort_keys=True,
                         ensure_ascii=False), file=sys.stderr)
        return 1
