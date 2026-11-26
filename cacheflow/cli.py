from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .canonical import canonical_bytes
from .errors import CacheFlowError
from .evolve import EvolutionChange, evolve_and_verify
from .replay import verify_ledger
from .runner import Runner
from .spec import load_bundle, load_nodes_spec


DEFAULT_NODES = Path("examples/nodes.json")
DEFAULT_ARTIFACTS = Path("examples/artifacts.json")
DEFAULT_EVENTS = Path("examples/events.json")


def _print_json(value: Any) -> None:
    sys.stdout.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _write(path: Path, raw: bytes | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(raw, bytes):
        path.write_bytes(raw)
    else:
        path.write_text(raw, encoding="utf-8")


def cmd_validate(args: argparse.Namespace) -> int:
    bundle = load_bundle(args.nodes, args.artifacts, args.events)
    _print_json(
        {
            "ok": True,
            "code": "E_OK",
            "spec_digest": bundle.spec_digest(),
            "node_count": len(bundle.nodes),
            "artifact_count": len(bundle.artifacts),
            "event_count": len(bundle.events),
            "ports": [{"node_id": node.node_id, "host": node.host, "port": node.port} for node in bundle.nodes],
        }
    )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    bundle = load_bundle(args.nodes, args.artifacts, args.events)
    work_dir = Path(args.work_dir)
    with Runner(bundle, work_dir, rpc_timeout_seconds=args.timeout_seconds) as runner:
        result = runner.run()
    conclusion_path = work_dir / "conclusion.canonical.json"
    _write(conclusion_path, result["snapshot_bytes"])
    metrics = {
        "ok": True,
        "snapshot_digest": result["snapshot_digest"],
        "conclusion_path": str(conclusion_path),
        "peaks": result["peaks"],
        "wait_records": result["wait_records"],
    }
    _write(work_dir / "run-report.json", canonical_bytes(metrics))
    _print_json(
        {
            "ok": True,
            "snapshot_digest": result["snapshot_digest"],
            "conclusion_path": str(conclusion_path),
            "peaks": result["peaks"],
            "wait_record_count": len(result["wait_records"]),
        }
    )
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    bundle = load_bundle(args.nodes, args.artifacts, args.events)
    result = verify_ledger(bundle, args.ledger)
    output = Path(args.out) if args.out else Path(args.ledger).parent / "replay.canonical.json"
    _write(output, result["canonical_bytes"])
    _print_json(
        {
            "ok": True,
            "code": "E_OK",
            "digest": result["digest"],
            "cutoff_seq": result["cutoff_seq"],
            "unique_holding_ok": result["unique_holding_ok"],
            "output": str(output),
        }
    )
    return 0


def cmd_evolve(args: argparse.Namespace) -> int:
    bundle = load_bundle(args.nodes, args.artifacts, args.events)
    raw = json.loads(Path(args.evolution).read_text(encoding="utf-8"))
    evolved_nodes = None
    if raw.get("evolved_nodes_file"):
        nodes_path = Path(raw["evolved_nodes_file"])
        if not nodes_path.is_absolute():
            nodes_path = Path(args.evolution).parent / nodes_path
        _, evolved_nodes = load_nodes_spec(nodes_path)
    change = EvolutionChange(
        seq=int(raw["seq"]),
        new_quota_version=None if raw.get("quota_version") is None else int(raw["quota_version"]),
        removed_node_ids=tuple(raw.get("removed_node_ids", [])),
        changed_seed_node_ids=tuple(raw.get("changed_seed_node_ids", [])),
        added_node_ids=tuple(raw.get("added_node_ids", [])),
        reason=str(raw.get("reason", "declared_evolution")),
    )
    result = evolve_and_verify(
        bundle,
        cutoff_seq=args.cutoff_seq,
        change=change,
        evolved_nodes=evolved_nodes,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write(out_dir / "evolution-conclusion.canonical.json", canonical_bytes(result["conclusion"]))
    report = {
        "ok": True,
        "digest": result["digest"],
        "untouched_ok": result["untouched_ok"],
        "incremental_matches_full_rebuild": result["incremental_matches_full_rebuild"],
        "impact": result["impact"],
    }
    _write(out_dir / "evolution-report.json", canonical_bytes(report))
    _print_json(
        {
            "ok": True,
            "code": "E_OK",
            "digest": result["digest"],
            "affected_lease_ids": result["impact"]["affected_lease_ids"],
            "affected_node_ids": result["impact"]["affected_node_ids"],
            "unaffected_lease_ids": result["impact"]["unaffected_lease_ids"],
            "untouched_ok": result["untouched_ok"],
            "report_dir": str(out_dir),
        }
    )
    return 0


def cmd_accept(args: argparse.Namespace) -> int:
    from .acceptance import run_acceptance

    report = run_acceptance(Path(args.work_dir), base_port=args.base_port)
    _print_json(report)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cacheflow", description="Deterministic multi-node cache pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate")
    validate.add_argument("--nodes", default=DEFAULT_NODES)
    validate.add_argument("--artifacts", default=DEFAULT_ARTIFACTS)
    validate.add_argument("--events", default=DEFAULT_EVENTS)
    validate.set_defaults(func=cmd_validate)

    run = sub.add_parser("run")
    run.add_argument("--nodes", default=DEFAULT_NODES)
    run.add_argument("--artifacts", default=DEFAULT_ARTIFACTS)
    run.add_argument("--events", default=DEFAULT_EVENTS)
    run.add_argument("--work-dir", default="workspace/sample")
    run.add_argument("--timeout-seconds", type=float, default=8.0)
    run.set_defaults(func=cmd_run)

    replay = sub.add_parser("replay")
    replay.add_argument("--nodes", default=DEFAULT_NODES)
    replay.add_argument("--artifacts", default=DEFAULT_ARTIFACTS)
    replay.add_argument("--events", default=DEFAULT_EVENTS)
    replay.add_argument("--ledger", default="workspace/sample/cacheflow.sqlite3")
    replay.add_argument("--out", default=None)
    replay.set_defaults(func=cmd_replay)

    evolve = sub.add_parser("evolve")
    evolve.add_argument("--nodes", default=DEFAULT_NODES)
    evolve.add_argument("--artifacts", default=DEFAULT_ARTIFACTS)
    evolve.add_argument("--events", default=DEFAULT_EVENTS)
    evolve.add_argument("--evolution", default="examples/evolution.json")
    evolve.add_argument("--cutoff-seq", type=int, default=20)
    evolve.add_argument("--out-dir", default="workspace/evolution")
    evolve.set_defaults(func=cmd_evolve)

    accept = sub.add_parser("accept")
    accept.add_argument("--work-dir", default="workspace/acceptance")
    accept.add_argument("--base-port", type=int, default=49700)
    accept.set_defaults(func=cmd_accept)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except CacheFlowError as exc:
        payload = {"ok": False, **exc.to_dict()}
        sys.stderr.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        return 2
    except FileNotFoundError as exc:
        sys.stderr.write(json.dumps({"ok": False, "code": "E_SPEC", "message": str(exc)}, ensure_ascii=False) + "\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
