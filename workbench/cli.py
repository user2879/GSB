from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .acceptance import run_acceptance
from .errors import SpecError, WorkbenchError
from .spec import dump_example_spec, load_spec


def _print(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="workbench", description="Local shard lease/water workbench")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="write a validated example JSON spec")
    init.add_argument("path", type=Path)

    validate = sub.add_parser("validate", help="validate a JSON spec")
    validate.add_argument("path", type=Path)

    run = sub.add_parser("run", help="start loopback nodes and run all shards")
    run.add_argument("path", type=Path)
    run.add_argument("--workdir", type=Path, default=Path("artifacts") / "run")
    run.add_argument("--db", type=Path, default=None)
    run.add_argument("--snapshot", type=Path, default=None)
    run.add_argument("--serial", action="store_true")
    run.add_argument("--workers", type=int, default=3)
    run.add_argument("--batch-size", type=int, default=None)

    samples = sub.add_parser("export-samples", help="write deterministic sample JSON files")
    samples.add_argument("path", type=Path)
    samples.add_argument("--out-dir", type=Path, default=Path("artifacts") / "samples")

    acceptance = sub.add_parser("acceptance", help="run local end-to-end acceptance")
    acceptance.add_argument("--output", type=Path, default=Path("artifacts") / "acceptance")

    sub.add_parser("gui", help="open the Tkinter diagnostic workbench")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "init":
        path = dump_example_spec(args.path)
        spec = load_spec(path)
        _print({"ok": True, "path": str(path), "spec_digest": spec.digest})
        return 0

    if args.command == "validate":
        try:
            spec = load_spec(args.path)
        except SpecError as exc:
            _print(exc.to_dict())
            return 2
        _print(
            {
                "ok": True,
                "code": "OK",
                "path": str(args.path),
                "spec_digest": spec.digest,
                "topo_order": list(spec.topo_order),
                "nodes": len(spec.nodes),
                "shards": len(spec.shards),
            }
        )
        return 0

    if args.command == "acceptance":
        report = run_acceptance(args.output)
        _print(
            {
                "ok": report["ok"],
                "output_dir": report["output_dir"],
                "checks": [check["name"] for check in report["checks"]],
                "error": report.get("error"),
            }
        )
        return 0 if report["ok"] else 1

    if args.command == "export-samples":
        from .canonical import content_digest
        from .invalidation import rendezvous_node_id
        from .samples import generate_samples, sample_digest
        from .snapshot import atomic_write_json

        spec = load_spec(args.path)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        written = []
        node_ids = [node["node_id"] for node in spec.nodes]
        for shard in spec.shards:
            shard_id = shard["shard_id"]
            node_id = shard.get("node_id") or rendezvous_node_id(node_ids, shard_id)
            node = spec.node_by_id[node_id]
            rows = int(shard["sample_rows"])
            payload = {
                "shard_id": shard_id,
                "node_id": node_id,
                "node_seed": int(node["seed"]),
                "sample_rows": rows,
                "sample_digest": sample_digest(int(node["seed"]), shard_id, rows),
                "samples": generate_samples(int(node["seed"]), shard_id, rows),
            }
            output = args.out_dir / f"{shard_id}.json"
            atomic_write_json(output, payload)
            written.append(str(output))
        _print({"ok": True, "files": written, "content_digest": content_digest(written)})
        return 0

    if args.command == "gui":
        from .gui import main as gui_main

        return gui_main()

    if args.command == "run":
        from .engine import WorkbenchEngine

        spec = load_spec(args.path)
        engine = WorkbenchEngine(
            spec,
            args.workdir,
            db_path=args.db,
            snapshot_path=args.snapshot,
            parallel=not args.serial,
            max_workers=args.workers,
            batch_size=args.batch_size,
        )
        try:
            result = engine.run_until_complete()
            _print({"ok": True, **result})
            return 0
        except WorkbenchError as exc:
            _print(exc.to_dict())
            return 1
        finally:
            engine.close()

    return 64


if __name__ == "__main__":
    sys.exit(main())
