from __future__ import annotations

import argparse
import json
from pathlib import Path

from .incremental import IncrementalStore
from .loading import load_input
from .pipeline import run_pipeline
from .verify import run_verification


ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="preference-ledger", description="Deterministic preference reward ledger")
    sub = parser.add_subparsers(dest="command", required=True)

    verify_parser = sub.add_parser("verify", help="run all local acceptance checks")
    verify_parser.add_argument("--root", default=str(ROOT))

    run_parser = sub.add_parser("run", help="run one full batch")
    run_parser.add_argument("--root", default=str(ROOT))
    run_parser.add_argument("--state", default=None)
    run_parser.add_argument("--workers", type=int, default=1)
    run_parser.add_argument("--out", default="artifacts/run_report.json")

    crash_parser = sub.add_parser("crash", help="crash during snapshot commit")
    crash_parser.add_argument("--state", required=True)
    crash_parser.add_argument("--stage", default="03_scores.json")

    args = parser.parse_args(argv)
    if args.command == "verify":
        report = run_verification(args.root)
        print(json.dumps({"ok": report["ok"], "report": "artifacts/verification_report.json"}, ensure_ascii=False))
        return 0 if report["ok"] else 1

    root = Path(getattr(args, "root", ROOT))
    if args.command == "run":
        samples = root / "samples"
        loaded = load_input(samples / "preferences.jsonl", samples / "annotators.yaml", samples / "config.yaml")
        if loaded.config is None:
            print(json.dumps([i.to_dict() for i in loaded.issues], ensure_ascii=False, indent=2))
            return 2
        result = run_pipeline(loaded.records, loaded.weights, loaded.config, loaded.issues, workers=args.workers)
        out = root / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result["report"], ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"ok": not loaded.issues, "out": str(out), "windows": len(result["report"]["windows"])}))
        return 0 if not loaded.issues else 1

    if args.command == "crash":
        state = Path(args.state)
        store = IncrementalStore.recover(state)
        if store is None:
            samples = ROOT / "samples"
            store = IncrementalStore.from_loaded_files(
                samples / "preferences.jsonl",
                samples / "annotators.yaml",
                samples / "config.yaml",
                root=state,
            )
        store.run(save=True, crash_after_stage=args.stage)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
