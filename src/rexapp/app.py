"""Application orchestration: full runs, interactive streams and reports."""

from __future__ import annotations

import json
import signal
import sys
from pathlib import Path

from .commands import apply_command, load_commands
from .config import RuleConfig, load_config
from .documents import Document, load_documents
from .engine import ExtractionEngine, WorkHook
from .errors import Failure
from .snapshots import SnapshotStore


def _write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def build_engine(
    config: RuleConfig,
    documents: dict[str, Document],
    failures: list[Failure],
    snapshot_dir: str | Path,
    work_hook: WorkHook | None = None,
) -> ExtractionEngine:
    store = SnapshotStore(snapshot_dir)
    return ExtractionEngine(
        config=config,
        documents=documents,
        failures=failures,
        snapshot_store=store,
        work_hook=work_hook,
    )


def run_full_execution(
    samples_dir: str | Path,
    output_dir: str | Path,
    work_hook: WorkHook | None = None,
) -> dict:
    samples_dir = Path(samples_dir)
    output_dir = Path(output_dir)
    config = load_config(samples_dir / "config.json")
    loaded = load_documents(samples_dir / "documents.jsonl", set(config.entity_vocab))
    documents = {doc.doc_id: doc for doc in loaded.documents}
    engine = build_engine(config, documents, loaded.failures, output_dir / "snapshots", work_hook)
    outcome = engine.run_full()
    _write_json(output_dir / "full_results.json", outcome["results"])
    _write_json(output_dir / "diagnostic_report.json", engine.build_report())
    engine.commit_final_snapshot()
    return {
        "results": outcome["results"],
        "failures": engine._stable_failures() + engine._argument_missing_failures(),
        "report": engine.build_report(),
    }


def run_interactive(
    samples_dir: str | Path,
    output_dir: str | Path,
    work_hook: WorkHook | None = None,
) -> dict:
    samples_dir = Path(samples_dir)
    output_dir = Path(output_dir)
    config = load_config(samples_dir / "config.json")
    loaded = load_documents(samples_dir / "documents.jsonl", set(config.entity_vocab))
    documents = {doc.doc_id: doc for doc in loaded.documents}
    commands = load_commands(samples_dir / "commands.json")
    engine = build_engine(config, documents, loaded.failures, output_dir / "snapshots", work_hook)

    responses = []
    stop_requested = False

    def _handler(signum, frame) -> None:  # pragma: no cover - exercised via CLI test
        nonlocal stop_requested
        stop_requested = True

    previous_handler = signal.signal(signal.SIGTERM, _handler)
    try:
        for command in commands:
            if stop_requested:
                engine.commit_final_snapshot()
                return {"shutdown": True, "exit_code": 0, "responses": responses}
            new_config, new_documents, effect = apply_command(command, config, documents)
            response = engine.apply_interaction(
                effect.command_id, effect, new_config, new_documents
            )
            config = new_config
            documents = new_documents
            responses.append(response.as_dict())
            if not response.latest:
                pending = engine.drain_pending(effect.affected_docs)
                responses[-1]["drained_pending"] = pending
    finally:
        signal.signal(signal.SIGTERM, previous_handler)

    engine.commit_final_snapshot()
    _write_json(output_dir / "interaction_responses.json", responses)
    _write_json(output_dir / "incremental_results.json", engine.export_results())
    _write_json(output_dir / "diagnostic_report.json", engine.build_report())
    return {
        "shutdown": False,
        "exit_code": 0,
        "responses": responses,
        "results": engine.export_results(),
        "failures": engine._stable_failures() + engine._argument_missing_failures(),
        "report": engine.build_report(),
    }


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    mode = argv[0] if argv else "full"
    samples_dir = Path(argv[1]) if len(argv) > 1 else Path("samples")
    output_dir = Path(argv[2]) if len(argv) > 2 else Path("output")
    if mode == "full":
        run_full_execution(samples_dir, output_dir)
        return 0
    if mode == "interactive":
        outcome = run_interactive(samples_dir, output_dir)
        return int(outcome["exit_code"])
    print(f"unknown mode {mode}", file=sys.stderr)
    return 2
