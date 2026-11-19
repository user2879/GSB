"""End-to-end acceptance: incremental vs cold full-run field equality."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from rexapp.app import build_engine
from rexapp.commands import apply_command, load_commands
from rexapp.config import load_config
from rexapp.documents import load_documents

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
OUTPUT = ROOT / "output"


def main() -> int:
    config = load_config(SAMPLES / "config.json")
    loaded = load_documents(SAMPLES / "documents.jsonl", set(config.entity_vocab))
    documents = {doc.doc_id: doc for doc in loaded.documents}
    commands = load_commands(SAMPLES / "commands.json")

    engine = build_engine(config, documents, loaded.failures, OUTPUT / "snapshots")
    documents_state = dict(documents)
    config_state = config
    checks = []
    for command in commands:
        new_config, new_documents, effect = apply_command(command, config_state, documents_state)
        response = engine.apply_interaction(effect.command_id, effect, new_config, new_documents)
        if not response.latest:
            engine.drain_pending(effect.affected_docs)
        config_state, documents_state = new_config, new_documents

        with tempfile.TemporaryDirectory() as tmp:
            cold = build_engine(config_state, documents_state, loaded.failures, Path(tmp) / "cold")
            cold.run_full()
            equal = engine.export_results() == cold.export_results()
            failures_equal = engine.build_report()["failures"] == cold.build_report()["failures"]
        checks.append(
            {
                "command_id": command["command_id"],
                "type": command["type"],
                "results_match": equal,
                "failures_match": failures_equal,
                "response_status": response.status,
                "response_latest": response.latest,
            }
        )

    all_match = all(item["results_match"] and item["failures_match"] for item in checks)
    payload = {
        "all_commands_match_cold_full_run": all_match,
        "checks": checks,
        "reuse": {
            "reused": engine.cache.reused_count,
            "recomputed": engine.cache.recomputed_count,
            "invalidated": engine.cache.invalidated_count,
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "acceptance.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if all_match else 1


if __name__ == "__main__":
    raise SystemExit(main())
