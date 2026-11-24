"""Real input-fault injection harness with isolation diagnostics."""

from __future__ import annotations

import copy
import json
import math
import tempfile
from pathlib import Path
from typing import Any

import yaml

from .commands import ensure_config_shape
from .engine import RuntimeInjection, Workbench
from .io_layer import load_json, load_yaml
from .util import atomic_write_json, atomic_write_text, fingerprint


def _find_trigger(config: dict[str, Any], domain: str | None, intent: str | None, term: str) -> dict[str, Any] | None:
    if intent is not None and domain is not None:
        triggers = config.get("domains", {}).get(domain, {}).get("intents", {}).get(intent, {}).get("triggers", [])
    elif domain is not None:
        triggers = config.get("domains", {}).get(domain, {}).get("triggers", [])
    else:
        triggers = []
    for entry in triggers:
        if isinstance(entry, dict) and (entry.get("term") == term or "".join(entry.get("tokens", [])) == term):
            return entry
    return None


def apply_injection_fault(
    fault: str,
    target: dict[str, Any],
    config: dict[str, Any],
    utterances: list[dict[str, Any]],
) -> None:
    if fault == "nonfinite_trigger_weight":
        entry = _find_trigger(config, target.get("domain"), target.get("intent"), str(target.get("term", "")))
        if entry is None:
            raise ValueError("injection target trigger not found")
        entry["weight"] = math.inf
    elif fault == "undeclared_slot_pattern":
        intent_name = str(target.get("intent", ""))
        slot_name = str(target.get("slot", "ghost_slot"))
        pattern_name = f"injected_{intent_name}_{slot_name}"
        patterns = config.setdefault("slot_patterns", {})
        patterns[pattern_name] = {
            "name": pattern_name,
            "intent": intent_name,
            "slot": slot_name,
            "tokens": ["注入锚点", "{slot}"],
            "window": 2,
            "direction": "after",
        }
        found = False
        for domain in config.get("domains", {}).values():
            if intent_name in domain.get("intents", {}):
                domain["intents"][intent_name].setdefault("slot_patterns", []).append(pattern_name)
                found = True
                break
        if not found:
            raise ValueError("injection target intent not found")
    elif fault == "turn_gap":
        utterance_id = str(target.get("utterance_id", ""))
        record = next((item for item in utterances if str(item.get("utterance_id", item.get("id"))) == utterance_id), None)
        if record is None:
            raise ValueError("injection target utterance not found")
        record["turn"] = int(target.get("turn", 999))
    elif fault == "out_of_vocab":
        utterance_id = str(target.get("utterance_id", ""))
        record = next((item for item in utterances if str(item.get("utterance_id", item.get("id"))) == utterance_id), None)
        if record is None:
            raise ValueError("injection target utterance not found")
        record.setdefault("tokens", []).append(str(target.get("token", "未登记词项")))
    elif fault == "runtime_exception":
        # This fault is supplied to the engine as a real stage exception.
        return
    else:
        raise ValueError(f"unknown injection fault {fault!r}")


def run_injections(
    *,
    utterances_path: str | Path,
    config_path: str | Path,
    injections_path: str | Path,
    vocabulary_path: str | Path | None = None,
    work_root: str | Path | None = None,
) -> dict[str, Any]:
    raw_config = ensure_config_shape(load_yaml(config_path))
    raw_utterances = [json.loads(line) for line in Path(utterances_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    injection_document = load_yaml(injections_path)
    items = injection_document.get("injections", injection_document) if isinstance(injection_document, dict) else injection_document
    if not isinstance(items, list):
        raise ValueError("injection configuration must contain an injections list")
    reports: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        injection_id = str(item.get("id", f"INJ-{index + 1:03d}"))
        fault = str(item.get("fault", ""))
        target = item.get("target", {})
        expected_code = str(item.get("expected_code", "E_ROUTE_INVALID"))
        config = copy.deepcopy(raw_config)
        utterances = copy.deepcopy(raw_utterances)
        mutation_error: str | None = None
        try:
            apply_injection_fault(fault, target, config, utterances)
        except Exception as exc:
            mutation_error = str(exc)
        with tempfile.TemporaryDirectory(prefix=f"rw-inj-{injection_id}-") as temp_name:
            root = Path(temp_name)
            atomic_write_text(root / "routing.yaml", yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
            atomic_write_text(root / "utterances.jsonl", "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in utterances))
            atomic_write_json(root / "commands.json", {"commands": []})
            runtime_injections: list[RuntimeInjection] = []
            if fault == "runtime_exception":
                runtime_injections.append(
                    RuntimeInjection(
                        injection_id=injection_id,
                        utterance_id=str(target.get("utterance_id", "")),
                        stage=str(target.get("stage", "intent")),
                        message=f"injection {injection_id}: runtime exception",
                    )
                )
            try:
                report = Workbench(
                    utterances_path=root / "utterances.jsonl",
                    config_path=root / "routing.yaml",
                    commands_path=root / "commands.json",
                    vocabulary_path=vocabulary_path,
                    state_dir=root / "state",
                    full=True,
                    batch_budget=10**9,
                    runtime_injections=runtime_injections,
                ).run()
                failure_codes = sorted({failure["business_code"] for failure in report["failures"]})
                hit = expected_code in failure_codes and mutation_error is None
                reports.append(
                    {
                        "id": injection_id,
                        "fault": fault,
                        "target": target,
                        "expected_code": expected_code,
                        "hit": hit,
                        "failure_codes": failure_codes,
                        "valid_count": report["summary"]["valid_count"],
                        "result_count": report["summary"]["result_count"],
                        "pending_count": report["summary"]["pending_count"],
                        "mutation_error": mutation_error,
                        "failure_examples": report["failures"][:5],
                    }
                )
            except Exception as exc:
                reports.append(
                    {
                        "id": injection_id,
                        "fault": fault,
                        "target": target,
                        "expected_code": expected_code,
                        "hit": False,
                        "failure_codes": [],
                        "valid_count": 0,
                        "result_count": 0,
                        "mutation_error": mutation_error,
                        "harness_error": f"{type(exc).__name__}: {exc}",
                    }
                )
    return {
        "injections": reports,
        "all_hit": all(report["hit"] for report in reports),
        "count": len(reports),
        "faults_fingerprint": fingerprint({item.get("id"): item.get("fault") for item in items if isinstance(item, dict)}),
    }

