"""Targeted fault injection that exercises the real stage validators."""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any

from .config import WorkbenchConfig


STAGE_ORDER = {"load": 0, "config": 1, "command": 2, "split": 3, "scoring": 4,
               "selection": 5, "rewrite": 6, "summary": 7, "snapshot": 8}


class InjectionController:
    def __init__(self, entries: list[dict[str, Any]]) -> None:
        self.entries = entries
        self.by_doc_stage: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for entry in entries:
            document_id = entry.get("document_id", entry.get("doc_id"))
            stage = entry.get("stage")
            if isinstance(document_id, str) and isinstance(stage, str):
                self.by_doc_stage.setdefault((document_id, stage), []).append(entry)
        self.reports: list[dict[str, Any]] = []
        self._reported: set[str] = set()

    def apply(self, document: dict[str, Any], config: WorkbenchConfig,
              stage: str) -> tuple[dict[str, Any], WorkbenchConfig, dict[str, Any] | None]:
        document_id = document.get("document_id", "")
        candidates = self.by_doc_stage.get((document_id, stage), [])
        if not candidates:
            return document, config, None
        entry = candidates[0]
        injection_id = str(entry.get("id", "INJ-UNKNOWN"))
        kind = entry.get("kind", entry.get("inject", ""))
        expected_code = entry.get("code", entry.get("expected_code"))
        mutated_document = copy.deepcopy(document)
        mutated_config = config
        hit = True
        status = "applied"

        try:
            if stage == "split" and kind == "empty_paragraph":
                mutated_document["paragraphs"].append("")
            elif stage == "scoring" and kind == "illegal_entity":
                mutated_document.setdefault("entities", []).append({"id": "", "text": "", "type": ""})
            elif stage == "selection" and kind == "zero_summary_length":
                mutated_config = replace(config, max_summary_sentences=0)
            elif stage == "scoring" and kind in {"decimal_width_out_of_range", "precision_out_of_range"}:
                mutated_config = replace(config, fixed_decimal_places=99)
            elif stage == "rewrite" and kind in {"missing_template", "missing_template_reference"}:
                raw = copy.deepcopy(config.raw)
                entries = raw["synonym_tables"][config.synonym_table_version]["entries"]
                if not entries:
                    entries.append({"template_id": "__missing__", "source": "x", "candidates": ["y"]})
                else:
                    entries[0]["template_id"] = "__missing__"
                mutated_config = replace(config, raw=raw)
            else:
                hit = False
                status = "unsupported_kind"
        except Exception:
            hit = False
            status = "mutation_failed"

        report = {
            "id": injection_id,
            "document_id": document_id,
            "stage": stage,
            "kind": kind,
            "expected_code": expected_code,
            "hit": hit,
            "status": status,
        }
        self.reports.append(report)
        self._reported.add(injection_id)
        if not hit:
            return document, config, None
        return mutated_document, mutated_config, report

    def final_report(self) -> list[dict[str, Any]]:
        reports = [dict(item) for item in self.reports]
        for entry in self.entries:
            injection_id = str(entry.get("id", "INJ-UNKNOWN"))
            if injection_id not in self._reported:
                reports.append({
                    "id": injection_id,
                    "document_id": entry.get("document_id", entry.get("doc_id", "")),
                    "stage": entry.get("stage", ""),
                    "kind": entry.get("kind", entry.get("inject", "")),
                    "expected_code": entry.get("code", entry.get("expected_code")),
                    "hit": False,
                    "status": "unreached",
                })
        return reports

