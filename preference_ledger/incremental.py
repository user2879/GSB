from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .loading import parse_config, parse_records_text, parse_weights
from .pipeline import run_pipeline
from .snapshot import SnapshotStore


def record_key_of(record: dict[str, Any]) -> tuple[str, int]:
    return str(record["session_id"]), int(record["turn"])


class IncrementalStore:
    def __init__(
        self,
        raw_records: list[dict[str, Any]],
        raw_weights: dict[str, Any],
        raw_config: dict[str, Any],
        root: str | Path | None = None,
    ):
        self.raw_records = [dict(record) for record in raw_records]
        self.raw_weights = dict(raw_weights)
        self.raw_config = dict(raw_config)
        self.root = Path(root) if root else None
        self.snapshots = SnapshotStore(self.root) if root else None
        self.cache: dict[str, Any] = {"features": {}, "scores": {}, "windows": {}, "partitions": {}}
        self.version = 0
        self.scheduler_state: dict[str, Any] = {}
        self.last_report: dict[str, Any] | None = None
        self.last_command: dict[str, Any] | None = None
        self.last_invalidation: list[dict[str, Any]] = []

    @classmethod
    def from_loaded_files(cls, records_path: Path, weights_path: Path, config_path: Path, root: Path | None = None) -> "IncrementalStore":
        import yaml

        raw_records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        raw_weights = yaml.safe_load(weights_path.read_text(encoding="utf-8"))
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        return cls(raw_records, raw_weights, raw_config, root)

    @classmethod
    def recover(cls, root: str | Path) -> "IncrementalStore | None":
        store = SnapshotStore(root)
        committed = store.load_committed()
        if committed is None:
            return None
        instance = cls(
            committed["raw_records"],
            committed["raw_weights"],
            committed["raw_config"],
            root=root,
        )
        instance.cache = committed["cache"]
        instance.version = int(committed["version"])
        instance.scheduler_state = committed.get("scheduler_state", {})
        instance.last_report = committed.get("last_report")
        return instance

    def apply_command(self, command: dict[str, Any]) -> dict[str, Any]:
        self.last_command = command
        kind = command.get("op") or command.get("type")
        if kind in {"replace_record", "add_record", "upsert_record"}:
            record = command.get("record") or {k: v for k, v in command.items() if k not in {"op", "type"}}
            key = record_key_of(record)
            self.raw_records = [r for r in self.raw_records if record_key_of(r) != key]
            self.raw_records.append(dict(record))
            self.raw_records.sort(key=lambda r: (str(r["session_id"]), int(r["turn"])))
        elif kind == "delete_record":
            session_id = command["session_id"]
            turn = int(command["turn"])
            self.raw_records = [r for r in self.raw_records if record_key_of(r) != (session_id, turn)]
        elif kind == "delete_session":
            session_id = command["session_id"]
            self.raw_records = [r for r in self.raw_records if str(r["session_id"]) != session_id]
        elif kind in {"set_weight", "add_annotator"}:
            annotator_id = command["annotator_id"]
            self.raw_weights.setdefault("weights", {})[annotator_id] = command["weight_bp"]
        elif kind == "remove_annotator":
            annotator_id = command["annotator_id"]
            self.raw_weights.get("weights", {}).pop(annotator_id, None)
        elif kind == "set_window_lengths":
            self.raw_config["window_lengths"] = list(command["value"])
        elif kind == "set_partition_boundaries":
            self.raw_config["partition_boundaries"] = list(command["value"])
        elif kind == "set_quant":
            if "quant_scheme" in command:
                self.raw_config["quant_scheme"] = command["quant_scheme"]
            if "fixed_decimal_places" in command:
                self.raw_config["fixed_decimal_places"] = command["fixed_decimal_places"]
        elif kind == "set_exchange_version":
            self.raw_config["exchange_version"] = command["value"]
        elif kind == "set_ledger_generation":
            self.raw_config["ledger_generation"] = command["value"]
        elif kind == "set_config":
            self.raw_config[command["key"]] = command["value"]
        else:
            raise ValueError(f"unknown command op: {kind}")
        return command

    def parse_current(self):
        weights, weight_issues = parse_weights(self.raw_weights)
        config, config_issues = parse_config(self.raw_config)
        issues = [*weight_issues, *config_issues]
        records = []
        if config is not None:
            text = "\n".join(json.dumps(record, ensure_ascii=False) for record in self.raw_records)
            records, record_issues = parse_records_text(text, weights, config)
            issues.extend(record_issues)
        return config, records, weights, issues

    def run(
        self,
        workers: int = 1,
        budget: int | None = None,
        session_quota: int | None = None,
        resume_pending: bool = True,
        save: bool = True,
        crash_after_stage: str | None = None,
    ) -> dict[str, Any]:
        config, records, weights, issues = self.parse_current()
        if config is None:
            report = {"issues": [issue.to_dict() for issue in issues], "failure_list": [issue.to_dict() for issue in issues]}
            return {"report": report, "book": None, "issues": issues}

        pending_ids = None
        initial_points = None
        if resume_pending and self.scheduler_state.get("pending_task_ids") and budget is not None:
            pending_ids = list(self.scheduler_state["pending_task_ids"])
        result = run_pipeline(
            records,
            weights,
            config,
            issues=issues,
            cache=self.cache,
            budget=budget,
            session_quota=session_quota,
            workers=workers,
            pending_task_ids=pending_ids,
            initial_session_points=initial_points,
        )
        report = result["report"]
        invalidation = self._diff_invalidation(report)
        report["invalidation"] = invalidation
        report["reuse_ledger"] = self._flatten_reuse(report["cache_events"])
        self.last_invalidation = invalidation
        self.last_report = report
        self.scheduler_state = report["scheduler"]

        if save and self.snapshots is not None:
            pointer = self.snapshots.commit(
                self.raw_records,
                self.raw_weights,
                self.raw_config,
                self.cache,
                report["scheduler"],
                report,
                previous_version=self.version,
                crash_after_stage=crash_after_stage,
            )
            self.version = int(pointer["version"])
        return result

    @staticmethod
    def _flatten_reuse(events: dict[str, list[dict[str, str]]]) -> list[dict[str, str]]:
        flat: list[dict[str, str]] = []
        for stage, items in events.items():
            if stage.endswith("_reuse"):
                for item in items:
                    flat.append({**item, "stage": stage[:-6]})
        return sorted(flat, key=lambda x: (x["stage"], x.get("id", x.get("scope_id", "")), x["cache_key"]))

    def _diff_invalidation(self, new_report: dict[str, Any]) -> list[dict[str, Any]]:
        old = self.last_report
        command = self.last_command or {}
        basis = f"command:{command.get('op', command.get('type', 'initial_run'))}"
        if old is None:
            return []
        old_maps = self._key_maps(old.get("cache_keys", {}))
        new_maps = self._key_maps(new_report["cache_keys"])
        session_maps = self._session_maps(old, new_report)
        events: list[dict[str, Any]] = []
        for stage in ("feature", "score", "window", "partition"):
            old_map = old_maps[stage]
            new_map = new_maps[stage]
            for item_id in sorted(set(old_map) | set(new_map)):
                old_key = old_map.get(item_id)
                new_key = new_map.get(item_id)
                if old_key == new_key and old_key is not None:
                    continue
                sessions = session_maps[stage].get(item_id, set())
                events.append(
                    {
                        "stage": stage,
                        "id": item_id,
                        "old_key": old_key,
                        "new_key": new_key,
                        "basis": basis,
                        "sessions": sorted(sessions),
                        "crosses_session_boundary": len(sessions) > 1,
                    }
                )
        cross_stage_sessions = {session for event in events for session in event["sessions"]}
        for event in events:
            event["batch_crosses_session_boundary"] = len(cross_stage_sessions) > 1
            event["crosses_session_boundary"] = len(cross_stage_sessions) > 1
        return events

    @staticmethod
    def _key_maps(cache_keys: dict[str, Any]) -> dict[str, dict[str, str]]:
        result: dict[str, dict[str, str]] = {}
        for stage in ("feature", "score", "window", "partition"):
            result[stage] = {item["id"]: item["key"] for item in cache_keys.get(stage, [])}
        return result

    @staticmethod
    def _session_maps(old_report: dict[str, Any], new_report: dict[str, Any]) -> dict[str, dict[str, set[str]]]:
        maps: dict[str, dict[str, set[str]]] = {stage: {} for stage in ("feature", "score", "window", "partition")}
        for report in (old_report, new_report):
            for item in report.get("bucket_summaries", []):
                maps["feature"].setdefault(item["record_id"], set()).add(item["session_id"])
            for item in report.get("record_scores", []):
                maps["score"].setdefault(f"{item['session_id']}#{item['turn']:06d}", set()).add(item["session_id"])
            for stage in ("windows", "partitions"):
                key_stage = "window" if stage == "windows" else "partition"
                for item in report.get(stage, []):
                    maps[key_stage].setdefault(item["scope_id"], set()).add(item["session_id"])
        return maps
