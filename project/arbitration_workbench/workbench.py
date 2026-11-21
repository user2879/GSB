"""Interactive workbench orchestration around the deterministic engine."""

from __future__ import annotations

import json
import threading
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from .cache import CacheStore, build_reference_index
from .config import config_fingerprint, parse_config
from .constants import (
    ERROR_EVIDENCE_TYPE_UNKNOWN,
    ERROR_INTERACT_BUDGET_PENDING,
    ERROR_JSON_INVALID,
    ERROR_ROLE_UNKNOWN,
    ERROR_SNAPSHOT_STALE,
)
from .engine import ArbitrationEngine, RunResult
from .errors import ConfigValidationError, WorkbenchError
from .loader import evidence_lines_to_jsonl, load_evidence_text
from .model import ArbitrationConfig, Rules
from .numeric import atomic_write_text, read_json, write_json
from .rules import parse_rules, rules_fingerprint
from .snapshot import SnapshotStore


class Workbench:
    def __init__(
        self,
        *,
        root: str | Path,
        rules_path: str | Path = "samples/rules.json",
        config_path: str | Path = "samples/config.json",
        evidence_path: str | Path = "samples/evidence.jsonl",
        exception_injections_path: str | Path | None = "samples/exception_injections.json",
        output_dir: str | Path = "outputs",
        snapshot_enabled: bool = True,
        persist_edits: bool = True,
        expected_snapshot_version: int | None = None,
    ):
        self.root = Path(root)
        self.rules_path = self._resolve(rules_path)
        self.config_path = self._resolve(config_path)
        self.evidence_path = self._resolve(evidence_path)
        self.exception_path = self._resolve(exception_injections_path) if exception_injections_path else None
        self.output_dir = output_dir
        self.persist_edits = persist_edits
        self.expected_snapshot_version = expected_snapshot_version
        try:
            rules_raw = read_json(self.rules_path)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise WorkbenchError(ERROR_JSON_INVALID, f"rules JSON is invalid: {exc}") from exc
        try:
            config_raw = read_json(self.config_path)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise WorkbenchError(ERROR_JSON_INVALID, f"config JSON is invalid: {exc}") from exc
        self.rules = parse_rules(rules_raw)
        self.config = parse_config(config_raw)
        self.evidence_lines = self.evidence_path.read_text(encoding="utf-8").splitlines()
        self.exception_injections = self._load_exception_injections()
        self.snapshot_store: SnapshotStore | None = None
        self.startup_code: str | None = None
        self.startup_status: dict = {"ok": False, "code": "E_SNAPSHOT_DISABLED"}
        self.cache_store = CacheStore()
        self.global_version = 0
        self.last_result: RunResult | None = None
        self.last_summary: dict = {
            "completed_ticket_count": 0,
            "current_upgrade_count": 0,
            "failure_count": 0,
            "boundary_ticket_ids": [],
            "snapshot_version": 0,
            "interrupted": False,
        }
        self.recovery_info: dict = {"available": False}
        self._background_thread: threading.Thread | None = None
        self._background_error: Exception | None = None
        if snapshot_enabled:
            self._initialize_snapshot_resume()
        self.engine = self._new_engine(self.cache_store, self.global_version, self.snapshot_store)

    def _resolve(self, path: str | Path | None) -> Path:
        if path is None:
            raise ValueError("path cannot be None")
        candidate = Path(path)
        return candidate if candidate.is_absolute() else self.root / candidate

    def _load_exception_injections(self) -> list[dict]:
        if self.exception_path is None or not self.exception_path.exists():
            return []
        raw = read_json(self.exception_path)
        if isinstance(raw, list):
            return [item for item in raw if isinstance(item, dict)]
        if isinstance(raw, dict) and isinstance(raw.get("injections"), list):
            return [item for item in raw["injections"] if isinstance(item, dict)]
        raise ConfigValidationError(ERROR_JSON_INVALID, "exception injection file must be a list or contain injections")

    def _initialize_snapshot_resume(self) -> None:
        self.snapshot_store = SnapshotStore(
            self.root / self.config.snapshot_dir,
            self.root / self.config.temp_dir,
        )
        try:
            manifest = self.snapshot_store.latest_manifest()
        except Exception as exc:  # snapshot parse problems are surfaced, not silently reused
            self.startup_status = {"ok": False, "code": getattr(exc, "code", "E_SNAPSHOT_INVALID"), "message": str(exc)}
            self.startup_code = self.startup_status["code"]
            return
        if manifest is None:
            self.startup_status = {"ok": False, "code": "E_SNAPSHOT_ABSENT", "message": "starting from blank workspace"}
            return
        config_match = manifest.get("config_hash") == config_fingerprint(self.config)
        rules_match = manifest.get("rules_fingerprint") == rules_fingerprint(self.rules)
        version_match = (
            self.expected_snapshot_version is None
            or int(manifest.get("global_version", -1)) == int(self.expected_snapshot_version)
        )
        if not config_match or not rules_match or not version_match:
            stale_reasons = []
            if not config_match:
                stale_reasons.append("config_hash")
            if not rules_match:
                stale_reasons.append("rules_fingerprint")
            if not version_match:
                stale_reasons.append("global_version")
            self.startup_code = ERROR_SNAPSHOT_STALE
            self.startup_status = {
                "ok": False,
                "code": ERROR_SNAPSHOT_STALE,
                "message": "snapshot is stale; cache resume disabled",
                "stale_reasons": stale_reasons,
                "manifest": manifest,
            }
            self.recovery_info = self.snapshot_store.recovery_info(manifest)
            return
        try:
            cache_payload = self.snapshot_store.load_cache(manifest)
            self.cache_store = CacheStore.from_dict(cache_payload)
            self.global_version = int(manifest.get("global_version", 0))
            self.last_summary = self.snapshot_store.load_artifact(manifest, "summary")
            self.recovery_info = self.snapshot_store.recovery_info(manifest)
            self.startup_status = {"ok": True, "code": "OK", "manifest": manifest}
        except Exception as exc:
            self.startup_code = getattr(exc, "code", "E_SNAPSHOT_INVALID")
            self.startup_status = {"ok": False, "code": self.startup_code, "message": str(exc)}
            self.recovery_info = self.snapshot_store.recovery_info(manifest)

    def _new_engine(
        self,
        cache_store: CacheStore,
        global_version: int,
        snapshot_store: SnapshotStore | None,
    ) -> ArbitrationEngine:
        return ArbitrationEngine(
            rules=self.rules,
            config=self.config,
            evidence_lines=self.evidence_lines,
            workspace_root=self.root,
            cache_store=cache_store,
            snapshot_store=snapshot_store,
            global_version=global_version,
            exception_injections=self.exception_injections,
            output_dir=self.output_dir,
            write_outputs=True,
        )

    def _save_inputs(self) -> None:
        if not self.persist_edits:
            return
        temp_dir = self.root / self.config.temp_dir
        write_json(self.rules_path, self.rules.as_dict(), temp_dir=temp_dir)
        write_json(self.config_path, self.config.as_dict(), temp_dir=temp_dir)
        atomic_write_text(
            self.evidence_path,
            evidence_lines_to_jsonl(self.evidence_lines),
            temp_dir=temp_dir,
        )

    def _wait_for_background(self) -> None:
        if self._background_thread is not None and self._background_thread.is_alive():
            self._background_thread.join()

    def latest_summary(self) -> dict:
        if self.last_result is not None:
            return self.last_result.summary()
        return deepcopy(self.last_summary)

    def status(self) -> dict:
        return {
            "startup": self.startup_status,
            "recovery": self.recovery_info,
            "latest_summary": self.latest_summary(),
            "snapshot_version": self.global_version,
            "config_hash": config_fingerprint(self.config),
            "rules_fingerprint": rules_fingerprint(self.rules),
        }

    def reference_index(self) -> dict:
        loaded = load_evidence_text("\n".join(self.evidence_lines), self.rules)
        return build_reference_index(loaded.evidence_by_ticket, self.rules)

    def _affected(
        self,
        *,
        old_index: dict,
        new_index: dict,
        op: str,
        target: str | None,
        ticket_id: str | None = None,
    ) -> list[str]:
        all_tickets = sorted(set(old_index.get("tickets", [])) | set(new_index.get("tickets", [])))
        if op == "set_type_weight":
            return sorted(
                set(old_index.get("evidence_types", {}).get(target or "", []))
                | set(new_index.get("evidence_types", {}).get(target or "", []))
            )
        if op == "set_role_trust":
            return sorted(
                set(old_index.get("roles", {}).get(target or "", []))
                | set(new_index.get("roles", {}).get(target or "", []))
            )
        if op in {"upsert_relation", "delete_relation"}:
            return sorted(
                set(old_index.get("relations", {}).get(target or "", []))
                | set(new_index.get("relations", {}).get(target or "", []))
            )
        if op == "replace_evidence":
            return [ticket_id] if ticket_id else []
        if op in {
            "set_threshold",
            "set_half_life",
            "set_fusion",
            "set_priority_bands",
            "set_target_version",
        }:
            return all_tickets
        return []

    def apply_edit(self, edit: dict) -> list[str]:
        if not isinstance(edit, dict) or not isinstance(edit.get("op"), str):
            raise WorkbenchError(ERROR_JSON_INVALID, "edit must contain a string op")
        self._wait_for_background()
        old_index = self.reference_index()
        old_rules = self.rules
        old_config = self.config
        rules_raw = deepcopy(self.rules.as_dict())
        config_raw = deepcopy(self.config.as_dict())
        evidence_lines = list(self.evidence_lines)
        op = edit["op"]
        target: str | None = None
        ticket_id: str | None = None

        if op == "set_type_weight":
            target = edit.get("evidence_type")
            if target not in rules_raw["evidence_type_weights"]:
                raise WorkbenchError(ERROR_EVIDENCE_TYPE_UNKNOWN, f"unknown evidence type {target!r}")
            rules_raw["evidence_type_weights"][target] = edit.get("value")
        elif op == "set_role_trust":
            target = edit.get("source_role") or edit.get("role")
            if target not in rules_raw["role_trust"]:
                raise WorkbenchError(ERROR_ROLE_UNKNOWN, f"unknown source role {target!r}")
            rules_raw["role_trust"][target] = edit.get("value")
        elif op == "upsert_relation":
            relation = edit.get("relation")
            if not isinstance(relation, dict):
                raise WorkbenchError(ERROR_JSON_INVALID, "upsert_relation requires a relation object")
            target = relation.get("id")
            if not isinstance(target, str) or not target:
                raise WorkbenchError(ERROR_JSON_INVALID, "relation id must be non-empty")
            relations = [item for item in rules_raw["relations"] if item.get("id") != target]
            relations.append(deepcopy(relation))
            rules_raw["relations"] = relations
        elif op == "delete_relation":
            target = edit.get("relation_id")
            before = len(rules_raw["relations"])
            rules_raw["relations"] = [item for item in rules_raw["relations"] if item.get("id") != target]
            if len(rules_raw["relations"]) == before:
                raise WorkbenchError(ERROR_JSON_INVALID, f"relation {target!r} does not exist")
        elif op == "set_threshold":
            config_raw["decision_threshold"] = edit.get("value")
        elif op == "set_half_life":
            config_raw["decay_half_life"] = edit.get("value")
        elif op == "set_fusion":
            config_raw["fusion"] = edit.get("value")
        elif op == "set_priority_bands":
            bands = edit.get("bands")
            if not isinstance(bands, list):
                raise WorkbenchError(ERROR_JSON_INVALID, "set_priority_bands requires bands array")
            rules_raw["priority_bands"] = deepcopy(bands)
        elif op == "set_target_version":
            config_raw["target_version"] = edit.get("value")
        elif op == "set_worker_threads":
            config_raw["worker_threads"] = edit.get("value")
        elif op == "set_interaction_budget":
            config_raw["interaction_budget_ms"] = edit.get("value")
        elif op == "set_snapshot_commit_interval":
            config_raw["snapshot_commit_interval"] = edit.get("value")
        elif op == "replace_evidence":
            ticket_id = edit.get("ticket_id")
            evidence_seq = edit.get("evidence_seq")
            replacement = edit.get("record")
            if not isinstance(ticket_id, str) or not isinstance(evidence_seq, int) or not isinstance(replacement, dict):
                raise WorkbenchError(ERROR_JSON_INVALID, "replace_evidence requires ticket_id, evidence_seq and record")
            found = False
            new_lines: list[str] = []
            for line in evidence_lines:
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    new_lines.append(line)
                    continue
                if (
                    isinstance(raw, dict)
                    and raw.get("ticket_id") == ticket_id
                    and raw.get("evidence_seq") == evidence_seq
                ):
                    if found:
                        raise WorkbenchError(ERROR_JSON_INVALID, "duplicate target evidence record")
                    merged = deepcopy(raw)
                    for key, value in replacement.items():
                        merged[key] = value
                    merged["ticket_id"] = ticket_id
                    merged["evidence_seq"] = evidence_seq
                    new_lines.append(json.dumps(merged, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
                    found = True
                else:
                    new_lines.append(line)
            if not found:
                raise WorkbenchError(ERROR_JSON_INVALID, f"evidence {ticket_id}/{evidence_seq} not found")
            evidence_lines = new_lines
        else:
            raise WorkbenchError(ERROR_JSON_INVALID, f"unknown edit op {op!r}")

        try:
            new_rules = parse_rules(rules_raw)
            new_config = parse_config(config_raw)
        except Exception:
            self.rules = old_rules
            self.config = old_config
            raise
        self.rules = new_rules
        self.config = new_config
        self.evidence_lines = evidence_lines
        new_index = self.reference_index()
        affected = self._affected(
            old_index=old_index,
            new_index=new_index,
            op=op,
            target=target,
            ticket_id=ticket_id,
        )
        self._save_inputs()
        self.engine.rules = new_rules
        self.engine.config = new_config
        self.engine.evidence_lines = list(evidence_lines)
        self.engine.clear_stop()
        return affected

    def run(self, *, use_cache: bool = True, mode: str = "incremental") -> RunResult:
        self._wait_for_background()
        result = self.engine.run(use_cache=use_cache, mode=mode)
        self.last_result = result
        self.last_summary = result.summary()
        self.global_version = result.global_version
        self.cache_store = self.engine.cache
        if result.manifest is not None and self.snapshot_store is not None:
            self.recovery_info = self.snapshot_store.recovery_info(result.manifest)
        return result

    def apply_edit_and_run(self, edit: dict, *, budget_ms: int | None = None) -> dict:
        command_started = time.monotonic()
        affected = self.apply_edit(edit)
        budget = self.config.interaction_budget_ms if budget_ms is None else max(0, int(budget_ms))
        previous_summary = self.latest_summary()
        holder: dict[str, Any] = {}

        def target() -> None:
            try:
                holder["result"] = self.engine.run(use_cache=True, mode="incremental")
            except Exception as exc:  # pragma: no cover - surfaced to caller
                holder["error"] = exc
                self._background_error = exc
            finally:
                if holder.get("result") is not None:
                    self.last_result = holder["result"]
                    self.last_summary = holder["result"].summary()
                    self.global_version = holder["result"].global_version
                    self.cache_store = self.engine.cache
                    if holder["result"].manifest is not None and self.snapshot_store is not None:
                        self.recovery_info = self.snapshot_store.recovery_info(holder["result"].manifest)

        thread = threading.Thread(target=target, name="arbitration-interactive-run", daemon=True)
        self._background_thread = thread
        thread.start()
        elapsed_ms = (time.monotonic() - command_started) * 1000
        thread.join(max(0.0, (budget - elapsed_ms) / 1000.0))
        base = {
            "budget_ms": budget,
            "affected_tickets": affected,
            "latest_summary": previous_summary,
            "progress": {
                "state": "pending",
                "completed_ticket_count": previous_summary.get("completed_ticket_count", 0),
                "affected_ticket_count": len(affected),
            },
        }
        if thread.is_alive():
            return {
                **base,
                "status": "pending",
                "code": ERROR_INTERACT_BUDGET_PENDING,
                "message": "latest result is not ready within the interaction budget; previous summary retained",
                "progress": {
                    **base["progress"],
                    "state": "pending",
                },
            }
        if "error" in holder:
            exc = holder["error"]
            return {
                **base,
                "status": "error",
                "code": getattr(exc, "code", "E_RUN_FAILED"),
                "message": str(exc),
            }
        result: RunResult = holder["result"]
        progress = {
            "state": "stopped" if result.interrupted else "complete",
            "completed_ticket_count": result.summary()["completed_ticket_count"],
            "affected_ticket_count": len(affected),
        }
        return {
            **base,
            "status": "stopped" if result.interrupted else "ok",
            "summary": result.summary(),
            "snapshot_version": result.global_version,
            "interrupted": result.interrupted,
            "progress": progress,
        }

    def full_rebuild(self, *, output_dir: str | Path | None = None) -> RunResult:
        full_output = output_dir or (self.root / "tmp" / "full-rebuild")
        engine = ArbitrationEngine(
            rules=self.rules,
            config=self.config,
            evidence_lines=self.evidence_lines,
            workspace_root=self.root,
            cache_store=CacheStore(),
            snapshot_store=None,
            global_version=0,
            exception_injections=self.exception_injections,
            output_dir=full_output,
            write_outputs=True,
        )
        return engine.run(use_cache=False, mode="full")

    def compare_incremental_with_full(self, incremental: RunResult | None = None) -> dict:
        incremental = incremental or self.last_result
        if incremental is None:
            raise RuntimeError("no incremental result to compare")
        full = self.full_rebuild()
        left = incremental.comparison_payload()
        right = full.comparison_payload()
        return {
            "equal": left == right,
            "incremental_ticket_count": len(incremental.tickets),
            "full_ticket_count": len(full.tickets),
            "incremental_failure_count": len(incremental.failures),
            "full_failure_count": len(full.failures),
            "artifact_hashes_equal": left.get("artifact_sha256s") == right.get("artifact_sha256s"),
            "diff_basis": "tickets+evidence scores+relation contributions+decisions+failures+exchange hashes",
        }

    def request_stop(self) -> None:
        self.engine.request_stop()
