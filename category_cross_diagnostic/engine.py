"""Incremental, cache-backed diagnostic engine."""

from __future__ import annotations

import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from .budget import (
    cardinality_fingerprint_params,
    decision_fingerprint_params,
    dictionary_fingerprint_params,
    pending_record,
    point_cost,
    adjudicate_combo,
)
from .cache import CacheStore
from .config import LoadedBundle, Strategy, policy_fingerprint, width_gear_fingerprint
from .constants import (
    ERROR_POINT_BUDGET_EXHAUSTED,
    ERROR_SHUTDOWN_REQUESTED,
    SCHEMA_VERSION,
    STAGE_CARDINALITY,
    STAGE_DECISION,
    STAGE_DICTIONARY,
    STAGE_ENUMERATION,
)
from .encoding import EncodingResult, encode_combo
from .enumeration import Combo, enumerate_combinations
from .flip_analysis import build_flip_report
from .hashing import stable_task_key
from .sample_data import TableData, generate_table, numeric_quantization_summary
from .snapshot import SnapshotStore


class ConcurrencyMonitor:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active = 0
        self.observed_max_active_tasks = 0

    def __enter__(self) -> "ConcurrencyMonitor":
        with self.lock:
            self.active += 1
            self.observed_max_active_tasks = max(
                self.observed_max_active_tasks, self.active
            )
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        with self.lock:
            self.active -= 1


class DiagnosticEngine:
    def __init__(
        self,
        bundle: LoadedBundle,
        table: TableData | None = None,
        *,
        state_dir: str | Path = ".diagnostic_state",
        serial: bool = False,
        stop_event: threading.Event | None = None,
        commit_interval: int | None = None,
    ) -> None:
        if bundle.strategy is None:
            raise ValueError("LoadedBundle does not contain a strategy")
        self.bundle = bundle
        self.strategy: Strategy = bundle.strategy
        self.table = table or generate_table(bundle)
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.cache = CacheStore(self.state_dir / "cache.json")
        self.snapshots = SnapshotStore(self.state_dir)
        self.serial = serial
        self.stop_event = stop_event or threading.Event()
        self.commit_interval = max(
            1, int(commit_interval or self.strategy.snapshot_commit_interval)
        )
        self.monitor = ConcurrencyMonitor()
        self._signal_installed = False

    @classmethod
    def from_yaml(
        cls,
        features_path: str | Path,
        strategy_path: str | Path | None = None,
        **kwargs: Any,
    ) -> "DiagnosticEngine":
        from .config import load_bundle

        bundle = load_bundle(features_path, strategy_path)
        return cls(bundle, **kwargs)

    def request_stop(self) -> None:
        self.stop_event.set()

    def install_signal_handlers(self) -> None:
        import signal

        def handler(signum: int, frame: Any) -> None:
            self.request_stop()

        for signal_name in ("SIGINT", "SIGTERM", "SIGBREAK"):
            signum = getattr(signal, signal_name, None)
            if signum is None:
                continue
            try:
                signal.signal(signum, handler)
            except (ValueError, OSError):
                # Non-main threads and some Windows execution hosts do not
                # allow registration for every termination signal.
                continue
        self._signal_installed = True

    def run(self) -> dict[str, Any]:
        width_fp = width_gear_fingerprint(
            self.strategy.encoding_width,
            self.strategy.fixed_point_width,
            self.strategy.quantization_levels,
        )
        policy_hash = policy_fingerprint(self.strategy)
        combos = enumerate_combinations(self.table, self.strategy)
        cost = point_cost(self.table.row_count, self.strategy.encoding_width)
        process_count = min(
            len(combos), self.strategy.point_budget // max(1, cost) + 1
        ) if cost else len(combos)

        recovery = self._load_recovery(policy_hash, width_fp)
        if recovery["decision"] == "resume":
            snapshot = recovery["snapshot"]
            processed_reports: list[dict[str, Any]] = list(
                snapshot.get("processed_combinations", {}).get("items", [])
            )
            ledger: list[dict[str, Any]] = list(
                snapshot.get("cache_ledger", {}).get("items", [])
            )
            completed_ids = set(
                snapshot["manifest"].get("completed_combo_ids", [])
            )
        else:
            processed_reports = []
            ledger = []
            completed_ids = set()

        identities = [combo.to_identity_dict(width_fp) for combo in combos]
        if not ledger:
            for combo in combos:
                self._ledger_enumeration(combo, width_fp, ledger)

        plans: dict[str, dict[str, Any]] = {}
        for index, combo in enumerate(combos[:process_count]):
            if combo.combo_id in completed_ids:
                continue
            plans[combo.combo_id] = self._plan_material(combo, width_fp, index)

        if not self.serial and plans and not self.stop_event.is_set():
            self._materialize_parallel(plans)

        processed_ids = {report["combo_id"] for report in processed_reports}
        since_commit = 0
        for index, combo in enumerate(combos[:process_count]):
            if combo.combo_id in processed_ids:
                continue
            if self.stop_event.is_set():
                break

            plan = plans[combo.combo_id]
            if self.serial:
                encoding_result = self._encode_with_monitor(combo)
                material = self._accept_material(
                    combo, plan, width_fp, ledger, encoding_result
                )
            else:
                material = self._accept_material(combo, plan, width_fp, ledger)

            dictionary_value, cardinality = material
            consumed_before = index * cost
            decision_param_fp = self._decision_parameter_fingerprint(
                index, consumed_before
            )
            decision_key = combo.cache_key(
                STAGE_DECISION, decision_param_fp, width_fp
            )
            if self.cache.has(decision_key):
                decision = self.cache.get(decision_key)
                decision_status = "reused"
                decision_reason = "cache_hit"
            else:
                decision = adjudicate_combo(
                    combo,
                    self.table.column_fingerprints,
                    cardinality,
                    dictionary_value["entries"],
                    row_count=self.table.row_count,
                    encoding_width=self.strategy.encoding_width,
                    consumed_points=consumed_before,
                    strategy=self.strategy,
                    width_gear_fp=width_fp,
                )
                self.cache.put(
                    decision_key,
                    decision,
                    stage=STAGE_DECISION,
                    combo_id=combo.combo_id,
                    combo_fingerprint=decision["combo_fingerprint"],
                    base_fingerprint=combo.base_fingerprint,
                    parameter_fingerprint=decision_param_fp,
                    width_gear_fingerprint=width_fp,
                    column_fingerprints={
                        name: self.table.column_fingerprints[name]
                        for name in combo.columns
                    },
                )
                decision_status = "recomputed"
                decision_reason = "cache_miss"

            ledger.append(
                self._ledger_entry(
                    sequence=len(ledger),
                    combo=combo,
                    stage=STAGE_DECISION,
                    cache_key=decision_key,
                    parameter_fp=decision_param_fp,
                    width_fp=width_fp,
                    status=decision_status,
                    reason=decision_reason,
                )
            )

            report = self._build_combo_report(
                combo,
                width_fp,
                decision_key,
                plan,
                dictionary_value,
                cardinality,
                decision,
            )
            processed_reports.append(report)
            processed_ids.add(combo.combo_id)
            since_commit += 1
            if since_commit >= self.commit_interval:
                self._commit_snapshot(
                    policy_hash,
                    width_fp,
                    combos,
                    processed_reports,
                    ledger,
                    status="checkpoint",
                )
                since_commit = 0

        report_by_id = {item["combo_id"]: item for item in processed_reports}
        ordered_reports = [
            report_by_id[combo.combo_id]
            for combo in combos
            if combo.combo_id in report_by_id
        ]

        pending = self._build_pending(
            combos, process_count, processed_ids, width_fp
        )
        stop_requested = self.stop_event.is_set()
        run_status = (
            "interrupted"
            if stop_requested and any(item["code"] == ERROR_SHUTDOWN_REQUESTED for item in pending)
            else ("point_budget_exhausted" if pending else "completed")
        )

        flip_report = (
            build_flip_report(combos, self.table, self.strategy)
            if self.strategy.enable_flip_analysis
            else {"summary": {"flip_count": 0}, "combo_flip_attribution": {}}
        )
        for item in ordered_reports:
            item["flip_attribution"] = {
                "flip": any(
                    record["flip"]
                    for record in flip_report.get("combo_flip_attribution", {}).get(
                        item["combo_id"], []
                    )
                ),
                "records": flip_report.get("combo_flip_attribution", {}).get(
                    item["combo_id"], []
                ),
            }

        final_snapshot = self._commit_snapshot(
            policy_hash,
            width_fp,
            combos,
            ordered_reports,
            ledger,
            status=run_status,
            pending=pending,
        )
        self.cache.save()

        degradations = [
            {
                "combo_id": item["combo_id"],
                "columns": item["columns"],
                "degradations": item["budget_decision"]["degradations"],
                "dominant_constraint": item["budget_decision"]["dominant_constraint"],
                "joint_constraint_triggered": item["budget_decision"][
                    "joint_constraint_triggered"
                ],
            }
            for item in ordered_reports
            if item["budget_decision"]["degradations"]
        ]
        collision_statistics = self._collision_statistics(ordered_reports)
        cache_lists = self._cache_lists(ledger)

        report: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": run_status,
            "ok": self.bundle.ok,
            "global_version": self.strategy.global_version,
            "policy_config_hash": policy_hash,
            "table_fingerprint": self.table.table_fingerprint,
            "width_gear_fingerprint": width_fp,
            "width_gear": self.strategy.width_gear_name,
            "issues": self.bundle.issues,
            "failure_count": len(
                [issue for issue in self.bundle.issues if issue["severity"] == "error"]
            ),
            "feature_table": {
                "table_name": self.table.table_name,
                "seed": self.table.seed,
                "row_count": self.table.row_count,
                "column_names": self.table.column_names(),
                "categorical_columns": self.table.categorical_columns(),
                "column_fingerprints": dict(sorted(self.table.column_fingerprints.items())),
                "replaced_columns": sorted(self.table.replacements),
            },
            "strategy": {
                "candidate_pairs": [
                    list(pair) for pair in self.strategy.candidate_pairs
                ],
                "order_limit": self.strategy.order_limit,
                "cardinality_budget": self.strategy.cardinality_budget,
                "point_budget": self.strategy.point_budget,
                "encoding_width": self.strategy.encoding_width,
                "fixed_point_width": self.strategy.fixed_point_width,
                "quantization_levels": self.strategy.quantization_levels,
                "over_budget_action": self.strategy.over_budget_action,
                "max_workers": self.strategy.max_workers,
                "snapshot_commit_interval": self.strategy.snapshot_commit_interval,
                "expansion_mode": self.strategy.expansion_mode,
            },
            "combination_count": len(combos),
            "adjudicated_count": len(ordered_reports),
            "combinations": ordered_reports,
            "pending": pending,
            "degradations": degradations,
            "collision_statistics": collision_statistics,
            "numeric_quantization": numeric_quantization_summary(
                self.table,
                self.strategy.quantization_levels,
                self.strategy.fixed_point_width,
            ),
            "flip_attribution": flip_report,
            "cache_ledger": ledger,
            "reused_cache_keys": cache_lists["reused"],
            "recomputed_cache_keys": cache_lists["recomputed"],
            "cache_summary": {
                "ledger_entries": len(ledger),
                "reused_count": len(cache_lists["reused"]),
                "recomputed_count": len(cache_lists["recomputed"]),
                "persisted_entries": len(self.cache.entries),
            },
            "snapshot": {
                "recovery_decision": recovery["decision"],
                "recovery_reason": recovery.get("reason"),
                "latest_manifest": final_snapshot["manifest"],
            },
            "concurrency": {
                "serial": self.serial,
                "configured_max_workers": self.strategy.max_workers,
                "observed_max_active_tasks": (
                    1 if self.serial else self.monitor.observed_max_active_tasks
                ),
                "stable_key_namespace": "sha256(stage_name + combo_fingerprint)",
            },
        }
        return report

    def _load_recovery(
        self, policy_hash: str, width_fp: str
    ) -> dict[str, Any]:
        snapshot = self.snapshots.load_latest()
        if snapshot is None:
            return {"decision": "fresh", "reason": "no_snapshot"}
        manifest = snapshot["manifest"]
        checks = {
            "global_version": manifest.get("global_version")
            == self.strategy.global_version,
            "policy_config_hash": manifest.get("policy_config_hash") == policy_hash,
            "width_gear_fingerprint": manifest.get("width_gear_fingerprint")
            == width_fp,
            "table_fingerprint": manifest.get("table_fingerprint")
            == self.table.table_fingerprint,
        }
        if all(checks.values()):
            return {"decision": "resume", "reason": "manifest_compatible", "snapshot": snapshot}
        failed = [name for name, passed in checks.items() if not passed]
        return {
            "decision": "rerun",
            "reason": "manifest_mismatch:" + ",".join(failed),
            "snapshot": None,
        }

    def _ledger_enumeration(
        self, combo: Combo, width_fp: str, ledger: list[dict[str, Any]]
    ) -> None:
        parameter_fp = combo.enumeration_params_fingerprint
        cache_key = combo.cache_key(STAGE_ENUMERATION, parameter_fp, width_fp)
        if self.cache.has(cache_key):
            status = "reused"
            reason = "cache_hit"
        else:
            identity = combo.to_identity_dict(width_fp)
            self.cache.put(
                cache_key,
                identity,
                stage=STAGE_ENUMERATION,
                combo_id=combo.combo_id,
                combo_fingerprint=identity["combo_fingerprint"],
                base_fingerprint=combo.base_fingerprint,
                parameter_fingerprint=parameter_fp,
                width_gear_fingerprint=width_fp,
                column_fingerprints={
                    name: self.table.column_fingerprints[name]
                    for name in combo.columns
                },
            )
            status = "recomputed"
            reason = "cache_miss"
        ledger.append(
            self._ledger_entry(
                sequence=len(ledger),
                combo=combo,
                stage=STAGE_ENUMERATION,
                cache_key=cache_key,
                parameter_fp=parameter_fp,
                width_fp=width_fp,
                status=status,
                reason=reason,
            )
        )

    def _plan_material(
        self, combo: Combo, width_fp: str, index: int
    ) -> dict[str, Any]:
        dictionary_param_fp = dictionary_fingerprint_params(self.strategy)
        cardinality_param_fp = cardinality_fingerprint_params(self.strategy)
        dictionary_key = combo.cache_key(
            STAGE_DICTIONARY, dictionary_param_fp, width_fp
        )
        cardinality_key = combo.cache_key(
            STAGE_CARDINALITY, cardinality_param_fp, width_fp
        )
        dict_hit = self.cache.has(dictionary_key)
        card_hit = self.cache.has(cardinality_key)
        return {
            "combo": combo,
            "index": index,
            "width_fp": width_fp,
            "dictionary_param_fp": dictionary_param_fp,
            "cardinality_param_fp": cardinality_param_fp,
            "dictionary_key": dictionary_key,
            "cardinality_key": cardinality_key,
            "dict_hit": dict_hit,
            "card_hit": card_hit,
            "needs_encode": not (dict_hit and card_hit),
            "task_key": stable_task_key(
                combo.width_fingerprint(width_fp), "cross_encode"
            ),
        }

    def _materialize_parallel(self, plans: dict[str, dict[str, Any]]) -> None:
        jobs = [plan for plan in plans.values() if plan["needs_encode"]]
        jobs.sort(key=lambda plan: plan["task_key"])
        results: dict[str, EncodingResult] = {}
        if not jobs:
            return
        with ThreadPoolExecutor(max_workers=self.strategy.max_workers) as executor:
            future_map = {
                executor.submit(self._encode_with_monitor, plan["combo"]): plan[
                    "combo"
                ].combo_id
                for plan in jobs
            }
            pending = set(future_map)
            while pending:
                if self.stop_event.is_set():
                    for future in pending:
                        future.cancel()
                    break
                done, pending = wait(
                    pending, timeout=0.05, return_when=FIRST_COMPLETED
                )
                for future in done:
                    combo_id = future_map[future]
                    results[combo_id] = future.result()
            for future in pending:
                future.cancel()
        for combo_id, result in results.items():
            plans[combo_id]["encoding_result"] = result

    def _encode_with_monitor(self, combo: Combo) -> EncodingResult:
        with self.monitor:
            return encode_combo(
                combo,
                self.table,
                encoding_width=self.strategy.encoding_width,
                fixed_point_width=self.strategy.fixed_point_width,
                quantization_levels=self.strategy.quantization_levels,
                preview_limit=self.strategy.dictionary_preview_limit,
            )

    def _accept_material(
        self,
        combo: Combo,
        plan: dict[str, Any],
        width_fp: str,
        ledger: list[dict[str, Any]],
        encoding_result: EncodingResult | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if encoding_result is None:
            encoding_result = plan.get("encoding_result")
        if plan["dict_hit"]:
            dictionary_envelope = self.cache.get(plan["dictionary_key"])
            dictionary_value = dictionary_envelope
            dict_status = "reused"
            dict_reason = "cache_hit"
        else:
            assert encoding_result is not None
            dictionary_value = {
                "dictionary_summary": encoding_result.dictionary_summary(
                    self.strategy.dictionary_preview_limit
                ),
                "entries": [dict(entry) for entry in encoding_result.entries],
                "encoding_profile": {
                    "encoding_width": encoding_result.encoding_width,
                    "fixed_point_width": encoding_result.fixed_point_width,
                    "quantization_levels": encoding_result.quantization_levels,
                    "code_space_size": encoding_result.code_space_size,
                },
            }
            self.cache.put(
                plan["dictionary_key"],
                dictionary_value,
                stage=STAGE_DICTIONARY,
                combo_id=combo.combo_id,
                combo_fingerprint=combo.width_fingerprint(width_fp),
                base_fingerprint=combo.base_fingerprint,
                parameter_fingerprint=plan["dictionary_param_fp"],
                width_gear_fingerprint=width_fp,
                column_fingerprints={
                    name: self.table.column_fingerprints[name]
                    for name in combo.columns
                },
            )
            dict_status = "recomputed"
            dict_reason = "cache_miss"

        ledger.append(
            self._ledger_entry(
                sequence=len(ledger),
                combo=combo,
                stage=STAGE_DICTIONARY,
                cache_key=plan["dictionary_key"],
                parameter_fp=plan["dictionary_param_fp"],
                width_fp=width_fp,
                status=dict_status,
                reason=dict_reason,
            )
        )

        if plan["card_hit"]:
            cardinality = self.cache.get(plan["cardinality_key"])
            card_status = "reused"
            card_reason = "cache_hit"
        else:
            assert encoding_result is not None
            cardinality = encoding_result.cardinality_diagnosis()
            self.cache.put(
                plan["cardinality_key"],
                cardinality,
                stage=STAGE_CARDINALITY,
                combo_id=combo.combo_id,
                combo_fingerprint=combo.width_fingerprint(width_fp),
                base_fingerprint=combo.base_fingerprint,
                parameter_fingerprint=plan["cardinality_param_fp"],
                width_gear_fingerprint=width_fp,
                column_fingerprints={
                    name: self.table.column_fingerprints[name]
                    for name in combo.columns
                },
            )
            card_status = "recomputed"
            card_reason = "cache_miss"

        ledger.append(
            self._ledger_entry(
                sequence=len(ledger),
                combo=combo,
                stage=STAGE_CARDINALITY,
                cache_key=plan["cardinality_key"],
                parameter_fp=plan["cardinality_param_fp"],
                width_fp=width_fp,
                status=card_status,
                reason=card_reason,
            )
        )
        return dictionary_value, cardinality

    def _decision_parameter_fingerprint(
        self, index: int, consumed_before: int
    ) -> str:
        from .hashing import sha256_hex

        return sha256_hex(
            "decision-sequence-v2",
            decision_fingerprint_params(self.strategy),
            "sequence_index",
            index,
            "consumed_points_before",
            consumed_before,
        )

    def _ledger_entry(
        self,
        *,
        sequence: int,
        combo: Combo,
        stage: str,
        cache_key: str,
        parameter_fp: str,
        width_fp: str,
        status: str,
        reason: str,
    ) -> dict[str, Any]:
        active_fp = combo.width_fingerprint(width_fp)
        return {
            "sequence": sequence,
            "combo_id": combo.combo_id,
            "stage": stage,
            "cache_key": cache_key,
            "status": status,
            "reason": reason,
            "task_sort_key": stable_task_key(active_fp, stage),
            "combo_fingerprint": active_fp,
            "base_fingerprint": combo.base_fingerprint,
            "parameter_fingerprint": parameter_fp,
            "width_gear_fingerprint": width_fp,
            "column_fingerprints": {
                name: self.table.column_fingerprints[name]
                for name in combo.columns
            },
        }

    def _build_combo_report(
        self,
        combo: Combo,
        width_fp: str,
        decision_key: str,
        plan: dict[str, Any],
        dictionary_value: dict[str, Any],
        cardinality: dict[str, Any],
        decision: dict[str, Any],
    ) -> dict[str, Any]:
        identity = combo.to_identity_dict(width_fp)
        return {
            **identity,
            "status": "adjudicated",
            "column_fingerprints": {
                name: self.table.column_fingerprints[name]
                for name in combo.columns
            },
            "cache_keys": {
                STAGE_ENUMERATION: combo.cache_key(
                    STAGE_ENUMERATION,
                    combo.enumeration_params_fingerprint,
                    width_fp,
                ),
                STAGE_DICTIONARY: plan["dictionary_key"],
                STAGE_CARDINALITY: plan["cardinality_key"],
                STAGE_DECISION: decision_key,
            },
            "dictionary_summary": dictionary_value["dictionary_summary"],
            "cardinality": cardinality,
            "budget_decision": decision,
        }

    def _build_pending(
        self,
        combos: list[Combo],
        process_count: int,
        processed_ids: set[str],
        width_fp: str,
    ) -> list[dict[str, Any]]:
        pending: list[dict[str, Any]] = []
        queue_index = 0
        for index, combo in enumerate(combos):
            if combo.combo_id in processed_ids:
                continue
            if index >= process_count:
                reason = ERROR_POINT_BUDGET_EXHAUSTED
            elif self.stop_event.is_set():
                reason = ERROR_SHUTDOWN_REQUESTED
            else:
                reason = ERROR_SHUTDOWN_REQUESTED
            record = pending_record(combo, width_fp, queue_index, reason)
            record["global_combination_index"] = index
            pending.append(record)
            queue_index += 1
        return pending

    def _commit_snapshot(
        self,
        policy_hash: str,
        width_fp: str,
        combos: list[Combo],
        processed_reports: list[dict[str, Any]],
        ledger: list[dict[str, Any]],
        *,
        status: str,
        pending: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        completed_ids = [item["combo_id"] for item in processed_reports]
        cost = point_cost(self.table.row_count, self.strategy.encoding_width)
        consumed_points = len(completed_ids) * cost
        if processed_reports:
            consumed_points = processed_reports[-1]["budget_decision"]["point"][
                "consumed_after"
            ]
        manifest = {
            "status": status,
            "global_version": self.strategy.global_version,
            "policy_config_hash": policy_hash,
            "table_fingerprint": self.table.table_fingerprint,
            "width_gear_fingerprint": width_fp,
            "completed_combo_ids": completed_ids,
            "completed_combo_count": len(completed_ids),
            "total_combo_count": len(combos),
            "consumed_points": consumed_points,
            "point_budget": self.strategy.point_budget,
        }
        artifacts = {
            "enumeration": {
                "items": [combo.to_identity_dict(width_fp) for combo in combos]
            },
            "dictionary": {
                "items": [
                    {
                        "combo_id": item["combo_id"],
                        "dictionary_summary": item["dictionary_summary"],
                    }
                    for item in processed_reports
                ]
            },
            "cardinality": {
                "items": [
                    {
                        "combo_id": item["combo_id"],
                        "cardinality": item["cardinality"],
                    }
                    for item in processed_reports
                ]
            },
            "decisions": {
                "items": [
                    {
                        "combo_id": item["combo_id"],
                        "budget_decision": item["budget_decision"],
                    }
                    for item in processed_reports
                ]
            },
            "pending": {"items": pending or []},
            "processed_combinations": {"items": processed_reports},
            "cache_ledger": {"items": ledger},
        }
        return self.snapshots.commit(manifest, artifacts)

    @staticmethod
    def _collision_statistics(
        reports: list[dict[str, Any]],
    ) -> dict[str, Any]:
        total_collisions = sum(
            item["cardinality"]["collision_count"] for item in reports
        )
        total_pairs = sum(
            item["cardinality"]["colliding_pair_count"] for item in reports
        )
        combos_with_collisions = [
            item["combo_id"]
            for item in reports
            if item["cardinality"]["collision_count"] > 0
        ]
        return {
            "adjudicated_combo_count": len(reports),
            "total_collision_count": total_collisions,
            "total_colliding_pair_count": total_pairs,
            "combinations_with_collisions": combos_with_collisions,
            "collision_combo_count": len(combos_with_collisions),
        }

    @staticmethod
    def _cache_lists(ledger: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        reused = [
            {
                "sequence": entry["sequence"],
                "combo_id": entry["combo_id"],
                "stage": entry["stage"],
                "cache_key": entry["cache_key"],
            }
            for entry in ledger
            if entry["status"] == "reused"
        ]
        recomputed = [
            {
                "sequence": entry["sequence"],
                "combo_id": entry["combo_id"],
                "stage": entry["stage"],
                "cache_key": entry["cache_key"],
            }
            for entry in ledger
            if entry["status"] == "recomputed"
        ]
        return {"reused": reused, "recomputed": recomputed}
