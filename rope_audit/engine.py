from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import Any

from .analytics import build_cross_scheme_comparison, build_scheme_statistics
from .errors import AuditError, failure_record
from .hashing import content_hash, prefixed_key
from .injections import InjectionHub
from .pipeline import (
    build_cell_errors,
    fixed_dot_product,
    fixed_normalize,
    fixed_position_encoding,
    fixed_rotation,
    position_vector_fixed,
    rotation_vector_fixed,
    run_float_pipeline,
)


class AuditEngine:
    def __init__(
        self,
        config: dict[str, Any],
        samples: list[dict[str, Any]],
        injection_hub: InjectionHub | None = None,
    ) -> None:
        self.config = deepcopy(config)
        self.samples = sorted(deepcopy(samples), key=lambda item: item["id"])
        self.sample_by_id = {sample["id"]: sample for sample in self.samples}
        self.hub = injection_hub or InjectionHub([])
        self.cache: dict[str, dict[str, Any]] = {}
        self.meta: dict[tuple[str, str], dict[str, Any]] = {}
        self.summary_meta: dict[str, dict[str, Any]] = {}
        self.ledger: list[dict[str, Any]] = []
        self.pending: list[str] = []
        self.consumed_total_points = 0
        self.last_batch_consumed_points = 0
        self.run_count = 0
        self.failures: dict[str, dict[str, Any]] = {}
        self.config_hash = content_hash(self.config, 20)

    def replace_state(self, config: dict[str, Any], samples: list[dict[str, Any]]) -> None:
        self.config = deepcopy(config)
        self.samples = sorted(deepcopy(samples), key=lambda item: item["id"])
        self.sample_by_id = {sample["id"]: sample for sample in self.samples}
        valid_ids = set(self.sample_by_id)
        self.pending = [sample_id for sample_id in self.pending if sample_id in valid_ids]
        self.config_hash = content_hash(self.config, 20)

    def scheme_names(self) -> list[str]:
        return [scheme["name"] for scheme in self.config["schemes"]]

    def get_scheme(self, name: str) -> dict[str, Any]:
        return next(scheme for scheme in self.config["schemes"] if scheme["name"] == name)

    def fingerprints(self, sample: dict[str, Any], scheme: dict[str, Any]) -> dict[str, str]:
        positions_input = {
            "dimension": self.config["dimension"],
            "query_position": sample["query_position"],
            "key_positions": sample["key_positions"],
        }
        vectors_input = {
            "dimension": self.config["dimension"],
            "query_vector": sample["query_vector"],
            "key_vectors": sample["key_vectors"],
        }
        return {
            "scheme_fp": content_hash(scheme),
            "quant_fp": content_hash(self.config["fixed_point"]),
            "sample_fp": content_hash(sample),
            "positions_fp": content_hash(positions_input),
            "vectors_fp": content_hash(vectors_input),
            "train_fp": content_hash(
                {
                    "config_train_length": self.config["train_length"],
                    "sample_train_length": sample["train_length"],
                }
            ),
        }

    def cache_keys(self, fingerprints: dict[str, str]) -> dict[str, str]:
        position_key = prefixed_key(
            "position",
            fingerprints["scheme_fp"],
            fingerprints["positions_fp"],
            fingerprints["quant_fp"],
        )
        rotation_key = prefixed_key(
            "rotation",
            position_key,
            fingerprints["vectors_fp"],
            fingerprints["quant_fp"],
        )
        dot_key = prefixed_key("dot", rotation_key)
        normalization_key = prefixed_key("norm", dot_key)
        return {
            "position": position_key,
            "rotation": rotation_key,
            "dot": dot_key,
            "normalization": normalization_key,
        }

    def _envelope(
        self,
        stage: str,
        key: str,
        sample: dict[str, Any] | None,
        scheme_name: str | None,
        fingerprints: dict[str, str] | None,
        input_fingerprint: str | None,
        value: Any,
    ) -> dict[str, Any]:
        return {
            "cache_key": key,
            "stage": stage,
            "config_hash": self.config_hash,
            "scheme_id": scheme_name,
            "sample_id": sample["id"] if sample else None,
            "scheme_fingerprint": (fingerprints or {}).get("scheme_fp"),
            "sample_fingerprint": (fingerprints or {}).get("sample_fp"),
            "input_fingerprint": input_fingerprint,
            "quant_fingerprint": (fingerprints or {}).get("quant_fp"),
            "references": {
                "sample_ids": [sample["id"]] if sample else [],
                "scheme_ids": [scheme_name] if scheme_name else [],
            },
            "value": value,
        }

    def _ledger_entry(
        self,
        run_id: int,
        stage: str,
        key: str,
        status: str,
        basis: str,
        sample_id: str | None = None,
        scheme_id: str | None = None,
    ) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "stage": stage,
            "cache_key": key,
            "status": status,
            "basis": basis,
            "sample_id": sample_id,
            "scheme_id": scheme_id,
            "references": {
                "sample_ids": [sample_id] if sample_id else [],
                "scheme_ids": [scheme_id] if scheme_id else [],
            },
        }

    def _stage_bases(
        self,
        previous: dict[str, Any] | None,
        fingerprints: dict[str, str],
        keys: dict[str, str],
    ) -> dict[str, str]:
        if previous is None:
            return {stage: "new_scheme_sample_pair" for stage in ("position", "rotation", "dot", "normalization")}
        previous_fp = previous["fingerprints"]
        previous_keys = previous["cache_keys"]

        position_reasons: list[str] = []
        if previous_fp["scheme_fp"] != fingerprints["scheme_fp"]:
            position_reasons.append("scheme_parameters_changed")
        if previous_fp["quant_fp"] != fingerprints["quant_fp"]:
            position_reasons.append("quantization_bits_or_level_changed")
        if previous_fp["positions_fp"] != fingerprints["positions_fp"]:
            position_reasons.append("query_or_key_positions_changed")
        position_basis = "+".join(position_reasons) if position_reasons else "position_dependencies_unchanged"

        if previous_keys["position"] != keys["position"]:
            rotation_basis = "upstream_position_key_changed"
        elif previous_fp["vectors_fp"] != fingerprints["vectors_fp"]:
            rotation_basis = "feature_vectors_changed"
        else:
            rotation_basis = "rotation_dependencies_unchanged"

        dot_basis = (
            "upstream_rotation_key_changed"
            if previous_keys["rotation"] != keys["rotation"]
            else "dot_dependencies_unchanged"
        )
        norm_basis = (
            "upstream_dot_key_changed"
            if previous_keys["dot"] != keys["dot"]
            else "normalization_dependencies_unchanged"
        )
        return {
            "position": position_basis,
            "rotation": rotation_basis,
            "dot": dot_basis,
            "normalization": norm_basis,
        }

    def injection_stages(self, sample_id: str, scheme_name: str) -> set[str]:
        return {
            injection["stage"]
            for injection in self.hub.injections
            if injection.get("sample_id") in (None, sample_id)
            and injection.get("scheme_id") in (None, scheme_name)
        }

    def cell_needs_work(self, sample: dict[str, Any], scheme_name: str) -> bool:
        scheme = self.get_scheme(scheme_name)
        fingerprints = self.fingerprints(sample, scheme)
        keys = self.cache_keys(fingerprints)
        meta = self.meta.get((scheme_name, sample["id"]))
        if meta is None:
            return True
        if meta["fingerprints"]["train_fp"] != fingerprints["train_fp"]:
            return True
        if any(key not in self.cache for key in keys.values()):
            return True
        if self.injection_stages(sample["id"], scheme_name):
            return True
        return False

    def invalid_sample_ids(self) -> list[str]:
        invalid: list[str] = []
        for sample_id in sorted(self.sample_by_id):
            sample = self.sample_by_id[sample_id]
            if any(self.cell_needs_work(sample, name) for name in self.scheme_names()):
                invalid.append(sample_id)
        return invalid

    def reconcile_cache_hits(self, run_id: int) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for sample_id in sorted(self.sample_by_id):
            sample = self.sample_by_id[sample_id]
            for scheme_name in self.scheme_names():
                if self.injection_stages(sample_id, scheme_name):
                    continue
                scheme = self.get_scheme(scheme_name)
                fingerprints = self.fingerprints(sample, scheme)
                keys = self.cache_keys(fingerprints)
                if not all(key in self.cache for key in keys.values()):
                    continue
                previous = self.meta.get((scheme_name, sample_id))
                bases = self._stage_bases(previous, fingerprints, keys)
                restored = previous is not None and previous.get("cache_keys") != keys
                for stage in ("position", "rotation", "dot", "normalization"):
                    basis = bases[stage]
                    if restored:
                        basis = f"{basis}+historical_content_key_reused"
                    entries.append(
                        self._ledger_entry(
                            run_id,
                            stage,
                            keys[stage],
                            "hit",
                            basis,
                            sample_id,
                            scheme_name,
                        )
                    )
                self.meta[(scheme_name, sample_id)] = {
                    "scheme_id": scheme_name,
                    "sample_id": sample_id,
                    "fingerprints": fingerprints,
                    "cache_keys": keys,
                    "last_run_id": run_id,
                    "reconciled_from_cache": True,
                }
                self.failures.pop(f"cell:{scheme_name}:{sample_id}", None)
        return entries

    def sample_cost(self, sample: dict[str, Any]) -> int:
        return int(self.config["dimension"]) * len(sample["key_positions"])

    def plan_batch(
        self,
        batch_budget: int | None = None,
        sample_quota: int | None = None,
    ) -> dict[str, Any]:
        budget = int(batch_budget if batch_budget is not None else self.config["budget"]["batch_point_budget"])
        quota = int(sample_quota if sample_quota is not None else self.config["budget"]["sample_quota_points"])
        priority_pending = [
            sample_id
            for sample_id in self.pending
            if sample_id in self.sample_by_id
            and any(self.cell_needs_work(self.sample_by_id[sample_id], name) for name in self.scheme_names())
        ]
        ordered: list[str] = []
        for sample_id in [*priority_pending, *self.invalid_sample_ids()]:
            if sample_id in self.sample_by_id and sample_id not in ordered:
                ordered.append(sample_id)

        accepted: list[dict[str, Any]] = []
        pending_tail: list[str] = []
        quota_failures: list[dict[str, Any]] = []
        remaining = budget
        stop_for_budget = False
        for sample_id in ordered:
            sample = self.sample_by_id[sample_id]
            cost = self.sample_cost(sample)
            if cost > quota:
                record = {
                    "code": "E_POINT_BUDGET_EXCEEDED",
                    "sample_id": sample_id,
                    "cost_points": cost,
                    "sample_quota_points": quota,
                    "message": f"sample cost {cost} exceeds per-sample quota {quota}",
                }
                quota_failures.append(record)
                self.failures[f"budget:{sample_id}"] = record
                continue
            if stop_for_budget or cost > remaining:
                stop_for_budget = True
                pending_tail.append(sample_id)
                marker = {
                    "code": "E_POINT_BUDGET_PENDING",
                    "sample_id": sample_id,
                    "cost_points": cost,
                    "remaining_points": remaining,
                    "batch_point_budget": budget,
                    "message": "batch budget exhausted; sample retained for next batch",
                }
                self.failures.pop(f"budget:{sample_id}", None)
                if not any(item["sample_id"] == sample_id and item["code"] == marker["code"] for item in quota_failures):
                    quota_failures.append(marker)
                continue
            accepted.append({"sample_id": sample_id, "cost_points": cost})
            remaining -= cost

        for sample_id in list(self.failures):
            if sample_id.startswith("budget:") and sample_id.split(":", 1)[1] not in {item["sample_id"] for item in quota_failures if item["code"] == "E_POINT_BUDGET_EXCEEDED"}:
                self.failures.pop(sample_id, None)
        self.pending = pending_tail
        consumed = budget - remaining
        self.last_batch_consumed_points = consumed
        self.consumed_total_points += consumed
        return {
            "ordered_sample_ids": ordered,
            "accepted": accepted,
            "pending": [item for item in quota_failures if item["code"] == "E_POINT_BUDGET_PENDING"],
            "quota_failures": [item for item in quota_failures if item["code"] == "E_POINT_BUDGET_EXCEEDED"],
            "remaining_points": remaining,
            "batch_point_budget": budget,
            "sample_quota_points": quota,
            "consumed_points": consumed,
        }

    def _translate_injection(
        self,
        exc: Exception,
        stage: str,
        sample_id: str,
        scheme_name: str,
    ) -> AuditError:
        active = self.hub.active_for(stage, sample_id, scheme_name)
        if isinstance(exc, AuditError):
            error = exc
            error.stage = stage
            error.sample_id = sample_id
            error.scheme_id = scheme_name
            if active:
                error.code = str(active[0].get("code", "E_INJECTED_FAULT"))
                injected_message = active[0].get("message", "injected deterministic input fault")
                error.message = f"{injected_message}: {error.message}"
            return error
        code = str(active[0].get("code", "E_INJECTED_FAULT")) if active else "E_SAMPLE_FAILED"
        return AuditError(
            code=code,
            message=f"{type(exc).__name__}: {exc}",
            stage=stage,
            sample_id=sample_id,
            scheme_id=scheme_name,
        )

    def execute_cell(self, sample_id: str, scheme_name: str, run_id: int) -> dict[str, Any]:
        sample = self.sample_by_id[sample_id]
        scheme = self.get_scheme(scheme_name)
        fingerprints = self.fingerprints(sample, scheme)
        keys = self.cache_keys(fingerprints)
        previous = self.meta.get((scheme_name, sample_id))
        bases = self._stage_bases(previous, fingerprints, keys)
        forced_stages = self.injection_stages(sample_id, scheme_name)
        entries: list[dict[str, Any]] = []
        envelopes: list[dict[str, Any]] = []

        def add_entry(stage: str, key: str, status: str, basis: str) -> None:
            entries.append(
                self._ledger_entry(run_id, stage, key, status, basis, sample_id, scheme_name)
            )

        try:
            position_key = keys["position"]
            if position_key in self.cache and "position" not in forced_stages:
                position = self.cache[position_key]["value"]
                add_entry("position", position_key, "hit", bases["position"])
            else:
                work_sample = sample
                if self.hub.active_for("position", sample_id, scheme_name):
                    work_sample = self.hub.corrupt_sample("position", sample, scheme_name)
                position = fixed_position_encoding(work_sample, scheme, self.config["fixed_point"], self.config["dimension"])
                envelopes.append(
                    self._envelope("position", position_key, sample, scheme_name, fingerprints, fingerprints["positions_fp"], position)
                )
                add_entry("position", position_key, "miss", bases["position"])

            rotation_key = keys["rotation"]
            if rotation_key in self.cache and "rotation" not in forced_stages:
                rotation = self.cache[rotation_key]["value"]
                add_entry("rotation", rotation_key, "hit", bases["rotation"])
            else:
                work_sample = sample
                if self.hub.active_for("rotation", sample_id, scheme_name):
                    work_sample = self.hub.corrupt_sample("rotation", sample, scheme_name)
                rotation = fixed_rotation(work_sample, position, self.config["fixed_point"])
                envelopes.append(
                    self._envelope("rotation", rotation_key, sample, scheme_name, fingerprints, fingerprints["vectors_fp"], rotation)
                )
                add_entry("rotation", rotation_key, "miss", bases["rotation"])

            dot_key = keys["dot"]
            if dot_key in self.cache and "dot" not in forced_stages:
                dot = self.cache[dot_key]["value"]
                add_entry("dot", dot_key, "hit", bases["dot"])
            else:
                dot_rotation = rotation
                dot_injections = self.hub.active_for("dot", sample_id, scheme_name)
                if dot_injections:
                    for injection in dot_injections:
                        self.hub.hit(injection)
                    dot_rotation = deepcopy(rotation)
                    if any(item["fault_type"] == "wrong_shape" for item in dot_injections):
                        dot_rotation["query"]["output_int"].pop()
                dot = fixed_dot_product(dot_rotation)
                envelopes.append(self._envelope("dot", dot_key, sample, scheme_name, fingerprints, None, dot))
                add_entry("dot", dot_key, "miss", bases["dot"])

            norm_key = keys["normalization"]
            if norm_key in self.cache and "normalization" not in forced_stages:
                normalization = self.cache[norm_key]["value"]
                add_entry("normalization", norm_key, "hit", bases["normalization"])
            else:
                norm_injections = self.hub.active_for("normalization", sample_id, scheme_name)
                force_zero_sum = False
                if norm_injections:
                    for injection in norm_injections:
                        self.hub.hit(injection)
                        if injection["fault_type"] == "zero_normalization_sum":
                            force_zero_sum = True
                normalization = fixed_normalize(dot, force_zero_sum=force_zero_sum)
                envelopes.append(self._envelope("normalization", norm_key, sample, scheme_name, fingerprints, None, normalization))
                add_entry("normalization", norm_key, "miss", bases["normalization"])

            fixed_view = {
                "position": position,
                "rotation": rotation,
                "dot": dot,
                "normalization": normalization,
                "position_vector": position_vector_fixed(position),
                "rotation_vector": rotation_vector_fixed(rotation),
            }
            truth = run_float_pipeline(sample, scheme, self.config["dimension"])
            errors = build_cell_errors(fixed_view, truth, sample, self.config["tolerances"])
            cell = {
                "sample_id": sample_id,
                "scheme_id": scheme_name,
                "status": "ok",
                "query_position": sample["query_position"],
                "train_length": sample["train_length"],
                "key_positions": sample["key_positions"],
                "scores_fixed": dot["scores"],
                "scores_truth": truth["scores"],
                "probabilities_fixed": normalization["probabilities"],
                "probabilities_truth": truth["probabilities"],
                "errors": errors,
                "cache_keys": keys,
                "fingerprints": fingerprints,
            }
            return {
                "ok": True,
                "cell_key": (scheme_name, sample_id),
                "cell": cell,
                "entries": entries,
                "envelopes": envelopes,
                "fingerprints": fingerprints,
                "keys": keys,
            }
        except Exception as exc:  # noqa: BLE001 - one sample must not stop batch
            stage = "position"
            completed = {entry["stage"] for entry in entries}
            for candidate in ("position", "rotation", "dot", "normalization"):
                if candidate not in completed:
                    stage = candidate
                    break
            else:
                stage = "normalization"
            error = self._translate_injection(exc, stage, sample_id, scheme_name)
            record = failure_record(error, sample_id=sample_id, scheme_id=scheme_name)
            return {
                "ok": False,
                "cell_key": (scheme_name, sample_id),
                "error": record,
                "entries": entries,
                "envelopes": envelopes,
                "fingerprints": fingerprints,
                "keys": keys,
            }

    def run_batch(
        self,
        batch_budget: int | None = None,
        sample_quota: int | None = None,
        workers: int | None = None,
        sample_quota_points: int | None = None,
    ) -> dict[str, Any]:
        self.run_count += 1
        run_id = self.run_count
        reconciled_entries = self.reconcile_cache_hits(run_id)
        quota_value = sample_quota_points if sample_quota_points is not None else sample_quota
        plan = self.plan_batch(batch_budget, quota_value)
        worker_count = int(workers if workers is not None else self.config["concurrency"]["workers"])
        tasks: list[tuple[str, str]] = []
        for item in plan["accepted"]:
            accepted_sample = self.sample_by_id[item["sample_id"]]
            for scheme_name in self.scheme_names():
                if self.cell_needs_work(accepted_sample, scheme_name):
                    tasks.append((item["sample_id"], scheme_name))

        results: list[dict[str, Any]] = []
        if tasks:
            if worker_count <= 1:
                results = [self.execute_cell(sample_id, scheme_name, run_id) for sample_id, scheme_name in tasks]
            else:
                with ThreadPoolExecutor(max_workers=min(worker_count, len(tasks))) as executor:
                    results = list(executor.map(lambda args: self.execute_cell(*args, run_id=run_id), tasks))

        run_ledger: list[dict[str, Any]] = list(reconciled_entries)
        for result in results:
            for envelope in result["envelopes"]:
                self.cache[envelope["cache_key"]] = envelope
            for entry in result["entries"]:
                run_ledger.append(entry)
            scheme_name, sample_id = result["cell_key"]
            self.meta[(scheme_name, sample_id)] = {
                "scheme_id": scheme_name,
                "sample_id": sample_id,
                "fingerprints": result["fingerprints"],
                "cache_keys": result["keys"],
                "last_run_id": run_id,
            }
            failure_key = f"cell:{scheme_name}:{sample_id}"
            if result["ok"]:
                self.failures.pop(failure_key, None)
            else:
                self.failures[failure_key] = result["error"]

        summary_ledger = self._commit_summaries(run_id, run_ledger)
        run_ledger.extend(summary_ledger)
        for index, entry in enumerate(run_ledger):
            entry["seq"] = len(self.ledger) + index + 1
        self.ledger.extend(run_ledger)

        report = self.build_report()
        report["budget"]["plan"] = plan
        report["run_ledger"] = run_ledger
        report["run_id"] = run_id
        self.last_report = report
        return report

    def _reconstruct_view(self, sample_id: str, scheme_name: str) -> dict[str, Any] | None:
        meta = self.meta.get((scheme_name, sample_id))
        if meta is None:
            return None
        keys = meta["cache_keys"]
        if any(key not in self.cache for key in keys.values()):
            return None
        sample = self.sample_by_id.get(sample_id)
        if sample is None:
            return None
        scheme = self.get_scheme(scheme_name)
        position = self.cache[keys["position"]]["value"]
        rotation = self.cache[keys["rotation"]]["value"]
        dot = self.cache[keys["dot"]]["value"]
        normalization = self.cache[keys["normalization"]]["value"]
        fixed_view = {
            "position": position,
            "rotation": rotation,
            "dot": dot,
            "normalization": normalization,
            "position_vector": position_vector_fixed(position),
            "rotation_vector": rotation_vector_fixed(rotation),
        }
        truth = run_float_pipeline(sample, scheme, self.config["dimension"])
        errors = build_cell_errors(fixed_view, truth, sample, self.config["tolerances"])
        return {
            "sample_id": sample_id,
            "scheme_id": scheme_name,
            "errors": errors,
            "fixed": fixed_view,
            "truth": truth,
            "cache_keys": keys,
            "fingerprints": meta["fingerprints"],
        }

    def _summary_key(self, scheme_name: str) -> tuple[str, str, dict[str, Any]]:
        scheme = self.get_scheme(scheme_name)
        item_keys: list[dict[str, Any]] = []
        for sample_id in sorted(self.sample_by_id):
            meta = self.meta.get((scheme_name, sample_id))
            if meta and all(key in self.cache for key in meta["cache_keys"].values()):
                item_keys.append(
                    {
                        "sample_id": sample_id,
                        "train_fp": meta["fingerprints"]["train_fp"],
                        "normalization_key": meta["cache_keys"]["normalization"],
                    }
                )
        source = {
            "scheme_fp": content_hash(scheme),
            "config_train_length": self.config["train_length"],
            "target_length": self.config["target_length"],
            "items": item_keys,
        }
        return prefixed_key("summary", content_hash(source)), content_hash(source), source

    def _commit_summaries(self, run_id: int, run_ledger: list[dict[str, Any]]) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for scheme_name in self.scheme_names():
            key, input_fp, source = self._summary_key(scheme_name)
            previous = self.summary_meta.get(scheme_name)
            basis = "summary_dependencies_unchanged"
            if previous is None:
                basis = "new_scheme_summary"
            elif previous["cache_key"] != key:
                changed = []
                if previous.get("input_fingerprint") != input_fp:
                    changed.append("sample_set_train_length_or_normalization_keys_changed")
                basis = "+".join(changed) or "summary_key_changed"
            if key in self.cache:
                status = "hit"
            else:
                views = [
                    view
                    for sample_id in sorted(self.sample_by_id)
                    if (view := self._reconstruct_view(sample_id, scheme_name)) is not None
                ]
                stats = build_scheme_statistics(scheme_name, views)
                self.cache[key] = self._envelope(
                    "summary",
                    key,
                    None,
                    scheme_name,
                    None,
                    input_fp,
                    {"statistics": stats, "source": source},
                )
                self.cache[key]["references"]["sample_ids"] = [view["sample_id"] for view in views]
                status = "miss"
            self.summary_meta[scheme_name] = {"cache_key": key, "input_fingerprint": input_fp}
            entries.append(self._ledger_entry(run_id, "summary", key, status, basis, None, scheme_name))
        return entries

    def build_report(self) -> dict[str, Any]:
        scheme_names = self.scheme_names()
        views_by_scheme: dict[str, list[dict[str, Any]]] = {name: [] for name in scheme_names}
        sample_results: list[dict[str, Any]] = []
        cache_index: list[dict[str, Any]] = []

        for sample_id in sorted(self.sample_by_id):
            sample_schemes: list[dict[str, Any]] = []
            for scheme_name in scheme_names:
                view = self._reconstruct_view(sample_id, scheme_name)
                if view is None:
                    failure = self.failures.get(f"cell:{scheme_name}:{sample_id}")
                    sample_schemes.append(
                        {
                            "scheme_id": scheme_name,
                            "status": "failed" if failure else "pending_or_missing",
                            "failure": failure,
                        }
                    )
                    continue
                views_by_scheme[scheme_name].append(view)
                errors = view["errors"]
                sample_schemes.append(
                    {
                        "scheme_id": scheme_name,
                        "status": "ok",
                        "scores_fixed": [round(float(v), 12) for v in view["fixed"]["dot"]["scores"]],
                        "scores_truth": [round(float(v), 12) for v in view["truth"]["scores"]],
                        "probabilities_fixed": [round(float(v), 12) for v in view["fixed"]["normalization"]["probabilities"]],
                        "probabilities_truth": [round(float(v), 12) for v in view["truth"]["probabilities"]],
                        "stage_errors": {
                            stage: {
                                key: value
                                for key, value in errors["stages"][stage].items()
                                if key not in {"worst_abs", "worst_rel"}
                            }
                            for stage in ("position", "rotation", "dot", "normalization")
                        },
                        "first_breach": errors["stages"].get("first_breach"),
                        "key_attribution": errors["keys"],
                        "cache_keys": view["cache_keys"],
                        "fingerprints": view["fingerprints"],
                    }
                )
                for stage in ("position", "rotation", "dot", "normalization"):
                    key = view["cache_keys"][stage]
                    envelope = self.cache[key]
                    cache_index.append(
                        {
                            "stage": stage,
                            "cache_key": key,
                            "sample_id": sample_id,
                            "scheme_id": scheme_name,
                            "references": envelope["references"],
                        }
                    )
            sample = self.sample_by_id[sample_id]
            sample_results.append(
                {
                    "sample_id": sample_id,
                    "query_position": sample["query_position"],
                    "train_length": sample["train_length"],
                    "key_positions": sample["key_positions"],
                    "cost_points": self.sample_cost(sample),
                    "schemes": sample_schemes,
                }
            )

        statistics: dict[str, Any] = {}
        for scheme_name in scheme_names:
            summary_key = self.summary_meta.get(scheme_name, {}).get("cache_key")
            if summary_key and summary_key in self.cache:
                statistics[scheme_name] = self.cache[summary_key]["value"]["statistics"]
                cache_index.append(
                    {
                        "stage": "summary",
                        "cache_key": summary_key,
                        "sample_id": None,
                        "scheme_id": scheme_name,
                        "references": self.cache[summary_key]["references"],
                    }
                )
            else:
                statistics[scheme_name] = build_scheme_statistics(scheme_name, views_by_scheme[scheme_name])

        cross_positions = build_cross_scheme_comparison(scheme_names, views_by_scheme)
        cache_index.sort(key=lambda item: (item["scheme_id"] or "", item["sample_id"] or "", item["stage"], item["cache_key"]))
        failures = [self.failures[key] for key in sorted(self.failures)]
        return {
            "config_hash": self.config_hash,
            "active_scheme": self.config["active_scheme"],
            "train_length": self.config["train_length"],
            "target_length": self.config["target_length"],
            "scheme_names": scheme_names,
            "sample_ids": sorted(self.sample_by_id),
            "samples": sample_results,
            "statistics": statistics,
            "cross_scheme_extrapolation_positions": cross_positions,
            "cache_index": cache_index,
            "failures": failures,
            "budget": {
                "batch_point_budget": self.config["budget"]["batch_point_budget"],
                "sample_quota_points": self.config["budget"]["sample_quota_points"],
                "consumed_total_points": self.consumed_total_points,
                "last_batch_consumed_points": self.last_batch_consumed_points,
                "pending_sample_ids": list(self.pending),
            },
            "injection_hits": list(self.hub.hits),
        }

    def comparison_payload(self, report: dict[str, Any] | None = None) -> dict[str, Any]:
        report = report or self.last_report or self.build_report()
        return {
            "config_hash": report["config_hash"],
            "scheme_names": report["scheme_names"],
            "sample_ids": report["sample_ids"],
            "samples": report["samples"],
            "statistics": report["statistics"],
            "cross_scheme_extrapolation_positions": report["cross_scheme_extrapolation_positions"],
            "cache_index": [
                {
                    "stage": item["stage"],
                    "cache_key": item["cache_key"],
                    "sample_id": item["sample_id"],
                    "scheme_id": item["scheme_id"],
                }
                for item in report["cache_index"]
            ],
        }

    def export_state(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "samples": self.samples,
            "cache": self.cache,
            "meta": [
                {"key": [scheme_name, sample_id], **value}
                for (scheme_name, sample_id), value in sorted(self.meta.items())
            ],
            "summary_meta": self.summary_meta,
            "ledger": self.ledger,
            "pending": self.pending,
            "consumed_total_points": self.consumed_total_points,
            "last_batch_consumed_points": self.last_batch_consumed_points,
            "run_count": self.run_count,
            "failures": self.failures,
            "config_hash": self.config_hash,
        }

    def import_state(self, state: dict[str, Any]) -> None:
        self.config = state["config"]
        self.samples = state["samples"]
        self.sample_by_id = {sample["id"]: sample for sample in self.samples}
        self.cache = state.get("cache", {})
        self.meta = {
            (item["key"][0], item["key"][1]): {k: v for k, v in item.items() if k != "key"}
            for item in state.get("meta", [])
        }
        self.summary_meta = state.get("summary_meta", {})
        self.ledger = state.get("ledger", [])
        self.pending = state.get("pending", [])
        self.consumed_total_points = int(state.get("consumed_total_points", 0))
        self.last_batch_consumed_points = int(state.get("last_batch_consumed_points", 0))
        self.run_count = int(state.get("run_count", 0))
        self.failures = state.get("failures", {})
        self.config_hash = state.get("config_hash", content_hash(self.config, 20))
