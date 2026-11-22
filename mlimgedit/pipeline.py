"""End-to-end deterministic incremental pipeline orchestration.

Stages: validation -> features -> scoring/decision -> metrics. Every stage
artifact is content-addressed and cached; cache keys derive from image content,
extraction parameters, classifier hyper-parameters and a scale-tier identifier.
The same logical state therefore produces identical artifacts whether reached
by incremental edits or by a single full rebuild from an empty workspace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .budget import BudgetDeduction, PendingItem, make_feature_tasks, schedule_budget
from .cachekeys import (
    DECISION,
    FEATURES,
    METRICS,
    SCORING,
    VALIDATION,
    AuditEntry,
    decision_artifact_key,
    feature_artifact_key,
    metric_artifact_key,
    model_artifact_key,
    sort_audit,
    validation_artifact_key,
)
from .classifiers import CLASSIFIER_ORDER, train_label_model
from .concurrency import OrderedTask, run_tasks
from .errors import E_STAGE, StageError
from .features import (
    build_feature_artifact,
    dequantize_matrix,
    feature_point_cost,
)
from .fixedpoint import FixedPointQuantizer
from .hashing import fingerprint_json, fingerprint_strings, scale_tier
from .metrics import evaluate_label, summarize
from .snapshot import SnapshotStore, StageState
from .validation import validate_directory


@dataclass
class RunArtifacts:
    validation: dict[str, dict] = field(default_factory=dict)
    features: dict[str, dict] = field(default_factory=dict)
    models: dict[tuple[str, str], dict] = field(default_factory=dict)
    decisions: dict[tuple[str, str], dict] = field(default_factory=dict)
    metrics: dict[tuple[str, str], dict] = field(default_factory=dict)
    ground_truth: dict[str, tuple[str, ...]] = field(default_factory=dict)
    sample_fingerprints: dict[str, str] = field(default_factory=dict)
    failures: list[dict] = field(default_factory=list)
    audit: list[AuditEntry] = field(default_factory=list)
    deductions: list[BudgetDeduction] = field(default_factory=list)
    pending: list[PendingItem] = field(default_factory=list)


class Pipeline:
    def __init__(self, store: SnapshotStore, injections=None, interruption=None):
        self.store = store
        self.injections = injections or {}
        # interruption is an optional callable(batch_index, stage) raising
        # KeyboardInterrupt to simulate a mid-run signal.
        self.interruption = interruption

    # ------------------------------------------------------------------ helpers
    def _quantizer(self, config) -> FixedPointQuantizer:
        return FixedPointQuantizer(
            quant_level=config.quant_level,
            frac_bits=config.frac_bits,
            rounding=config.rounding,
        )

    def _load_existing(self, manifest: dict):
        artifacts = RunArtifacts()
        stages = manifest.get("stages", {})
        for name, key in stages.get(VALIDATION, {}).get("keys", {}).items():
            payload = self.store.get_artifact(key)
            artifacts.validation[name] = payload
            artifacts.sample_fingerprints[name] = payload["image_fingerprint"]
            artifacts.ground_truth[name] = tuple(payload["labels"])
        for name, key in stages.get(FEATURES, {}).get("keys", {}).items():
            artifacts.features[name] = self.store.get_artifact(key)
        for (stage_name, holder) in ((SCORING, artifacts.models), (METRICS, artifacts.metrics)):
            for logical, artifact_id in stages.get(stage_name, {}).get("keys", {}).items():
                classifier, label = logical.split("\u0001", 1)
                payload = self.store.get_artifact(artifact_id)
                # Artifact payloads persist their logical cache key; fall back
                # to the content id only for legacy/defensive reads.
                payload.setdefault("cache_key", artifact_id)
                holder[(classifier, label)] = payload
        self._restore_feature_cache_keys(artifacts, stages)
        # Decisions are embedded in model artifacts but also tracked separately
        # via their own stage keys for threshold-only invalidation.
        artifacts.pending = [PendingItem(**item) for item in manifest.get("pending", [])]
        artifacts.deductions = [
            BudgetDeduction(**item) for item in manifest.get("deductions", [])
        ]
        return artifacts

    def _restore_feature_cache_keys(self, artifacts: RunArtifacts, stages: dict):
        for name, artifact_id in stages.get(FEATURES, {}).get("keys", {}).items():
            payload = artifacts.features.get(name)
            if payload is not None and "cache_key" not in payload:
                payload["cache_key"] = artifact_id

    # ------------------------------------------------------------- stage 1
    def run_validation(self, config, sample_dir) -> RunArtifacts:
        result = validate_directory(sample_dir, config, self.injections)
        artifacts = RunArtifacts()
        artifacts.failures = [failure.to_dict() for failure in result.failures]
        for sample in result.valid:
            key = validation_artifact_key(
                sample.name,
                sample.image_fingerprint,
                sample.annotation_fingerprint,
                config.labels,
            )
            payload = {
                "sample": sample.name,
                "image_fingerprint": sample.image_fingerprint,
                "annotation_fingerprint": sample.annotation_fingerprint,
                "width": sample.width,
                "height": sample.height,
                "labels": list(sample.labels),
                "boxes": sample.boxes,
            }
            artifacts.validation[sample.name] = payload
            artifacts.sample_fingerprints[sample.name] = sample.image_fingerprint
            artifacts.ground_truth[sample.name] = sample.labels
            artifacts.audit.append(
                AuditEntry(
                    stage=VALIDATION,
                    name=sample.name,
                    action="recompute",
                    reason="validation artifact materialized for current input state",
                    cache_key=key,
                    sort_key=f"{VALIDATION}|{sample.name}",
                )
            )
        return artifacts

    # ------------------------------------------------------------- stage 2
    def run_features(self, config, current: RunArtifacts, previous: RunArtifacts, sample_dir=None):
        cost = feature_point_cost(config)
        valid_names = sorted(current.validation)

        # Reuse features whose content+parameter key already exists. Reused
        # artifacts do not consume any feature-point budget.
        reusable: dict[str, dict] = {}
        for name, payload in previous.features.items():
            if name not in current.validation:
                continue
            key = feature_artifact_key(
                name, current.sample_fingerprints[name], config
            )
            if payload.get("cache_key") == key:
                reusable[name] = payload

        # Only samples without a reusable artifact are scheduled. Carried
        # pending samples are always placed ahead of new work.
        required_names = [name for name in valid_names if name not in reusable]
        carried_names = [
            item.sample
            for item in previous.pending
            if item.sample in required_names
        ]
        fresh_names = [name for name in required_names if name not in carried_names]
        carried = make_feature_tasks(carried_names, cost)
        fresh = make_feature_tasks(fresh_names, cost)
        schedule = schedule_budget(fresh, config.feature_point_budget, carried)
        current.deductions = schedule.deductions
        current.pending = schedule.pending_records
        needed = list(schedule.ready)

        sample_by_name = {
            name: _ValidSampleView(current.validation[name], sample_dir=sample_dir) for name in valid_names
        }
        quantizer = self._quantizer(config)

        def build(task):
            view = sample_by_name[task.sample]
            return build_feature_artifact(view, config, quantizer).to_dict()

        tasks = [
            OrderedTask(
                key=task.sort_key,
                payload=task,
                func=build,
                stage=FEATURES,
                sample=task.sample,
            )
            for task in needed
        ]
        for name, payload in sorted(reusable.items()):
            current.features[name] = payload
            current.audit.append(
                AuditEntry(
                    stage=FEATURES,
                    name=name,
                    action="reuse",
                    reason="image content and extraction parameters unchanged",
                    cache_key=payload["cache_key"],
                    sort_key=f"{FEATURES}|{name}",
                )
            )
        outputs = run_tasks(tasks, config.threads)
        for task, output in outputs:
            if isinstance(output, Exception):
                current.failures.append(
                    {
                        "sample": task.sample,
                        "stage": FEATURES,
                        "code": E_STAGE,
                        "message": str(output),
                        "injection_id": "",
                    }
                )
                continue
            name = task.sample
            key = feature_artifact_key(name, current.sample_fingerprints[name], config)
            output["cache_key"] = key
            current.features[name] = output
            current.audit.append(
                AuditEntry(
                    stage=FEATURES,
                    name=name,
                    action="recompute",
                    reason="new/changed image or extraction parameters",
                    cache_key=key,
                    sort_key=task.key,
                )
            )
        # Samples not scheduled because the budget was exhausted stay pending.
        return schedule

    # ------------------------------------------------------------- stage 3 + 4
    def run_scoring_and_metrics(self, config, current: RunArtifacts, previous: RunArtifacts):
        feature_names = sorted(current.features)
        if not feature_names:
            return
        quantizer = self._quantizer(config)
        # Reconstruct numpy inputs from cached fixed-point feature artifacts.
        feature_artifacts = [
            _FeatureView(current.features[name]) for name in feature_names
        ]
        matrix = dequantize_matrix(feature_artifacts, quantizer)
        tier = scale_tier(len(feature_names))

        # One deterministic task per (classifier, label); label order is stable.
        task_specs = [
            (classifier, label)
            for classifier in CLASSIFIER_ORDER
            for label in config.labels
        ]

        def build_model(spec):
            classifier, label = spec
            targets = np_from_truth(current.ground_truth, label, feature_names)
            return train_label_model(
                classifier,
                label,
                feature_names,
                matrix,
                targets,
                config,
                quantizer,
                tier,
            )

        tasks = [
            OrderedTask(
                key=f"{SCORING}|{classifier}|{label}",
                payload=(classifier, label),
                func=build_model,
                stage=SCORING,
                classifier=classifier,
                sample="",
            )
            for classifier, label in task_specs
        ]
        outputs = run_tasks(tasks, config.threads)
        for task, output in outputs:
            classifier, label = task.payload
            if isinstance(output, Exception):
                current.failures.append(
                    {
                        "sample": "",
                        "stage": SCORING,
                        "code": E_STAGE,
                        "message": f"{classifier}/{label}: {output}",
                        "injection_id": "",
                    }
                )
                continue
            model = output
            key = model_artifact_key(
                classifier,
                label,
                list(model.referenced_samples),
                current.sample_fingerprints,
                config,
            )
            threshold = config.effective_threshold(label)
            decision_key = decision_artifact_key(
                key, label, threshold, config.quant_level, config.frac_bits
            )
            previous_model = previous.models.get((classifier, label))
            model_dict = model.to_dict()
            model_dict["cache_key"] = key
            model_dict["decision_cache_key"] = decision_key
            decision_payload = {
                "classifier": classifier,
                "label": label,
                "threshold": threshold,
                "tie_rule": "positive_on_equal",
                "decisions": [
                    {
                        "sample": score["sample"],
                        "probability_int": score["probability_int"],
                        "threshold_int": score["threshold_int"],
                        "decision": score["decision"],
                        "boundary": score["boundary"],
                    }
                    for score in model_dict["scores"]
                ],
            }
            metric = evaluate_label(model, current.ground_truth)
            metric_key = metric_artifact_key(decision_key, classifier, label)
            metric_dict = metric.to_dict()
            metric_dict["cache_key"] = metric_key

            reused_model = (
                previous_model is not None
                and previous_model.get("cache_key") == key
            )
            if reused_model:
                current.audit.append(
                    AuditEntry(
                        stage=SCORING,
                        name=f"{classifier}/{label}",
                        action="reuse",
                        reason="hyper-parameters, referenced samples and scale tier unchanged",
                        cache_key=key,
                        sort_key=task.key,
                        referenced_samples=model.referenced_samples,
                    )
                )
            else:
                current.audit.append(
                    AuditEntry(
                        stage=SCORING,
                        name=f"{classifier}/{label}",
                        action="recompute",
                        reason="label, hyper-parameters, referenced samples or scale tier changed",
                        cache_key=key,
                        sort_key=task.key,
                        referenced_samples=model.referenced_samples,
                    )
                )
            previous_decision_key = (
                previous_model.get("decision_cache_key") if previous_model else None
            )
            decision_action = "reuse" if previous_decision_key == decision_key else "recompute"
            decision_reason = (
                "threshold and quantization unchanged"
                if decision_action == "reuse"
                else "threshold or quantization changed"
            )
            current.audit.append(
                AuditEntry(
                    stage=DECISION,
                    name=f"{classifier}/{label}",
                    action=decision_action,
                    reason=decision_reason,
                    cache_key=decision_key,
                    sort_key=f"{DECISION}|{classifier}|{label}",
                    referenced_samples=model.referenced_samples,
                )
            )
            previous_metric = previous.metrics.get((classifier, label))
            metric_action = (
                "reuse"
                if previous_metric is not None
                and previous_metric.get("cache_key") == metric_key
                else "recompute"
            )
            current.audit.append(
                AuditEntry(
                    stage=METRICS,
                    name=f"{classifier}/{label}",
                    action=metric_action,
                    reason=(
                        "decision key unchanged"
                        if metric_action == "reuse"
                        else "decision or referenced labels changed"
                    ),
                    cache_key=metric_key,
                    sort_key=f"{METRICS}|{classifier}|{label}",
                    referenced_samples=model.referenced_samples,
                )
            )
            current.models[(classifier, label)] = model_dict
            current.decisions[(classifier, label)] = decision_payload
            current.metrics[(classifier, label)] = metric_dict

    # ------------------------------------------------------------------ run batch
    def run_batch(self, config, sample_dir, batch_index: int) -> dict:
        sample_dir = Path(sample_dir)
        manifest = self.store.load_manifest()
        previous = self._load_existing(manifest)

        if self.interruption is not None:
            self.interruption(batch_index, VALIDATION)
        current = self.run_validation(config, sample_dir)
        if self.interruption is not None:
            self.interruption(batch_index, FEATURES)
        schedule = self.run_features(config, current, previous, sample_dir=sample_dir)

        # Scoring waits for the complete feature matrix; budget-pending batches
        # therefore retain pending work and skip downstream stages until drained.
        blocked = bool(schedule.pending)
        if not blocked:
            if self.interruption is not None:
                self.interruption(batch_index, SCORING)
            self.run_scoring_and_metrics(config, current, previous)
            if self.interruption is not None:
                self.interruption(batch_index, METRICS)

        current.audit = sort_audit(current.audit)
        current.failures = sorted(
            current.failures,
            key=lambda item: (item["sample"], item["stage"], item["code"], item.get("injection_id", "")),
        )
        report = self._build_report(
            config, current, batch_index, blocked=blocked, manifest=manifest
        )
        self._commit(config, current, batch_index, manifest)
        return report

    def should_commit(self, config, batch_index: int, total_batches: int | None = None) -> bool:
        """Snapshot cadence derived from ``runtime.snapshot_every``.

        Commits land on interval boundaries; the final batch is always forced
        so the manifest cursor reaches the plan length.
        """
        interval = max(1, int(config.snapshot_every))
        if total_batches is not None and batch_index >= total_batches - 1:
            return True
        return (batch_index + 1) % interval == 0

    # ------------------------------------------------------------------ commit
    def _commit(self, config, current: RunArtifacts, batch_index, manifest):
        stages = {
            VALIDATION: StageState(version="", keys={}),
            FEATURES: StageState(version="", keys={}),
            SCORING: StageState(version="", keys={}),
            DECISION: StageState(version="", keys={}),
            METRICS: StageState(version="", keys={}),
        }
        for name, payload in sorted(current.validation.items()):
            key = validation_artifact_key(
                name,
                payload["image_fingerprint"],
                payload["annotation_fingerprint"],
                config.labels,
            )
            payload = dict(payload)
            payload["cache_key"] = key
            artifact_id = self.store.put_artifact(payload)
            stages[VALIDATION].keys[name] = artifact_id
        for name, payload in sorted(current.features.items()):
            artifact_id = self.store.put_artifact(payload)
            stages[FEATURES].keys[name] = artifact_id
        for (classifier, label), payload in sorted(current.models.items()):
            logical = f"{classifier}\u0001{label}"
            stages[SCORING].keys[logical] = self.store.put_artifact(payload)
        for (classifier, label), payload in sorted(current.decisions.items()):
            logical = f"{classifier}\u0001{label}"
            stages[DECISION].keys[logical] = self.store.put_artifact(payload)
        for (classifier, label), payload in sorted(current.metrics.items()):
            logical = f"{classifier}\u0001{label}"
            stages[METRICS].keys[logical] = self.store.put_artifact(payload)
        for state in stages.values():
            state.version = fingerprint_json(state.keys)
        self.store.commit(
            manifest,
            config_hash=fingerprint_json(config.to_dict()),
            labels=tuple(config.labels),
            batch_cursor=batch_index + 1,
            stages=stages,
            pending=[item.to_dict() for item in current.pending],
            deduction_cursor=manifest.get("deduction_cursor", 0)
            + len(current.deductions),
        )

    # ------------------------------------------------------------------ report
    def _build_report(self, config, current: RunArtifacts, batch_index, *, blocked, manifest):
        metric_objects = [
            _MetricView(payload) for payload in sorted(
                current.metrics.values(),
                key=lambda item: (item["classifier"], item["label"]),
            )
        ]
        summary = summarize(metric_objects) if metric_objects else {"per_label": {}, "aggregate": {}}
        per_sample = self._per_sample_predictions(config, current)
        reuse = [entry.to_dict() for entry in current.audit if entry.action == "reuse"]
        recompute = [entry.to_dict() for entry in current.audit if entry.action == "recompute"]
        boundary = []
        for payload in current.models.values():
            for score in payload["scores"]:
                if score["boundary"]:
                    boundary.append(
                        {
                            "classifier": payload["classifier"],
                            "label": payload["label"],
                            "sample": score["sample"],
                            "tie_rule": score["tie_rule"],
                        }
                    )
        return {
            "batch_index": batch_index,
            "global_version": int(manifest.get("global_version", 0)) + 1,
            "config_hash": fingerprint_json(config.to_dict()),
            "label_set_fingerprint": fingerprint_strings(config.labels),
            "feature_mode": config.feature_mode,
            "quantization": {
                "quant_level": config.quant_level,
                "frac_bits": config.frac_bits,
                "rounding": config.rounding,
            },
            "samples": sorted(current.validation),
            "per_sample_predictions": per_sample,
            "metrics_summary": summary,
            "boundary_samples": sorted(
                boundary, key=lambda item: (item["classifier"], item["label"], item["sample"])
            ),
            "decision_records": self._decision_records(current),
            "reuse": reuse,
            "recompute": recompute,
            "audit": [entry.to_dict() for entry in current.audit],
            "counts": {
                "reuse": len(reuse),
                "recompute": len(recompute),
                "samples": len(current.validation),
                "failures": len(current.failures),
                "pending": len(current.pending),
            },
            "pending": [item.to_dict() for item in current.pending],
            "budget": {
                "initial_budget": config.feature_point_budget,
                "deductions": [item.to_dict() for item in current.deductions],
                "remaining_budget": (
                    config.feature_point_budget
                    - sum(item.cost for item in current.deductions)
                ),
            },
            "failures": current.failures,
            "downstream_blocked_by_budget": blocked,
        }

    def _per_sample_predictions(self, config, current: RunArtifacts):
        output = []
        for sample in sorted(current.validation):
            truth = list(current.ground_truth.get(sample, ()))
            entry = {
                "sample": sample,
                "ground_truth": truth,
                "classifiers": {},
            }
            for classifier in CLASSIFIER_ORDER:
                labels_payload = {}
                predicted = []
                for label in config.labels:
                    model = current.models.get((classifier, label))
                    if model is None:
                        continue
                    score = next(item for item in model["scores"] if item["sample"] == sample)
                    labels_payload[label] = {
                        "raw_score": score["raw_score"],
                        "probability": score["probability"],
                        "probability_int": score["probability_int"],
                        "threshold_int": score["threshold_int"],
                        "decision": score["decision"],
                        "boundary": score["boundary"],
                    }
                    if score["decision"]:
                        predicted.append(label)
                entry["classifiers"][classifier] = {
                    "labels": labels_payload,
                    "predicted": predicted,
                }
            output.append(entry)
        return output

    def _decision_records(self, current: RunArtifacts):
        records = []
        for classifier in CLASSIFIER_ORDER:
            for label, payload in sorted(
                ((key[1], value) for key, value in current.decisions.items() if key[0] == classifier)
            ):
                records.append(
                    {
                        "classifier": classifier,
                        "label": label,
                        "threshold": payload["threshold"],
                        "tie_rule": payload["tie_rule"],
                        "positive": sorted(
                            item["sample"] for item in payload["decisions"] if item["decision"]
                        ),
                        "boundary": sorted(
                            item["sample"] for item in payload["decisions"] if item["boundary"]
                        ),
                    }
                )
        return records


class _ValidSampleView:
    """Minimal sample view consumed by the feature builder."""

    def __init__(self, payload: dict, sample_dir=None):
        self.name = payload["sample"]
        self.width = payload["width"]
        self.height = payload["height"]
        self.image_fingerprint = payload["image_fingerprint"]
        self._sample_dir = sample_dir
        self._png_bytes = None

    @property
    def png_bytes(self) -> bytes:
        if self._png_bytes is None:
            if self._sample_dir is None:
                raise StageError("sample directory unavailable for lazy decode")
            self._png_bytes = (self._sample_dir / f"{self.name}.png").read_bytes()
        return self._png_bytes


class _FeatureView:
    """Adapts a persisted feature artifact dict to the quantizer interface."""

    def __init__(self, payload: dict):
        self.sample = payload["sample"]
        self.integers = np.asarray(payload["integers"], dtype=np.int64)
        self.residual = np.asarray(payload["residual"], dtype=np.float64)


class _MetricView:
    def __init__(self, payload: dict):
        self.classifier = payload["classifier"]
        self.label = payload["label"]
        self.threshold = payload["threshold"]
        self.true_positive = payload["tp"]
        self.false_positive = payload["fp"]
        self.true_negative = payload["tn"]
        self.false_negative = payload["fn"]
        self.boundary_count = payload["boundary_count"]
        self.precision = payload["precision"]
        self.recall = payload["recall"]
        self.f1 = payload["f1"]
        self.accuracy = payload["accuracy"]

    def to_dict(self):
        return {
            "classifier": self.classifier,
            "label": self.label,
            "threshold": self.threshold,
            "tp": self.true_positive,
            "fp": self.false_positive,
            "tn": self.true_negative,
            "fn": self.false_negative,
            "boundary_count": self.boundary_count,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "accuracy": self.accuracy,
        }


def np_from_truth(ground_truth: dict, label: str, sample_names: list[str]):
    return np.asarray(
        [1 if label in ground_truth.get(name, ()) else 0 for name in sample_names],
        dtype=np.int64,
    )
