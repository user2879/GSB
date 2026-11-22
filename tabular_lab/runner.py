"""End-to-end orchestration for incremental, resumable experiment runs."""

from __future__ import annotations

import math
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .cache import (
    ArtifactCache,
    CandidateIdentity,
    build_candidate_identity,
    build_conversion_key,
    penalty_fingerprint,
    stable_model_order_key,
)
from .config import ExperimentConfig, ModelConfig, load_config
from .data import DatasetBundle, SelectedDataset, dataset_summary, generate_dataset
from .errors import (
    E_MODEL_EXCEPTION,
    E_TRAIN_BUDGET_EXHAUSTED,
)
from .model_io import (
    ModelRecord,
    conversion_artifact,
    records_to_v2_payload,
    sidecar_filename,
)
from .scheduler import allocate_budgets, run_tasks_concurrently, TaskSpec
from .snapshot import LiveCheckpointStore, SnapshotSession, SnapshotStore
from .trainer import (
    TrainingFailure,
    TrainingHyperparameters,
    TrainingInterrupted,
    train_model,
)
from .util import atomic_write_json, atomic_write_text, canonical_json, content_hash, safe_filename


@dataclass
class Candidate:
    model: ModelConfig
    selected: SelectedDataset
    hyperparameters: TrainingHyperparameters
    penalty_fp: str
    stable_order_key: str
    quota: int = 0
    fault: dict[str, Any] | None = None
    conversion_fault: dict[str, Any] | None = None
    identity: CandidateIdentity | None = None
    training_artifact: dict[str, Any] | None = None
    training_artifact_fingerprint: str | None = None
    training_action: str = "pending"
    conversion_key: str | None = None
    conversion_artifact: dict[str, Any] | None = None
    conversion_action: str = "skipped"
    failure: dict[str, Any] | None = None
    interrupted: bool = False
    record: ModelRecord | None = None
    stage_seconds: dict[str, float] = field(default_factory=dict)
    execution_epochs: int = 0
    checkpoint_completed_epochs: int = 0
    resumed_from_checkpoint: bool = False

    @property
    def name(self) -> str:
        return self.model.name

    @property
    def feature_indices(self) -> list[int]:
        return list(self.selected.indices)

    @property
    def feature_names(self) -> list[str]:
        return list(self.selected.feature_names)


def _training_hyperparameters(config: ExperimentConfig, model: ModelConfig) -> TrainingHyperparameters:
    return TrainingHyperparameters(
        epochs=model.epochs_override or config.training.epochs,
        learning_rate=config.training.learning_rate,
        batch_size=config.training.batch_size,
        tolerance=config.training.tolerance,
        l1_zero_subgradient=config.training.l1_zero_subgradient,
        snapshot_every_batches=config.training.snapshot_every_batches,
    )


def _candidate_sort_key(candidate: Candidate) -> str:
    return candidate.stable_order_key


def _names_in_stable_order(candidates: list[Candidate], predicate: Any) -> list[str]:
    return [
        candidate.name
        for candidate in sorted(candidates, key=_candidate_sort_key)
        if predicate(candidate)
    ]


def _model_record(candidate: Candidate) -> ModelRecord:
    assert candidate.training_artifact is not None
    assert candidate.identity is not None
    artifact = candidate.training_artifact
    record = ModelRecord(
        name=candidate.name,
        scheme=candidate.model.scheme,
        coefficients=[float(value) for value in artifact["coefficients"]],
        intercept=float(artifact["intercept"]),
        feature_indices=list(candidate.identity.feature_indices),
        convergence_status=str(artifact["convergence_status"]),
        precision_label=str(artifact.get("precision_label", "float64")),
        penalty=float(candidate.model.penalty),
        mix=candidate.model.mix,
        feature_names=list(candidate.identity.feature_names),
        source_version="internal",
    )
    record.validate()
    return record


def _exception_type(fault: dict[str, Any]) -> type[Exception]:
    name = str(fault.get("exception", "RuntimeError"))
    return {
        "RuntimeError": RuntimeError,
        "ValueError": ValueError,
        "ZeroDivisionError": ZeroDivisionError,
        "ArithmeticError": ArithmeticError,
    }.get(name, RuntimeError)


def _maybe_raise_fault(
    fault: dict[str, Any] | None,
    *,
    stage: str,
    epoch: int,
) -> None:
    if not fault:
        return
    if fault.get("stage") != stage:
        return
    if stage in {"gradient", "train"} and int(fault.get("epoch", 1)) != int(epoch):
        return
    exception_type = _exception_type(fault)
    message = str(fault.get("message", f"injected {exception_type.__name__} in {stage}"))
    raise exception_type(message)


def _failure(
    model: str,
    *,
    stage: str,
    epoch: int,
    code: str,
    exception_type: str,
    message: str,
    traceback_text: str | None = None,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "epoch": int(epoch),
        "stage": stage,
        "code": code,
        "exception_type": exception_type,
        "message": message,
    }
    if traceback_text:
        payload["traceback"] = traceback_text
    return payload


def _execute_candidate(
    candidate: Candidate,
    *,
    config: ExperimentConfig,
    dataset: DatasetBundle,
    cache: ArtifactCache,
    live: LiveCheckpointStore,
    stop_event: threading.Event | None,
) -> dict[str, Any]:
    """Worker body. Returns structured data instead of leaking exceptions."""

    assert candidate.identity is not None
    identity = candidate.identity
    timings: dict[str, float] = {"standardize": 0.0, "train": 0.0, "convert": 0.0}
    training_action = "pending"
    conversion_action = "skipped"
    failure: dict[str, Any] | None = None
    interrupted = False
    execution_epochs = 0
    checkpoint_completed = 0
    resumed = False

    standardize_started = time.perf_counter()
    try:
        _maybe_raise_fault(candidate.fault, stage="standardize", epoch=0)
        # Materialize the selected matrices now so shape/preprocessing errors are
        # attributed to the model-local standardization stage.
        _ = candidate.selected.x_train
        _ = candidate.selected.x_validation
    except Exception as exc:
        timings["standardize"] = time.perf_counter() - standardize_started
        failure = _failure(
            candidate.name,
            stage="standardize",
            epoch=0,
            code=E_MODEL_EXCEPTION,
            exception_type=type(exc).__name__,
            message=str(exc),
        )
        return {
            "candidate": candidate.name,
            "training_action": "failed",
            "conversion_action": "skipped",
            "failure": failure,
            "timings": timings,
            "execution_epochs": 0,
            "checkpoint_completed_epochs": 0,
        }
    timings["standardize"] = time.perf_counter() - standardize_started

    cached = cache.load_train(identity.train_key)
    identity_payload = identity.to_dict()
    if cached is not None and canonical_json(cached.get("identity")) == canonical_json(identity_payload):
        artifact = cached["artifact"]
        artifact_fingerprint = str(cached["artifact_fingerprint"])
        training_action = "reused"
        live.remove(identity.train_key)
        checkpoint_completed = int(artifact.get("epochs_consumed", 0))
    else:
        cached = None
        resume_state = live.load(
            train_key=identity.train_key,
            config_hash=config.config_hash,
            data_fingerprint=dataset.data_fingerprint,
        )
        if resume_state:
            resumed = True
            checkpoint_completed = int(resume_state.get("completed_epochs", 0))

        def checkpoint_callback(state: dict[str, Any], reason: str) -> None:
            nonlocal checkpoint_completed
            checkpoint_completed = max(checkpoint_completed, int(state.get("completed_epochs", 0)))
            live.save(
                train_key=identity.train_key,
                model_name=candidate.name,
                config_hash=config.config_hash,
                data_fingerprint=dataset.data_fingerprint,
                state=state,
            )
            live.update_ledger(
                train_key=identity.train_key,
                model_name=candidate.name,
                consumed_epochs=int(state.get("completed_epochs", 0)),
                config_hash=config.config_hash,
                data_fingerprint=dataset.data_fingerprint,
            )

        train_started = time.perf_counter()
        try:
            artifact = train_model(
                model_name=candidate.name,
                scheme=candidate.model.scheme,
                penalty=candidate.model.penalty,
                mix=candidate.model.mix,
                x_train=candidate.selected.x_train,
                y_train=candidate.selected.y_train,
                x_validation=candidate.selected.x_validation,
                y_validation=candidate.selected.y_validation,
                hyperparameters=candidate.hyperparameters,
                quota=candidate.quota,
                fault=candidate.fault,
                resume_state=resume_state,
                checkpoint_callback=checkpoint_callback,
                stop_event=stop_event,
            )
        except TrainingInterrupted:
            timings["train"] = time.perf_counter() - train_started
            execution_epochs = max(0, checkpoint_completed - int((resume_state or {}).get("completed_epochs", 0)))
            return {
                "candidate": candidate.name,
                "training_action": "interrupted",
                "conversion_action": "skipped",
                "failure": None,
                "interrupted": True,
                "timings": timings,
                "execution_epochs": execution_epochs,
                "checkpoint_completed_epochs": checkpoint_completed,
                "resumed_from_checkpoint": resumed,
            }
        except TrainingFailure as exc:
            timings["train"] = time.perf_counter() - train_started
            failure = exc.to_dict(candidate.name)
            execution_epochs = max(0, checkpoint_completed - int((resume_state or {}).get("completed_epochs", 0)))
            return {
                "candidate": candidate.name,
                "training_action": "failed",
                "conversion_action": "skipped",
                "failure": failure,
                "timings": timings,
                "execution_epochs": execution_epochs,
                "checkpoint_completed_epochs": checkpoint_completed,
                "resumed_from_checkpoint": resumed,
            }
        except Exception as exc:
            timings["train"] = time.perf_counter() - train_started
            failure = _failure(
                candidate.name,
                stage="gradient",
                epoch=0,
                code=E_MODEL_EXCEPTION,
                exception_type=type(exc).__name__,
                message=str(exc),
            )
            return {
                "candidate": candidate.name,
                "training_action": "failed",
                "conversion_action": "skipped",
                "failure": failure,
                "timings": timings,
                "execution_epochs": execution_epochs,
                "checkpoint_completed_epochs": checkpoint_completed,
                "resumed_from_checkpoint": resumed,
            }
        timings["train"] = time.perf_counter() - train_started
        cached = cache.save_train(identity.train_key, artifact, identity)
        artifact_fingerprint = str(cached["artifact_fingerprint"])
        training_action = "trained"
        execution_epochs = max(0, int(artifact["epochs_consumed"]) - int((resume_state or {}).get("completed_epochs", 0)))
        checkpoint_completed = int(artifact["epochs_consumed"])
        live.remove(identity.train_key)
        live.update_ledger(
            train_key=identity.train_key,
            model_name=candidate.name,
            consumed_epochs=int(artifact["epochs_consumed"]),
            config_hash=config.config_hash,
            data_fingerprint=dataset.data_fingerprint,
        )

    candidate.training_artifact = artifact
    candidate.training_artifact_fingerprint = artifact_fingerprint
    record = _model_record(candidate)
    conversion_key = build_conversion_key(
        identity,
        artifact_fingerprint,
        config.exchange_version,
        dataset.schema_fingerprint,
        candidate.conversion_fault,
        config.conversion_tolerance,
    )

    conversion_cached = cache.load_conversion(conversion_key)
    if conversion_cached is not None:
        conversion = conversion_cached.get("conversion")
        if (
            isinstance(conversion, dict)
            and conversion.get("model") == candidate.name
            and conversion.get("version") == config.exchange_version
        ):
            conversion_action = "reused"
        else:
            conversion_cached = None
            conversion = None
    if conversion_cached is None:
        convert_started = time.perf_counter()
        try:
            _maybe_raise_fault(
                candidate.conversion_fault,
                stage="conversion",
                epoch=int(artifact.get("epochs_consumed", 0)),
            )
            conversion = conversion_artifact(record, config.exchange_version)
            if config.exchange_version == "v1":
                sidecar = conversion.get("sidecar", {})
                fidelity = sidecar.get("fidelity", {}) if isinstance(sidecar, dict) else {}
                precision_error = max(
                    float(fidelity.get("max_abs_coefficient_residual", 0.0)),
                    float(fidelity.get("abs_intercept_residual", 0.0)),
                )
                if precision_error > config.conversion_tolerance:
                    raise ValueError(
                        "v1 milli conversion exceeds conversion_tolerance: "
                        f"{precision_error} > {config.conversion_tolerance}"
                    )
            cache.save_conversion(
                conversion_key,
                conversion,
                training_artifact_fingerprint=artifact_fingerprint,
                exchange_version=config.exchange_version,
            )
            conversion_action = "converted"
        except Exception as exc:
            timings["convert"] = time.perf_counter() - convert_started
            failure = _failure(
                candidate.name,
                stage="conversion",
                epoch=int(artifact.get("epochs_consumed", 0)),
                code=E_MODEL_EXCEPTION,
                exception_type=type(exc).__name__,
                message=str(exc),
            )
            return {
                "candidate": candidate.name,
                "training_action": training_action,
                "conversion_action": "failed",
                "training_artifact": artifact,
                "training_artifact_fingerprint": artifact_fingerprint,
                "conversion_key": conversion_key,
                "failure": failure,
                "timings": timings,
                "execution_epochs": execution_epochs,
                "checkpoint_completed_epochs": checkpoint_completed,
                "resumed_from_checkpoint": resumed,
            }
        timings["convert"] = time.perf_counter() - convert_started

    return {
        "candidate": candidate.name,
        "training_action": training_action,
        "conversion_action": conversion_action,
        "training_artifact": artifact,
        "training_artifact_fingerprint": artifact_fingerprint,
        "conversion_key": conversion_key,
        "conversion_artifact": conversion,
        "failure": None,
        "timings": timings,
        "execution_epochs": execution_epochs,
        "checkpoint_completed_epochs": checkpoint_completed,
        "resumed_from_checkpoint": resumed,
    }


def _clean_derived_directories(session: SnapshotSession) -> None:
    for child in ("models", "exports", "sidecars", "reports"):
        target = session.staging / child
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)


def _invalidation_reasons(
    candidate: Candidate,
    previous_manifest: dict[str, Any] | None,
    config: ExperimentConfig,
) -> list[str]:
    if candidate.identity is None:
        return ["identity_unavailable"]
    current = candidate.identity
    previous = None
    if previous_manifest:
        refs = previous_manifest.get("model_refs", {})
        if isinstance(refs, dict):
            previous = refs.get(candidate.name)
    reasons: list[str] = []
    if not isinstance(previous, dict):
        reasons.append("new_model")
    else:
        if previous.get("scheme") != current.scheme:
            reasons.append("scheme_changed")
        if previous.get("penalty_fingerprint") != current.penalty_fingerprint:
            reasons.append("penalty_or_mix_changed")
        if list(previous.get("feature_names", [])) != list(current.feature_names):
            reasons.append("feature_whitelist_changed")
        elif previous.get("data_slice_fingerprint") != current.data_slice_fingerprint:
            reasons.append("data_slice_changed")
        if previous.get("standardization_fingerprint") != current.standardization_fingerprint:
            reasons.append("standardization_stats_changed")
        if previous.get("sample_fingerprint") != current.sample_fingerprint:
            reasons.append("sample_set_changed")
        if int(previous.get("quota", -1)) != int(current.quota):
            reasons.append("budget_quota_changed")
        if previous.get("training_params_fingerprint") != current.training_params_fingerprint:
            reasons.append("training_parameters_changed")
        if previous.get("train_key") != current.train_key:
            reasons.append("training_key_changed")
        previous_version = previous.get("exchange_version")
        if previous_version is not None and previous_version != config.exchange_version:
            reasons.append("exchange_version_changed")
        if previous.get("conversion_key") and candidate.conversion_key and previous.get("conversion_key") != candidate.conversion_key:
            reasons.append("version_conversion_invalidated")
    if candidate.training_action in {"trained", "failed"} and not reasons:
        reasons.append("cache_artifact_missing")
    if candidate.conversion_action in {"converted", "failed"} and not reasons:
        reasons.append("conversion_cache_missing")
    return sorted(set(reasons))


def _aggregate_validation(candidates: list[Candidate]) -> dict[str, Any]:
    successful = [
        candidate
        for candidate in candidates
        if candidate.training_artifact is not None
        and candidate.conversion_artifact is not None
        and candidate.failure is None
        and not candidate.interrupted
    ]
    rows: list[dict[str, Any]] = []
    for candidate in sorted(successful, key=_candidate_sort_key):
        metrics = candidate.training_artifact.get("validation_metrics", {})
        rows.append({"model": candidate.name, "scheme": candidate.model.scheme, **metrics})
    count = len(rows)
    return {
        "model_count": count,
        "mean_mse": math.fsum(float(row.get("mse", 0.0)) for row in rows) / count if count else None,
        "mean_rmse": math.fsum(float(row.get("rmse", 0.0)) for row in rows) / count if count else None,
        "mean_accuracy@0.5": math.fsum(float(row.get("accuracy@0.5", 0.0)) for row in rows) / count if count else None,
        "models": rows,
    }


def _materialize_outputs(
    session: SnapshotSession,
    candidates: list[Candidate],
    config: ExperimentConfig,
) -> dict[str, Any]:
    successful = [
        candidate
        for candidate in sorted(candidates, key=_candidate_sort_key)
        if candidate.training_artifact is not None
        and candidate.conversion_artifact is not None
        and candidate.failure is None
        and not candidate.interrupted
    ]
    if config.exchange_version == "v1":
        lines: list[str] = []
        sidecar_refs: dict[str, str] = {}
        for candidate in successful:
            conversion = candidate.conversion_artifact or {}
            lines.extend(str(line) for line in conversion.get("lines", []))
            sidecar = conversion.get("sidecar")
            if isinstance(sidecar, dict):
                relative = f"sidecars/{sidecar_filename(candidate.name)}"
                atomic_write_json(session.staging / relative, sidecar)
                sidecar_refs[candidate.name] = relative
        main_path = "exports/models.v1.txt"
        if lines:
            atomic_write_text(session.staging / main_path, "\n".join(lines) + "\n")
        else:
            atomic_write_text(session.staging / main_path, "# no successfully converted models\n")
        conversion_manifest = {
            "version": "v1",
            "main_file": main_path,
            "field_widths": {
                "model": 20,
                "feature_index": 4,
                "coefficient_milli": 12,
                "penalty_code": 2,
                "intercept_milli": 12,
            },
            "sidecars": sidecar_refs,
            "models": [candidate.name for candidate in successful],
        }
    else:
        payload = records_to_v2_payload([candidate.record for candidate in successful if candidate.record is not None])
        payload["models"] = [
            candidate.conversion_artifact["payload"]
            for candidate in successful
            if candidate.conversion_artifact is not None
        ]
        main_path = "exports/models.v2.json"
        atomic_write_json(session.staging / main_path, payload)
        conversion_manifest = {
            "version": "v2",
            "main_file": main_path,
            "models": [candidate.name for candidate in successful],
            "sidecars": {},
        }
    atomic_write_json(session.staging / "exports/conversion_manifest.json", conversion_manifest)
    return conversion_manifest


def run_experiment(
    config_path: str | Path,
    *,
    state_dir: str | Path,
    rebuild: bool = False,
    resume: bool = False,
    exchange_version: str | None = None,
    report_path: str | Path | None = None,
    stop_event: threading.Event | None = None,
) -> dict[str, Any]:
    overall_started = time.perf_counter()
    overrides: dict[str, Any] = {}
    if exchange_version is not None:
        overrides["exchange_version"] = exchange_version
    config = load_config(config_path, overrides=overrides or None)
    stop_event = stop_event or threading.Event()

    load_started = time.perf_counter()
    dataset = generate_dataset(config.data, config.seed)
    candidates: list[Candidate] = []
    for model in config.models:
        indices = dataset.resolve_features(model.features)
        selected = dataset.selected(indices)
        hyperparameters = _training_hyperparameters(config, model)
        penalty_fp = penalty_fingerprint(model.scheme, model.penalty, model.mix)
        candidates.append(
            Candidate(
                model=model,
                selected=selected,
                hyperparameters=hyperparameters,
                penalty_fp=penalty_fp,
                stable_order_key=stable_model_order_key(model.scheme, penalty_fp, model.name),
                fault=config.faults.get(model.name),
            )
        )
    candidates.sort(key=_candidate_sort_key)
    load_seconds = time.perf_counter() - load_started

    store = SnapshotStore(state_dir, keep_generations=config.snapshot.keep_generations)
    strict_resume = resume or config.snapshot.strict_resume
    expected_version = config.snapshot.expected_version if strict_resume else None
    previous = store.check_resume(
        config_hash=config.config_hash,
        data_fingerprint=dataset.data_fingerprint,
        strict=strict_resume and not rebuild,
        expected_version=expected_version,
    )
    if rebuild:
        previous_for_session = None
        run_mode = "rebuild"
    elif previous is None:
        previous_for_session = None
        run_mode = "clean"
    else:
        previous_for_session = previous
        run_mode = "incremental"
    session = store.begin(previous=previous_for_session, rebuild=rebuild)
    _clean_derived_directories(session)
    live = LiveCheckpointStore(state_dir)

    standardize_started = time.perf_counter()
    allocation = allocate_budgets(
        [(candidate.name, candidate.stable_order_key) for candidate in candidates],
        global_epochs=config.budget.global_epochs,
        per_model_epochs=config.budget.per_model_epochs,
    )
    for candidate in candidates:
        candidate.quota = allocation.quotas[candidate.name]
        training_fault = candidate.fault if candidate.fault and candidate.fault.get("stage") != "conversion" else None
        conversion_fault = candidate.fault if candidate.fault and candidate.fault.get("stage") == "conversion" else None
        candidate.conversion_fault = conversion_fault
        candidate.identity = build_candidate_identity(
            model_name=candidate.name,
            scheme=candidate.model.scheme,
            penalty=candidate.model.penalty,
            mix=candidate.model.mix,
            feature_names=candidate.feature_names,
            feature_indices=candidate.feature_indices,
            selected_data_slice_fingerprint=candidate.selected.data_slice_fingerprint(),
            standardization_fingerprint=candidate.selected.standardization_fingerprint(),
            sample_fingerprint=candidate.selected.sample_fingerprint(),
            training_hyperparameters=candidate.hyperparameters.canonical(),
            quota=candidate.quota,
            requested_epochs=candidate.hyperparameters.epochs,
            seed=config.seed,
            fault=training_fault,
        )
    standardize_seconds = time.perf_counter() - standardize_started

    pending_failures: list[dict[str, Any]] = []
    tasks: list[TaskSpec] = []
    for candidate in candidates:
        if candidate.quota <= 0:
            candidate.training_action = "pending"
            candidate.conversion_action = "skipped"
            failure = _failure(
                candidate.name,
                stage="schedule",
                epoch=0,
                code=E_TRAIN_BUDGET_EXHAUSTED,
                exception_type="BudgetExhausted",
                message="global training budget exhausted before this model could start",
            )
            candidate.failure = failure
            pending_failures.append(failure)
            session.write_json(
                f"models/{safe_filename(candidate.name)}.pending.json",
                {"model": candidate.name, "reason": failure, "identity": candidate.identity.to_dict()},
            )
            continue
        tasks.append(
            TaskSpec(
                model=candidate.name,
                stable_order_key=candidate.stable_order_key,
                function=lambda candidate=candidate: _execute_candidate(
                    candidate,
                    config=config,
                    dataset=dataset,
                    cache=session.cache,
                    live=live,
                    stop_event=stop_event,
                ),
            )
        )

    outcomes = run_tasks_concurrently(
        tasks,
        max_workers=config.runtime.max_concurrency,
        stop_event=stop_event,
    )
    outcome_by_name = {outcome.model: outcome for outcome in outcomes}
    failures: list[dict[str, Any]] = list(pending_failures)
    interrupted_models: list[str] = []

    for candidate in candidates:
        if candidate.quota <= 0:
            continue
        outcome = outcome_by_name.get(candidate.name)
        if outcome is None:
            failure = _failure(
                candidate.name,
                stage="schedule",
                epoch=0,
                code=E_MODEL_EXCEPTION,
                exception_type="MissingTaskOutcome",
                message="scheduler returned no outcome for this model",
            )
            candidate.training_action = "failed"
            candidate.failure = failure
            failures.append(failure)
            continue
        if not outcome.succeeded:
            failure = _failure(
                candidate.name,
                stage="unknown",
                epoch=0,
                code=E_MODEL_EXCEPTION,
                exception_type=outcome.exception_type or "Exception",
                message=outcome.exception_message or "unknown worker failure",
                traceback_text=outcome.traceback_text,
            )
            candidate.training_action = "failed"
            candidate.failure = failure
            failures.append(failure)
            continue
        result = outcome.value or {}
        candidate.training_action = str(result.get("training_action", "failed"))
        candidate.conversion_action = str(result.get("conversion_action", "skipped"))
        candidate.training_artifact = result.get("training_artifact")
        candidate.training_artifact_fingerprint = result.get("training_artifact_fingerprint")
        candidate.conversion_key = result.get("conversion_key")
        candidate.conversion_artifact = result.get("conversion_artifact")
        candidate.stage_seconds = dict(result.get("timings", {}))
        candidate.execution_epochs = int(result.get("execution_epochs", 0))
        candidate.checkpoint_completed_epochs = int(result.get("checkpoint_completed_epochs", 0))
        candidate.resumed_from_checkpoint = bool(result.get("resumed_from_checkpoint", False))
        candidate.interrupted = bool(result.get("interrupted", False))
        if candidate.interrupted:
            interrupted_models.append(candidate.name)
        if result.get("failure"):
            candidate.failure = dict(result["failure"])
            failures.append(candidate.failure)
        if candidate.training_artifact is not None:
            candidate.record = _model_record(candidate)

    failures.sort(key=lambda failure: next(c.stable_order_key for c in candidates if c.name == failure["model"]))
    interrupted_set = set(interrupted_models)
    interrupted_models = [
        candidate.name
        for candidate in sorted(candidates, key=_candidate_sort_key)
        if candidate.name in interrupted_set
    ]

    completed_candidates = [
        candidate
        for candidate in candidates
        if candidate.training_artifact is not None
        and candidate.conversion_artifact is not None
        and candidate.failure is None
        and not candidate.interrupted
    ]
    completed_names = [candidate.name for candidate in completed_candidates]
    failed_names = _names_in_stable_order(
        candidates,
        lambda candidate: candidate.failure is not None and candidate.training_action != "pending",
    )
    pending_names = _names_in_stable_order(
        candidates,
        lambda candidate: candidate.training_action == "pending" or candidate.interrupted,
    )

    for candidate in candidates:
        if candidate.identity is not None:
            reference = candidate.identity.to_dict()
            reference.update(
                {
                    "exchange_version": config.exchange_version,
                    "training_artifact_fingerprint": candidate.training_artifact_fingerprint,
                    "conversion_key": candidate.conversion_key,
                    "conversion_payload_hash": (
                        candidate.conversion_artifact.get("payload_hash")
                        if isinstance(candidate.conversion_artifact, dict)
                        else None
                    ),
                    "schema_fingerprint": dataset.schema_fingerprint,
                    "train_indices": candidate.selected.bundle.train_indices.tolist(),
                    "validation_indices": candidate.selected.bundle.validation_indices.tolist(),
                }
            )
            session.write_json(
                f"models/{safe_filename(candidate.name)}.json",
                {
                    "identity": reference,
                    "record": candidate.record.canonical() if candidate.record else None,
                    "artifact": candidate.training_artifact,
                    "conversion": candidate.conversion_artifact,
                    "failure": candidate.failure,
                },
            )

    output_manifest = _materialize_outputs(session, candidates, config)

    reuse_list = _names_in_stable_order(
        candidates,
        lambda candidate: candidate.training_action == "reused" and candidate.conversion_action == "reused",
    )
    recompute_list = _names_in_stable_order(
        candidates,
        lambda candidate: candidate.training_action in {"trained", "failed", "interrupted"}
        or candidate.conversion_action in {"converted", "failed"},
    )
    training_reused = _names_in_stable_order(candidates, lambda candidate: candidate.training_action == "reused")
    training_retrained = _names_in_stable_order(candidates, lambda candidate: candidate.training_action == "trained")
    conversion_reused = _names_in_stable_order(candidates, lambda candidate: candidate.conversion_action == "reused")
    conversion_recomputed = _names_in_stable_order(
        candidates, lambda candidate: candidate.conversion_action == "converted"
    )
    invalidations = [
        {
            "model": candidate.name,
            "stable_order_key": candidate.stable_order_key,
            "training_action": candidate.training_action,
            "conversion_action": candidate.conversion_action,
            "reasons": _invalidation_reasons(
                candidate,
                previous.manifest if previous and not rebuild else None,
                config,
            ),
        }
        for candidate in candidates
        if candidate.training_action != "reused" or candidate.conversion_action != "reused"
    ]
    invalidations.sort(key=lambda item: item["stable_order_key"])

    validation_summary = _aggregate_validation(candidates)
    amended_consumed = 0
    for candidate in candidates:
        if candidate.training_artifact is not None:
            amended_consumed += int(candidate.training_artifact.get("epochs_consumed", 0))
        elif candidate.training_action == "failed" and candidate.failure:
            amended_consumed += int(candidate.checkpoint_completed_epochs)
        elif candidate.interrupted:
            amended_consumed += int(candidate.checkpoint_completed_epochs)
    execution_consumed = sum(candidate.execution_epochs for candidate in candidates)

    train_stage_seconds = math.fsum(float(c.stage_seconds.get("train", 0.0)) for c in candidates)
    standardize_stage_seconds = standardize_seconds + math.fsum(
        float(c.stage_seconds.get("standardize", 0.0)) for c in candidates
    )
    convert_stage_seconds = math.fsum(float(c.stage_seconds.get("convert", 0.0)) for c in candidates)

    model_refs = {
        candidate.name: {
            **candidate.identity.to_dict(),
            "exchange_version": config.exchange_version,
            "training_artifact_fingerprint": candidate.training_artifact_fingerprint,
            "conversion_key": candidate.conversion_key,
            "conversion_payload_hash": (
                candidate.conversion_artifact.get("payload_hash")
                if isinstance(candidate.conversion_artifact, dict)
                else None
            ),
            "schema_fingerprint": dataset.schema_fingerprint,
            "train_indices": candidate.selected.bundle.train_indices.tolist(),
            "validation_indices": candidate.selected.bundle.validation_indices.tolist(),
        }
        for candidate in candidates
        if candidate.identity is not None
    }
    model_refs = {name: model_refs[name] for name in sorted(model_refs)}

    load_stage_version = content_hash("stage-load", {"config": config.config_hash, "data": dataset.data_fingerprint})
    standardize_stage_version = content_hash(
        "stage-standardize",
        {
            "data": dataset.data_fingerprint,
            "stats": [
                {
                    "model": candidate.name,
                    "data_slice": candidate.identity.data_slice_fingerprint,
                    "standardization": candidate.identity.standardization_fingerprint,
                }
                for candidate in candidates
                if candidate.identity is not None
            ],
        },
    )
    train_stage_version = content_hash(
        "stage-train",
        {
            "references": [
                {
                    "model": candidate.name,
                    "train_key": candidate.identity.train_key if candidate.identity else None,
                    "artifact": candidate.training_artifact_fingerprint,
                    "failure": candidate.failure if candidate.training_action == "failed" else None,
                }
                for candidate in candidates
            ]
        },
    )
    convert_stage_version = content_hash(
        "stage-convert",
        {
            "version": config.exchange_version,
            "references": [
                {
                    "model": candidate.name,
                    "conversion_key": candidate.conversion_key,
                    "payload_hash": (
                        candidate.conversion_artifact.get("payload_hash")
                        if isinstance(candidate.conversion_artifact, dict)
                        else None
                    ),
                    "failure": candidate.failure if candidate.conversion_action == "failed" else None,
                }
                for candidate in candidates
            ],
        },
    )
    session.write_stage("load", load_stage_version, dataset_summary(dataset))
    session.write_stage(
        "standardize",
        standardize_stage_version,
        {
            "models": [
                {
                    "model": candidate.name,
                    "feature_names": candidate.feature_names,
                    "feature_indices": candidate.feature_indices,
                    "means": [float(dataset.feature_means[index]) for index in candidate.feature_indices],
                    "scales": [float(dataset.feature_scales[index]) for index in candidate.feature_indices],
                    "data_slice_fingerprint": candidate.identity.data_slice_fingerprint if candidate.identity else None,
                    "standardization_fingerprint": (
                        candidate.identity.standardization_fingerprint if candidate.identity else None
                    ),
                }
                for candidate in candidates
            ]
        },
    )
    session.write_stage(
        "train",
        train_stage_version,
        {
            "models": [
                {
                    "model": candidate.name,
                    "train_key": candidate.identity.train_key if candidate.identity else None,
                    "training_artifact_fingerprint": candidate.training_artifact_fingerprint,
                    "epochs_consumed": (
                        int(candidate.training_artifact.get("epochs_consumed", 0))
                        if candidate.training_artifact
                        else candidate.checkpoint_completed_epochs
                    ),
                }
                for candidate in candidates
            ]
        },
    )
    session.write_stage(
        "convert",
        convert_stage_version,
        {
            "exchange_version": config.exchange_version,
            "output_manifest": output_manifest,
            "models": [
                {
                    "model": candidate.name,
                    "conversion_key": candidate.conversion_key,
                    "payload_hash": (
                        candidate.conversion_artifact.get("payload_hash")
                        if isinstance(candidate.conversion_artifact, dict)
                        else None
                    ),
                }
                for candidate in candidates
            ],
        },
    )

    status = "ok" if not failures and not interrupted_models else ("interrupted" if interrupted_models else "partial")
    exit_code = 130 if interrupted_models else (4 if failures else 0)
    stage_seconds = {
        "load": load_seconds,
        "standardize": standardize_stage_seconds,
        "train": train_stage_seconds,
        "convert": convert_stage_seconds,
    }
    diagnostics = {
        "stage_seconds": {key: round(value, 9) for key, value in stage_seconds.items()},
        "total_seconds_before_commit": round(time.perf_counter() - overall_started, 9),
        "concurrency": {
            "configured_max": config.runtime.max_concurrency,
            "upper_limit": config.runtime.concurrency_limit,
            "scheduled_training_tasks": len(tasks),
            "scheduler": "thread-pool" if config.runtime.max_concurrency > 1 else "serial-stable",
        },
        "budget": {
            "initial_global_epochs": allocation.initial_budget,
            "per_model_quota": config.budget.per_model_epochs,
            "reserved_epochs": allocation.total_reserved,
            "reservations": {
                name: allocation.reserved_epochs[name] for name in sorted(allocation.reserved_epochs)
            },
            "consumed_epochs_this_execution": int(execution_consumed),
            "model_epochs_persisted": int(amended_consumed),
            "unstarted_models": list(allocation.unstarted_models),
            "live_ledger": live.ledger_summary(),
        },
        "model_timings": {
            candidate.name: {
                "standardize_seconds": round(float(candidate.stage_seconds.get("standardize", 0.0)), 9),
                "train_seconds": round(float(candidate.stage_seconds.get("train", 0.0)), 9),
                "convert_seconds": round(float(candidate.stage_seconds.get("convert", 0.0)), 9),
                "training_action": candidate.training_action,
                "conversion_action": candidate.conversion_action,
                "resumed_from_checkpoint": candidate.resumed_from_checkpoint,
            }
            for candidate in candidates
        },
    }
    audit = {
        "reuse_list": reuse_list,
        "recompute_list": recompute_list,
        "training_reused": training_reused,
        "training_retrained": training_retrained,
        "conversion_reused": conversion_reused,
        "conversion_recomputed": conversion_recomputed,
        "invalidations": invalidations,
        "failure_order": [failure["model"] for failure in failures],
        "pending_order": list(pending_names),
        "interrupted_order": interrupted_models,
    }
    report: dict[str, Any] = {
        "status": status,
        "exit_code": exit_code,
        "experiment": config.name,
        "config_hash": config.config_hash,
        "data_fingerprint": dataset.data_fingerprint,
        "exchange_version": config.exchange_version,
        "run_mode": run_mode,
        "models": [],
        "reused_models": reuse_list,
        "retrained_models": training_retrained,
        "reconverted_models": conversion_recomputed,
        "recompute_list": recompute_list,
        "reuse_list": reuse_list,
        "failures": failures,
        "pending_models": list(pending_names),
        "interrupted_models": interrupted_models,
        "validation_summary": validation_summary,
        "audit": audit,
        "invalidations": invalidations,
        "diagnostics": diagnostics,
        "outputs": output_manifest,
    }

    for candidate in sorted(candidates, key=_candidate_sort_key):
        entry: dict[str, Any] = {
            "name": candidate.name,
            "scheme": candidate.model.scheme,
            "penalty": candidate.model.penalty,
            "mix": candidate.model.mix,
            "feature_names": candidate.feature_names,
            "feature_indices": candidate.feature_indices,
            "stable_order_key": candidate.stable_order_key,
            "quota": candidate.quota,
            "training_action": candidate.training_action,
            "conversion_action": candidate.conversion_action,
            "training_key": candidate.identity.train_key if candidate.identity else None,
            "conversion_key": candidate.conversion_key,
            "training_artifact_fingerprint": candidate.training_artifact_fingerprint,
        }
        if candidate.training_artifact:
            entry.update(
                {
                    "status": candidate.training_artifact.get("convergence_status"),
                    "coefficients": candidate.training_artifact.get("coefficients"),
                    "intercept": candidate.training_artifact.get("intercept"),
                    "loss_trajectory": candidate.training_artifact.get("loss_trajectory"),
                    "coefficient_snapshots": candidate.training_artifact.get("coefficient_snapshots"),
                    "validation_metrics": candidate.training_artifact.get("validation_metrics"),
                    "epochs_consumed": candidate.training_artifact.get("epochs_consumed"),
                }
            )
        if candidate.failure:
            entry["failure"] = candidate.failure
        report["models"].append(entry)

    semantic_payload = {
        "experiment": config.name,
        "data_fingerprint": dataset.data_fingerprint,
        "exchange_version": config.exchange_version,
        "stage_versions": dict(sorted(session.stage_versions.items())),
        "completed_models": list(completed_names),
        "failed_models": list(failed_names),
        "pending_models": list(pending_names),
        "consumed_epochs": int(amended_consumed),
        "models": [
            {
                "name": candidate.name,
                "scheme": candidate.model.scheme,
                "penalty": candidate.model.penalty,
                "mix": candidate.model.mix,
                "feature_names": candidate.feature_names,
                "feature_indices": candidate.feature_indices,
                "training_key": candidate.identity.train_key if candidate.identity else None,
                "conversion_key": candidate.conversion_key,
                "training_artifact_fingerprint": candidate.training_artifact_fingerprint,
                "artifact": candidate.training_artifact,
                "record": candidate.record.canonical() if candidate.record else None,
                "conversion": candidate.conversion_artifact,
                "failure": candidate.failure,
            }
            for candidate in sorted(candidates, key=_candidate_sort_key)
        ],
        "failures": [
            {key: value for key, value in failure.items() if key != "traceback"} for failure in failures
        ],
        "validation_summary": validation_summary,
        "output_manifest": output_manifest,
    }
    report["semantic_digest"] = content_hash("result", semantic_payload)

    session.write_json(
        "reports/run_report.json",
        {**report, "snapshot": {"global_version": session.generation}},
    )
    snapshot_commit_started = time.perf_counter()
    handle = session.commit(
        config_hash=config.config_hash,
        data_fingerprint=dataset.data_fingerprint,
        exchange_version=config.exchange_version,
        completed_models=completed_names,
        failed_models=failed_names,
        pending_models=list(pending_names),
        consumed_epochs=int(amended_consumed),
        model_refs=model_refs,
        extra={
            "run_mode": run_mode,
            "audit": audit,
            "diagnostics": diagnostics,
            "output_manifest": output_manifest,
            "semantic_digest": report["semantic_digest"],
        },
        final_status=status,
    )
    diagnostics["stage_seconds"]["snapshot_commit"] = round(
        time.perf_counter() - snapshot_commit_started, 9
    )
    report["snapshot"] = {
        "global_version": handle.manifest["global_version"],
        "path": str(handle.path),
        "stage_versions": handle.manifest["stage_versions"],
        "completed_models": handle.manifest["completed_models"],
        "consumed_epochs": handle.manifest["consumed_epochs"],
    }
    report["diagnostics"] = diagnostics
    if report_path is not None:
        atomic_write_json(report_path, report)
    return report
