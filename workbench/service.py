"""Application service: replay, estimation, snapshots, recovery and replay checks."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Sequence

from .constants import (
    CLOCK_BASIS,
    SCHEMA_VERSION,
    SNAPSHOT_FORMAT_VERSION,
    TRUTH_SOURCE,
)
from .db import Repository
from .errors import (
    LogValidationError,
    ParameterError,
    RecoveryError,
    ReplayError,
    SnapshotError,
    WorkbenchError,
)
from .estimator import SampleInput, estimate_windows
from .jsonlog import StructuredLogger
from .models import (
    AggregationSpec,
    MasterRecord,
    Progress,
    ReplayBatch,
    ReplayParams,
    ReplayRun,
    Sample,
)
from .simulator import build_replay_plan, materialize_samples
from .snapshot import (
    load_snapshot_document,
    parse_snapshot_contract,
    snapshot_sequence,
)
from .util import (
    atomic_write_json,
    canonical_json,
    deep_first_diff,
    fixed_from_ns,
    is_writable_by_mode,
    read_json,
    sha256_json,
    sha256_file,
    utc_now_iso,
)
from .validation import (
    validate_aggregation_spec,
    validate_master_records,
    validate_progress,
    validate_replay_params,
)


class WorkbenchService:
    def __init__(
        self,
        workspace: str | Path,
        *,
        db_path: str | Path | None = None,
        log_path: str | Path | None = None,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir = self.workspace / "artifacts" / "workbench"
        self.snapshot_dir = self.artifacts_dir / "snapshots"
        self.report_dir = self.artifacts_dir / "reports"
        self.log_dir = self.artifacts_dir / "logs"
        for directory in (self.artifacts_dir, self.snapshot_dir, self.report_dir, self.log_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self.db_path = Path(db_path) if db_path is not None else self.artifacts_dir / "workbench.sqlite3"
        self.log_path = Path(log_path) if log_path is not None else self.log_dir / "workbench.jsonl"
        self.logger = StructuredLogger(self.log_path)
        self.repo = Repository(self.db_path)
        self.recovery_report = self.recover_after_crash()
        self.logger.event(
            "service_opened",
            workspace=str(self.workspace),
            db_path=str(self.db_path),
            recovery=self.recovery_report,
        )

    def close(self) -> None:
        self.repo.close()

    def __enter__(self) -> "WorkbenchService":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def import_master_log(
        self, records: Sequence[MasterRecord], *, source: str | None = None
    ) -> dict[str, Any]:
        validated = validate_master_records([record.to_dict() for record in records])
        if self.repo.log_stats()["count"]:
            self.logger.event(
                "master_log_replace_requested",
                source=source,
                previous=self.repo.log_stats(),
                incoming_count=len(validated),
            )
        stats = self.repo.replace_master_log(validated)
        self.logger.event("master_log_imported", source=source, log=stats)
        return stats

    def import_master_records(
        self, raw_records: Any, *, source: str | None = None
    ) -> dict[str, Any]:
        """Validate a complete JSON-style batch before any write occurs."""

        records = validate_master_records(raw_records)
        return self.import_master_log(records, source=source)

    def recover_after_crash(self) -> dict[str, Any]:
        unknown = self.repo.mark_running_batches_unknown()
        report = {
            "recovered_unknown_batches": [batch.to_dict() for batch in unknown],
            "requires_rerun": bool(unknown),
            "exit_report_available": False,
            "notice": (
                "Only fully committed batches are trusted. No report exists for a forcibly "
                "terminated process; incomplete batches must be rerun."
            ),
        }
        if unknown:
            self.logger.event("crash_recovery_unknown_batches", **report)
        return report

    def run_replay(
        self,
        params: ReplayParams,
        spec: AggregationSpec,
        *,
        crash_after_completed_batches: int | None = None,
    ) -> dict[str, Any]:
        records = self.repo.get_master_log()
        if not records:
            raise LogValidationError("主库日志为空，不能回放", seq=0, field="master_log")
        plan = build_replay_plan(records, params, spec.scale_power)
        run = self.repo.ensure_run(params, spec)
        progress = self.repo.progress_for(params.version)
        if progress is not None and progress.status == "COMPLETE":
            samples = self.repo.get_samples(params.version)
            if len(samples) != len(records):
                raise RecoveryError(
                    "进度声称完成但样本数不完整",
                    field="samples",
                    details={"samples": len(samples), "records": len(records)},
                )
            return self._run_summary(params, spec, progress, samples, resumed=True)
        if progress is not None and progress.status == "RESTORED_STALE_LOG_MISMATCH":
            raise ReplayError(
                "当前回放状态由日志摘要不匹配的快照恢复，必须先修复日志或恢复匹配快照",
                field="replay_progress.status",
            )

        last_seq = 0 if progress is None else progress.last_replayed_seq
        validate_progress(last_seq, len(records))
        if last_seq and last_seq % params.concurrency != 0 and last_seq != len(records):
            raise RecoveryError(
                "完成游标不在并发批次边界上，无法安全续跑",
                seq=last_seq,
                field="last_replayed_seq",
                details={"concurrency": params.concurrency},
            )
        existing_samples = self.repo.get_samples(params.version)
        existing_seqs = sorted(sample.commit_seq for sample in existing_samples)
        expected_existing = list(range(1, last_seq + 1))
        if existing_seqs != expected_existing:
            first_issue = next(
                (seq for seq, actual in zip(expected_existing, existing_seqs) if seq != actual),
                expected_existing[len(existing_seqs)] if len(existing_seqs) < len(expected_existing) else 0,
            )
            raise RecoveryError(
                "完成游标与已持久化样本不一致",
                seq=first_issue or None,
                field="replay_samples",
            )

        start_batch_index = 0 if last_seq == 0 else last_seq // params.concurrency
        completed_batches = 0 if progress is None else progress.completed_batches
        for batch in plan[start_batch_index:]:
            attempt = self.repo.next_attempt(params.version, batch.batch_index)
            start_position = batch.batch_index * params.concurrency
            started_mono_ns = params.clock_base_ms * 1_000_000 + (
                start_position * 1_000_000_000 // params.rate_rps
            )
            batch_key = self.repo.begin_batch(
                params.version,
                batch.batch_index,
                attempt,
                batch.first_seq,
                batch.last_seq,
                started_mono_ns,
            )
            self.logger.event(
                "replay_batch_started",
                param_version=params.version,
                batch_index=batch.batch_index,
                attempt=attempt,
                batch_key=batch_key,
                first_seq=batch.first_seq,
                last_seq=batch.last_seq,
            )
            if (
                crash_after_completed_batches is not None
                and completed_batches == crash_after_completed_batches
            ):
                self.logger.event(
                    "forced_termination_injected",
                    param_version=params.version,
                    batch_index=batch.batch_index,
                    batch_key=batch_key,
                    completed_batches=completed_batches,
                )
                os._exit(137)
            planned_samples = materialize_samples(
                batch, params.version, batch_key, spec.scale_power
            )
            samples = [
                Sample(
                    param_version=params.version,
                    commit_seq=planned.commit_seq,
                    batch_id=batch_key,
                    replay_completed_ns=planned.replay_completed_ns,
                    delay_ns=planned.delay_ns,
                    fixed_delay=planned.fixed_delay,
                )
                for planned in planned_samples
            ]
            progress = self.repo.complete_batch(
                param_version=params.version,
                batch_key=batch_key,
                last_replayed_seq=batch.last_seq,
                completed_batches=completed_batches + 1,
                completed_mono_ns=batch.completed_mono_ns,
                is_final=batch.last_seq == len(records),
                last_log_seq=len(records),
                samples=samples,
            )
            completed_batches = progress.completed_batches
            self.logger.event(
                "replay_batch_committed",
                param_version=params.version,
                batch_index=batch.batch_index,
                batch_key=batch_key,
                last_replayed_seq=progress.last_replayed_seq,
                completed_batches=progress.completed_batches,
                status=progress.status,
                sample_count=len(samples),
            )

        final_progress = self.repo.progress_for(params.version)
        if final_progress is None:
            raise RecoveryError("回放结束但进度不存在", field="replay_progress")
        final_samples = self.repo.get_samples(params.version)
        if len(final_samples) != len(records) or final_progress.status != "COMPLETE":
            raise RecoveryError(
                "回放未完整提交",
                field="replay_progress",
                details={
                    "sample_count": len(final_samples),
                    "record_count": len(records),
                    "status": final_progress.status,
                },
            )
        return self._run_summary(params, spec, final_progress, final_samples, resumed=False)

    def _run_summary(
        self,
        params: ReplayParams,
        spec: AggregationSpec,
        progress: Progress,
        samples: Sequence[Sample],
        *,
        resumed: bool,
    ) -> dict[str, Any]:
        return {
            "param_version": params.version,
            "replay_params": params.to_dict(),
            "estimator": spec.to_dict(),
            "progress": progress.to_dict(),
            "sample_count": len(samples),
            "first_commit_seq": samples[0].commit_seq if samples else None,
            "last_commit_seq": samples[-1].commit_seq if samples else None,
            "resumed": resumed,
        }

    def _contract_for_run(self, run: ReplayRun) -> tuple[ReplayParams, AggregationSpec]:
        try:
            spec_raw = json.loads(run.estimator_json)
            spec = validate_aggregation_spec(spec_raw)
            params_raw = json.loads(run.params_json)
            params = validate_replay_params(params_raw, spec)
        except WorkbenchError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ParameterError(
                "持久化回放参数无法解析",
                field="replay_runs",
                details={"param_version": run.param_version, "reason": str(exc)},
            ) from exc
        if spec.digest != run.estimator_digest or params.version != run.param_version:
            raise ParameterError(
                "持久化参数摘要不一致",
                field="param_version",
                details={"param_version": run.param_version},
            )
        return params, spec

    def estimate(self, param_version: str) -> dict[str, Any]:
        run = self.repo.get_run(param_version)
        if run is None:
            raise ParameterError("回放运行不存在", field="param_version", details={"param_version": param_version})
        params, spec = self._contract_for_run(run)
        progress = self.repo.progress_for(param_version)
        if progress is None:
            raise ParameterError("回放进度不存在", field="replay_progress")
        records = self.repo.get_master_log()
        validate_progress(progress.last_replayed_seq, len(records))
        expected_records = records[: progress.last_replayed_seq]
        samples = self.repo.get_samples(param_version)
        sample_inputs = [
            SampleInput(
                commit_seq=sample.commit_seq,
                replay_completed_ns=sample.replay_completed_ns,
                delay_ns=sample.delay_ns,
                fixed_delay=sample.fixed_delay,
            )
            for sample in samples
        ]
        estimation = estimate_windows(expected_records, sample_inputs, params, spec)
        self.logger.event(
            "estimation_computed",
            param_version=param_version,
            result_fingerprint=estimation["result_fingerprint"],
            windows=estimation["window_count"],
            determinable=estimation["determinable_window_count"],
            undecidable=estimation["undecidable_window_count"],
        )
        return {
            "param_version": param_version,
            "replay_params": params.to_dict(),
            "progress": progress.to_dict(),
            "log": self.repo.log_stats(),
            "estimation": estimation,
            "result_fingerprint": estimation["result_fingerprint"],
        }

    def latest_run(self) -> ReplayRun | None:
        runs = self.repo.list_runs()
        if not runs:
            return None
        progress = {item.param_version: item for item in self.repo.list_progress()}
        return sorted(
            runs,
            key=lambda run: (
                progress.get(run.param_version).updated_utc
                if run.param_version in progress
                else run.created_utc,
                run.param_version,
            ),
        )[-1]

    def estimate_latest(self) -> dict[str, Any] | None:
        run = self.latest_run()
        return None if run is None else self.estimate(run.param_version)

    def reset_replay_state(self) -> dict[str, int]:
        counts = self.repo.clear_replay_state()
        self.logger.event("replay_state_cleared", previous_counts=counts)
        return counts

    def save_snapshot(
        self,
        *,
        param_version: str | None = None,
        path: str | Path | None = None,
    ) -> dict[str, Any]:
        run = self.repo.get_run(param_version) if param_version else self.latest_run()
        if run is None:
            raise SnapshotError("没有可保存的回放运行", field="param_version")
        progress = self.repo.progress_for(run.param_version)
        if progress is None or progress.status != "COMPLETE":
            raise SnapshotError(
                "只能保存已完整提交的回放快照",
                field="replay_progress.status",
                details={"status": None if progress is None else progress.status},
            )
        params, spec = self._contract_for_run(run)
        samples = self.repo.get_samples(run.param_version)
        batches = self.repo.get_batches(run.param_version)
        estimation = self.estimate(run.param_version)["estimation"]
        sample_dicts = [sample.to_dict() for sample in samples]
        sequence = snapshot_sequence(sample_dicts)
        document = {
            "snapshot_format_version": SNAPSHOT_FORMAT_VERSION,
            "db_schema_version": SCHEMA_VERSION,
            "created_at_utc": utc_now_iso(),
            "clock_basis": CLOCK_BASIS,
            "truth_source": TRUTH_SOURCE,
            "log": self.repo.log_stats(),
            "run": run.to_dict(),
            "replay_params": {"version": params.version, **params.binding_dict()},
            "estimator": spec.to_dict(),
            "progress": progress.to_dict(),
            "batches": [batch.to_dict() for batch in batches],
            "samples": sample_dicts,
            "sample_sequence": sequence,
            "sample_sequence_digest": sha256_json(sequence),
            "estimation": estimation,
            "estimation_digest": sha256_json(estimation),
        }
        if path is None:
            stamp = utc_now_iso().replace("-", "").replace(":", "").replace(".", "")
            destination = self.snapshot_dir / f"snapshot-{stamp}-{run.param_version}.json"
        else:
            destination = Path(path)
        if destination.exists():
            raise SnapshotError(
                "快照目标已存在；快照采用不可变版本化保存",
                field="snapshot.path",
                details={"path": str(destination)},
            )
        atomic_write_json(destination, document, pretty=True, read_only=True)
        file_digest = sha256_file(destination)
        report = {
            "snapshot_path": str(destination),
            "snapshot_sha256": file_digest,
            "param_version": run.param_version,
            "sample_count": len(samples),
            "sample_sequence_digest": document["sample_sequence_digest"],
            "estimation_digest": document["estimation_digest"],
            "readonly": not is_writable_by_mode(destination),
        }
        self.logger.event("snapshot_saved", **report)
        return report

    def list_snapshots(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for path in sorted(self.snapshot_dir.glob("*.json"), key=lambda item: item.name):
            item: dict[str, Any] = {
                "path": str(path),
                "filename": path.name,
                "size": path.stat().st_size,
                "readonly": not is_writable_by_mode(path),
            }
            try:
                document = read_json(path)
                item.update(
                    {
                        "valid_envelope": isinstance(document, dict),
                        "created_at_utc": document.get("created_at_utc"),
                        "snapshot_format_version": document.get("snapshot_format_version"),
                        "param_version": (document.get("run") or {}).get("param_version"),
                        "sample_count": len(document.get("samples") or []),
                    }
                )
            except Exception as exc:  # noqa: BLE001 - listing must remain stable
                item.update({"valid_envelope": False, "error": str(exc)})
            result.append(item)
        return result

    def restore_snapshot(self, path: str | Path) -> dict[str, Any]:
        document, source_digest = load_snapshot_document(path)
        if document.get("db_schema_version") != SCHEMA_VERSION:
            raise SnapshotError(
                "快照数据库结构版本不兼容",
                field="db_schema_version",
                details={
                    "expected": SCHEMA_VERSION,
                    "actual": document.get("db_schema_version"),
                },
            )
        params, spec = parse_snapshot_contract(document)
        run_raw = document.get("run")
        if not isinstance(run_raw, dict):
            raise SnapshotError("快照 run 字段非法", field="run")
        if run_raw.get("param_version") != params.version:
            raise SnapshotError(
                "快照回放参数版本摘要不一致",
                field="run.param_version",
                details={"expected": params.version, "actual": run_raw.get("param_version")},
            )
        if run_raw.get("estimator_digest") != spec.digest:
            raise SnapshotError(
                "快照估算口径与运行绑定不一致",
                field="run.estimator_digest",
            )

        batches_raw = document.get("batches")
        samples_raw = document.get("samples")
        progress_raw = document.get("progress")
        if not isinstance(batches_raw, list) or not isinstance(samples_raw, list):
            raise SnapshotError("快照批次或样本不是数组", field="samples")
        if not isinstance(progress_raw, dict):
            raise SnapshotError("快照进度字段非法", field="progress")

        batches = self._parse_snapshot_batches(batches_raw, params.version)
        samples = self._parse_snapshot_samples(samples_raw, params.version, spec)
        sequence = snapshot_sequence([sample.to_dict() for sample in samples])
        if document.get("sample_sequence_digest") != sha256_json(sequence):
            raise SnapshotError(
                "快照样本序列摘要不一致",
                field="sample_sequence_digest",
                details={"computed": sha256_json(sequence)},
            )
        if document.get("estimation_digest") != sha256_json(document.get("estimation")):
            raise SnapshotError(
                "快照估算结果摘要不一致", field="estimation_digest"
            )

        try:
            progress = Progress(
                param_version=str(progress_raw["param_version"]),
                last_replayed_seq=int(progress_raw["last_replayed_seq"]),
                last_completed_batch_key=(
                    None
                    if progress_raw.get("last_completed_batch_key") is None
                    else int(progress_raw["last_completed_batch_key"])
                ),
                completed_batches=int(progress_raw["completed_batches"]),
                status=str(progress_raw["status"]),
                updated_utc=str(progress_raw["updated_utc"]),
            )
        except Exception as exc:  # noqa: BLE001
            raise SnapshotError("快照进度无法解析", field="progress", details={"reason": str(exc)}) from exc
        if progress.param_version != params.version:
            raise SnapshotError("快照进度与参数版本不一致", field="progress.param_version")

        log = self.repo.log_stats()
        validate_progress(progress.last_replayed_seq, int(log["last_seq"] or 0))
        log_digest_match = document.get("log", {}).get("digest") == log["digest"]
        status_override = None if log_digest_match else "RESTORED_STALE_LOG_MISMATCH"
        before_counts = self.repo.replay_state_counts()
        before_progress = [item.to_dict() for item in self.repo.list_progress()]
        before_sample_seqs = [sample.commit_seq for sample in self.repo.list_samples()]
        run = ReplayRun(
            param_version=params.version,
            rate_rps=params.rate_rps,
            concurrency=params.concurrency,
            clock_mode=params.clock_mode,
            clock_base_ms=params.clock_base_ms,
            estimator_digest=spec.digest,
            estimator_json=canonical_json(spec.to_dict()),
            params_json=canonical_json(params.binding_dict()),
            created_utc=str(run_raw.get("created_utc") or utc_now_iso()),
        )
        try:
            self.repo.replace_replay_state(
                run=run,
                batches=batches,
                samples=samples,
                progress=progress,
                status_override=status_override,
            )
        except LogValidationError as exc:
            raise SnapshotError(
                "快照进度不得超前当前磁盘日志",
                seq=exc.seq,
                field=exc.field,
                details=exc.details,
            ) from exc

        if is_writable_by_mode(path):
            os.chmod(path, 0o444)
        after_counts = self.repo.replay_state_counts()
        after_samples = self.repo.get_samples(params.version)
        report = {
            "code": None,
            "snapshot_path": str(path),
            "snapshot_sha256": source_digest,
            "snapshot_sha256_after": sha256_file(path),
            "snapshot_retained_readonly": not is_writable_by_mode(path),
            "mode": "WHOLE_REPLACEMENT",
            "merge": False,
            "log_digest_match": log_digest_match,
            "status_override": status_override,
            "requires_historical_replay": not log_digest_match,
            "replaced_fields": [
                {
                    "field": "replay_progress",
                    "before": before_progress,
                    "after": [self.repo.progress_for(params.version).to_dict()]
                    if self.repo.progress_for(params.version)
                    else [],
                },
                {
                    "field": "replay_samples",
                    "before": {
                        "count": len(before_sample_seqs),
                        "first_commit_seq": before_sample_seqs[0] if before_sample_seqs else None,
                        "last_commit_seq": before_sample_seqs[-1] if before_sample_seqs else None,
                    },
                    "after": {
                        "count": len(after_samples),
                        "first_commit_seq": after_samples[0].commit_seq if after_samples else None,
                        "last_commit_seq": after_samples[-1].commit_seq if after_samples else None,
                    },
                },
                {
                    "field": "replay_batches",
                    "before_count": before_counts["replay_batches"],
                    "after_count": after_counts["replay_batches"],
                },
                {
                    "field": "replay_runs",
                    "before_count": before_counts["replay_runs"],
                    "after_count": after_counts["replay_runs"],
                },
            ],
            "remaining_snapshot_files": [item["filename"] for item in self.list_snapshots()],
        }
        self.repo.add_snapshot_restore_audit(source_digest, str(path), report)
        self.logger.event("snapshot_restored", **report)
        return report

    @staticmethod
    def _parse_snapshot_batches(raw_batches: list[Any], param_version: str) -> list[ReplayBatch]:
        result: list[ReplayBatch] = []
        for raw in raw_batches:
            if not isinstance(raw, dict):
                raise SnapshotError("快照批次项必须是对象", field="batches")
            try:
                result.append(
                    ReplayBatch(
                        batch_key=int(raw["batch_key"]),
                        param_version=str(raw["param_version"]),
                        batch_index=int(raw["batch_index"]),
                        attempt=int(raw["attempt"]),
                        first_seq=int(raw["first_seq"]),
                        last_seq=int(raw["last_seq"]),
                        status=str(raw["status"]),
                        started_mono_ns=int(raw["started_mono_ns"]),
                        completed_mono_ns=(
                            None if raw.get("completed_mono_ns") is None else int(raw["completed_mono_ns"])
                        ),
                        started_utc=str(raw["started_utc"]),
                        completed_utc=(
                            None if raw.get("completed_utc") is None else str(raw["completed_utc"])
                        ),
                        notes=str(raw.get("notes", "")),
                    )
                )
            except Exception as exc:  # noqa: BLE001
                raise SnapshotError("快照批次字段非法", field="batches", details={"reason": str(exc)}) from exc
        for batch in result:
            if batch.param_version != param_version:
                raise SnapshotError("快照批次参数版本不一致", field="batches.param_version")
        return sorted(result, key=lambda batch: (batch.batch_index, batch.attempt))

    @staticmethod
    def _parse_snapshot_samples(
        raw_samples: list[Any], param_version: str, spec: AggregationSpec
    ) -> list[Sample]:
        result: list[Sample] = []
        seen: set[int] = set()
        for raw in raw_samples:
            if not isinstance(raw, dict):
                raise SnapshotError("快照样本项必须是对象", field="samples")
            try:
                sample = Sample(
                    param_version=str(raw["param_version"]),
                    commit_seq=int(raw["commit_seq"]),
                    batch_id=int(raw["batch_id"]),
                    replay_completed_ns=int(raw["replay_completed_ns"]),
                    delay_ns=int(raw["delay_ns"]),
                    fixed_delay=int(raw["fixed_delay"]),
                )
            except Exception as exc:  # noqa: BLE001
                raise SnapshotError("快照样本字段非法", field="samples", details={"reason": str(exc)}) from exc
            if sample.param_version != param_version:
                raise SnapshotError("快照样本参数版本不一致", field="samples.param_version")
            if sample.commit_seq in seen:
                raise SnapshotError(
                    "快照样本提交序号重复", seq=sample.commit_seq, field="samples.commit_seq"
                )
            expected_fixed = fixed_from_ns(sample.delay_ns, spec.scale_power)
            if sample.fixed_delay != expected_fixed:
                raise SnapshotError(
                    "快照样本定点口径不一致",
                    seq=sample.commit_seq,
                    field="samples.fixed_delay",
                    details={"expected": expected_fixed, "actual": sample.fixed_delay},
                )
            seen.add(sample.commit_seq)
            result.append(sample)
        return sorted(result, key=lambda sample: sample.commit_seq)

    def verify_snapshot(self, path: str | Path) -> dict[str, Any]:
        try:
            document, source_digest = load_snapshot_document(path)
        except SnapshotError as exc:
            report = {
                "code": "E_REPLAY",
                "reproducible": False,
                "snapshot_path": str(path),
                "reason": "snapshot_envelope_invalid",
                "details": exc.as_dict(),
                "first_mismatch": None,
            }
            self.logger.event("historical_replay_verification", **report)
            return report

        mismatch: dict[str, Any] | None = None
        reason = ""
        params: ReplayParams | None = None
        spec: AggregationSpec | None = None
        try:
            if document.get("db_schema_version") != SCHEMA_VERSION:
                raise ReplayError(
                    "数据库结构版本不一致",
                    field="db_schema_version",
                    details={"expected": SCHEMA_VERSION, "actual": document.get("db_schema_version")},
                )
            params, spec = parse_snapshot_contract(document)
            run_raw = document.get("run") if isinstance(document.get("run"), dict) else {}
            if run_raw.get("param_version") != params.version:
                raise ReplayError("回放参数版本不一致", field="run.param_version")
            if run_raw.get("estimator_digest") != spec.digest:
                raise ReplayError("估算口径摘要不一致", field="run.estimator_digest")
            current_log = self.repo.log_stats()
            if document.get("log", {}).get("digest") != current_log["digest"]:
                raise ReplayError(
                    "快照日志摘要与当前磁盘日志不一致",
                    field="log.digest",
                    details={
                        "snapshot": document.get("log", {}).get("digest"),
                        "current": current_log["digest"],
                    },
                )
            records = self.repo.get_master_log()
            plan = build_replay_plan(records, params, spec.scale_power)
            expected_samples: list[Sample] = []
            for batch in plan:
                planned = materialize_samples(batch, params.version, batch.batch_index + 1, spec.scale_power)
                for item in planned:
                    expected_samples.append(
                        Sample(
                            param_version=params.version,
                            commit_seq=item.commit_seq,
                            batch_id=batch.batch_index + 1,
                            replay_completed_ns=item.replay_completed_ns,
                            delay_ns=item.delay_ns,
                            fixed_delay=item.fixed_delay,
                        )
                    )
            saved_samples = self._parse_snapshot_samples(document.get("samples"), params.version, spec)
            expected_by_seq = {sample.commit_seq: sample for sample in expected_samples}
            saved_by_seq = {sample.commit_seq: sample for sample in saved_samples}
            for seq in sorted(set(expected_by_seq) | set(saved_by_seq)):
                expected_sample = expected_by_seq.get(seq)
                saved_sample = saved_by_seq.get(seq)
                if expected_sample is None:
                    mismatch = {
                        "commit_seq": seq,
                        "field": "sample.extra",
                        "expected": None,
                        "actual": saved_sample.to_dict(),
                    }
                    break
                if saved_sample is None:
                    mismatch = {
                        "commit_seq": seq,
                        "field": "sample.missing",
                        "expected": expected_sample.to_dict(),
                        "actual": None,
                    }
                    break
                for field in ("replay_completed_ns", "delay_ns", "fixed_delay"):
                    left = getattr(expected_sample, field)
                    right = getattr(saved_sample, field)
                    if left != right:
                        mismatch = {
                            "commit_seq": seq,
                            "field": f"sample.{field}",
                            "expected": left,
                            "actual": right,
                        }
                        break
                if mismatch:
                    break

            sample_inputs = [
                SampleInput(
                    commit_seq=sample.commit_seq,
                    replay_completed_ns=sample.replay_completed_ns,
                    delay_ns=sample.delay_ns,
                    fixed_delay=sample.fixed_delay,
                )
                for sample in expected_samples
            ]
            recomputed = estimate_windows(records, sample_inputs, params, spec)
            statistic_diff = deep_first_diff(document.get("estimation"), recomputed)
            if mismatch is None and statistic_diff is not None:
                mismatch = {
                    "commit_seq": None,
                    "field": statistic_diff["path"],
                    "expected": statistic_diff["expected"],
                    "actual": statistic_diff["actual"],
                }
            if mismatch is None:
                sequence = snapshot_sequence([sample.to_dict() for sample in saved_samples])
                if document.get("sample_sequence_digest") != sha256_json(sequence):
                    mismatch = {
                        "commit_seq": None,
                        "field": "sample_sequence_digest",
                        "expected": sha256_json(sequence),
                        "actual": document.get("sample_sequence_digest"),
                    }
        except ReplayError as exc:
            reason = exc.message
            report = {
                "code": "E_REPLAY",
                "reproducible": False,
                "snapshot_path": str(path),
                "snapshot_sha256": source_digest,
                "reason": reason,
                "details": exc.as_dict(),
                "first_mismatch": None,
            }
            self.logger.event("historical_replay_verification", **report)
            return report
        except WorkbenchError as exc:
            report = {
                "code": "E_REPLAY",
                "reproducible": False,
                "snapshot_path": str(path),
                "snapshot_sha256": source_digest,
                "reason": "snapshot_contract_not_reproducible",
                "details": exc.as_dict(),
                "first_mismatch": None,
            }
            self.logger.event("historical_replay_verification", **report)
            return report
        except Exception as exc:  # noqa: BLE001 - non-reproducible for any comparison failure
            report = {
                "code": "E_REPLAY",
                "reproducible": False,
                "snapshot_path": str(path),
                "snapshot_sha256": source_digest,
                "reason": "historical_replay_comparison_failed",
                "details": {"type": type(exc).__name__, "message": str(exc)},
                "first_mismatch": None,
            }
            self.logger.event("historical_replay_verification", **report)
            return report

        reproducible = mismatch is None
        report = {
            "code": None if reproducible else "E_REPLAY",
            "reproducible": reproducible,
            "snapshot_path": str(path),
            "snapshot_sha256": source_digest,
            "param_version": params.version if params else None,
            "sample_count": len(document.get("samples") or []),
            "first_mismatch": mismatch,
            "reason": None if reproducible else "first_difference",
        }
        self.logger.event("historical_replay_verification", **report)
        return report

    def state_summary(self) -> dict[str, Any]:
        runs = self.repo.list_runs()
        progress = self.repo.list_progress()
        samples = self.repo.list_samples()
        latest = self.latest_run()
        estimation = None
        if latest is not None and self.repo.progress_for(latest.param_version) is not None:
            estimation = self.estimate(latest.param_version)
        return {
            "log": self.repo.log_stats(),
            "runs": [run.to_dict() for run in runs],
            "progress": [item.to_dict() for item in progress],
            "sample_counts": {
                version: len([sample for sample in samples if sample.param_version == version])
                for version in sorted({sample.param_version for sample in samples})
            },
            "latest_param_version": None if latest is None else latest.param_version,
            "latest_estimation": estimation,
            "snapshots": self.list_snapshots(),
            "recovery": self.recovery_report,
        }
