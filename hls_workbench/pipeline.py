"""End-to-end workbench orchestration and deterministic reporting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .buffer import build_delay_table, write_delay_table
from .canonical import hash_object, sha256_file, write_canonical
from .config import WorkbenchConfig
from .decision import DecisionEngine, DecisionResult
from .errors import ERROR_CODES
from .hls import write_hls_playlists
from .media import GeneratedMedia, SyntheticMediaGenerator
from .scheduler import BudgetScheduler, SchedulerResult, TranscodeTask
from .transcode import TranscodeArtifact, transcode_segment, validate_transcode_manifest


@dataclass
class PipelineResult:
    root: Path
    media: GeneratedMedia
    scheduler: SchedulerResult
    engine: DecisionEngine
    decision: DecisionResult
    delay_table: dict[str, Any]
    artifacts: list[TranscodeArtifact]
    report: dict[str, Any]
    report_path: Path
    report_sha256: str
    transcode_manifest_path: Path
    transcode_manifest_sha256: str
    delay_table_path: Path
    delay_table_sha256: str
    buffer_paths: dict[str, Path]
    buffer_hashes: dict[str, str]
    decision_path: Path
    decision_sha256: str
    hashes_path: Path
    hashes_sha256: str


def build_tasks(generated: GeneratedMedia, config: WorkbenchConfig) -> list[TranscodeTask]:
    tasks: list[TranscodeTask] = []
    for profile in config.profiles:
        for slice_record in generated.manifest["slices"]:
            sequence = int(slice_record["index"])
            tasks.append(
                TranscodeTask(
                    profile=profile,
                    sequence=sequence,
                    input_video=generated.root / slice_record["video_file"],
                    input_audio=generated.root / slice_record["audio_file"],
                    output_path=generated.root / "hls" / profile.name / f"segment_{sequence:05d}.ts",
                )
            )
    return tasks


def run_pipeline(
    config: WorkbenchConfig,
    root: str | Path,
    *,
    sleep_fn: Callable[[float], None] | None = None,
) -> PipelineResult:
    config.validate()
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)

    generated = SyntheticMediaGenerator(config.media, config.encoder, root_path).generate()
    source_manifest_sha = sha256_file(generated.manifest_path)
    tasks = build_tasks(generated, config)

    scheduler_kwargs: dict[str, Any] = {
        "budget": config.budget,
        "retry": config.retry,
        "fault": config.fault,
        "root": root_path,
    }
    if sleep_fn is not None:
        scheduler_kwargs["sleep_fn"] = sleep_fn
    scheduler = BudgetScheduler(**scheduler_kwargs)

    def execute(task: TranscodeTask, temporary_path: Path) -> TranscodeArtifact:
        return transcode_segment(
            input_video=task.input_video,
            input_audio=task.input_audio,
            output_path=temporary_path,
            profile=task.profile,
            sequence=task.sequence,
            media=config.media,
            encoder=config.encoder,
        )

    scheduler_result = scheduler.run(tasks, execute)
    artifacts = sorted(
        scheduler_result.artifacts,
        key=lambda artifact: (
            next(profile.priority for profile in config.profiles if profile.name == artifact.profile),
            artifact.sequence,
        ),
    )
    timeline_checks = validate_transcode_manifest(artifacts, config.profiles, config.media, root_path)
    playlist_hashes = write_hls_playlists(
        root=root_path,
        profiles=config.profiles,
        media=config.media,
        artifacts=artifacts,
    )

    transcode_manifest = {
        "schema_version": 1,
        "requested_parallelism": scheduler_result.requested_parallelism,
        "effective_parallelism": scheduler_result.effective_parallelism,
        "budget": config.budget.to_dict(),
        "retry": config.retry.to_dict(),
        "fault": config.fault.to_dict(),
        "artifacts": [artifact.to_dict() for artifact in artifacts],
        "ready_times_ms": scheduler_result.ready_times(),
        "degradations": list(scheduler_result.degradations),
        "budget_denials": list(scheduler_result.budget_denials),
        "retry_events": list(scheduler_result.retry_events),
        "isolation_events": list(scheduler_result.isolation_events),
        "reassignment_events": list(scheduler_result.reassignment_events),
        "timeline_checks": timeline_checks,
    }
    reports_dir = root_path / "reports"
    transcode_manifest_path = reports_dir / "transcode_manifest.json"
    transcode_manifest_sha = write_canonical(transcode_manifest_path, transcode_manifest)

    delay_table = build_delay_table(config.profiles, config.media, config.media.seed)
    delay_table_path = root_path / "input" / "pseudo_download_delays.json"
    delay_table_sha = write_delay_table(delay_table_path, delay_table)

    segment_durations_ms = [
        int(record["duration_ms"]) for record in generated.manifest["slices"]
    ]
    engine = DecisionEngine(
        profiles=config.profiles,
        ready_times_ms=scheduler_result.ready_times(),
        delay_table=delay_table,
        segment_durations_ms=segment_durations_ms,
        media_duration_ms=int(round(config.media.duration_seconds * 1000)),
        buffer_config=config.client_buffer,
        budget=config.budget,
        bandwidth_safety_ratio=config.decision.bandwidth_safety_ratio,
    )
    decision = engine.decide(
        bandwidth_kbps=config.decision.bandwidth_kbps,
        threshold_ms=config.client_buffer.threshold_ms,
    )

    buffer_paths: dict[str, Path] = {}
    buffer_hashes: dict[str, str] = {}
    buffer_summaries: dict[str, Any] = {}
    for profile in sorted(config.profiles, key=lambda item: item.priority):
        model = engine.buffer_model(profile.name, config.client_buffer.threshold_ms)
        model_path = reports_dir / f"buffer_{profile.name}.json"
        buffer_hash = write_canonical(model_path, model.to_dict())
        buffer_paths[profile.name] = model_path
        buffer_hashes[profile.name] = buffer_hash
        buffer_summaries[profile.name] = {
            "minimum_buffer_ms": model.minimum_buffer_ms,
            "peak_buffer_ms": model.peak_buffer_ms,
            "underrun_count": model.underrun_count,
            "switch_recommendation_count": len(model.switch_recommendations),
            "safe": model.safe,
            "playback_start_ms": model.playback_start_ms,
            "playback_end_ms": model.playback_end_ms,
        }

    decision_path = reports_dir / "decision.json"
    decision_sha = write_canonical(decision_path, decision.to_dict())

    artifact_hashes = {
        artifact.relative_path: artifact.sha256 for artifact in artifacts
    }
    synthesis_step = {
        "status": "ok",
        "source_manifest": generated.manifest_path.relative_to(root_path).as_posix(),
        "source_manifest_sha256": source_manifest_sha,
        "source_video": generated.source_video.relative_to(root_path).as_posix(),
        "source_video_sha256": sha256_file(generated.source_video),
        "source_audio": generated.source_audio.relative_to(root_path).as_posix(),
        "source_audio_sha256": sha256_file(generated.source_audio),
        "slice_count": config.media.slice_count,
        "frame_count": config.media.total_frames,
        "audio_samples": config.media.total_audio_samples,
        "content_sha256": generated.manifest["source"]["content_sha256"],
    }
    transcode_step = {
        "status": "ok",
        "manifest": transcode_manifest_path.relative_to(root_path).as_posix(),
        "manifest_sha256": transcode_manifest_sha,
        "artifact_count": len(artifacts),
        "artifact_hashes": artifact_hashes,
        "requested_parallelism": scheduler_result.requested_parallelism,
        "effective_parallelism": scheduler_result.effective_parallelism,
        "degradation_count": len(scheduler_result.degradations),
        "budget_denial_count": len(scheduler_result.budget_denials),
        "retry_event_count": len(scheduler_result.retry_events),
        "isolation_count": len(scheduler_result.isolation_events),
        "reassignment_count": len(scheduler_result.reassignment_events),
        "timeline_check_count": len(timeline_checks),
        "playlist_hashes": playlist_hashes,
        "content_sha256": hash_object(transcode_manifest),
    }
    buffer_step = {
        "status": "ok",
        "delay_table": delay_table_path.relative_to(root_path).as_posix(),
        "delay_table_sha256": delay_table_sha,
        "profiles": buffer_summaries,
        "model_hashes": buffer_hashes,
        "content_sha256": hash_object(buffer_summaries),
    }
    decision_step = {
        "status": "ok" if decision.safe else "risk",
        "decision_file": decision_path.relative_to(root_path).as_posix(),
        "decision_sha256": decision_sha,
        "decision": decision.to_dict(),
        "content_sha256": decision_sha,
    }
    report = {
        "schema_version": 1,
        "config": config.to_dict(),
        "error_codes": ERROR_CODES,
        "steps": {
            "synthesis": synthesis_step,
            "transcode": transcode_step,
            "buffer": buffer_step,
            "decision": decision_step,
        },
        "hash_chain": {
            "synthesis": synthesis_step["content_sha256"],
            "transcode": transcode_step["content_sha256"],
            "buffer": buffer_step["content_sha256"],
            "decision": decision_sha,
        },
        "reports": {
            "transcode_manifest_sha256": transcode_manifest_sha,
            "buffer_model_sha256": buffer_hashes,
            "decision_sha256": decision_sha,
        },
    }
    report_path = reports_dir / "report.json"
    report_sha = write_canonical(report_path, report)

    hashes = {
        "source_manifest_sha256": source_manifest_sha,
        "transcode_manifest_sha256": transcode_manifest_sha,
        "delay_table_sha256": delay_table_sha,
        "buffer_model_sha256": buffer_hashes,
        "decision_sha256": decision_sha,
        "artifacts": artifact_hashes,
        "report_sha256": report_sha,
    }
    hashes_path = reports_dir / "hashes.json"
    hashes_sha = write_canonical(hashes_path, hashes)

    return PipelineResult(
        root=root_path,
        media=generated,
        scheduler=scheduler_result,
        engine=engine,
        decision=decision,
        delay_table=delay_table,
        artifacts=artifacts,
        report=report,
        report_path=report_path,
        report_sha256=report_sha,
        transcode_manifest_path=transcode_manifest_path,
        transcode_manifest_sha256=transcode_manifest_sha,
        delay_table_path=delay_table_path,
        delay_table_sha256=delay_table_sha,
        buffer_paths=buffer_paths,
        buffer_hashes=buffer_hashes,
        decision_path=decision_path,
        decision_sha256=decision_sha,
        hashes_path=hashes_path,
        hashes_sha256=hashes_sha,
    )
