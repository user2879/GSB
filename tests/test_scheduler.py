from __future__ import annotations

from pathlib import Path

import pytest

from hls_workbench.canonical import sha256_file
from hls_workbench.config import BudgetConfig, DEFAULT_PROFILES, FaultConfig, RetryConfig
from hls_workbench.errors import WorkbenchError
from hls_workbench.scheduler import BudgetScheduler, TranscodeTask
from hls_workbench.transcode import TranscodeArtifact


def make_tasks(root: Path, sequences: int = 2) -> list[TranscodeTask]:
    return [
        TranscodeTask(
            profile=profile,
            sequence=sequence,
            input_video=root / "input.ts",
            input_audio=root / "input.wav",
            output_path=root / "out" / profile.name / f"segment_{sequence:05d}.ts",
        )
        for profile in DEFAULT_PROFILES
        for sequence in range(sequences)
    ]


def fake_executor(task: TranscodeTask, temporary_path: Path) -> TranscodeArtifact:
    temporary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path.write_bytes(b"deterministic-" + task.key.encode("ascii"))
    return TranscodeArtifact(
        profile=task.profile.name,
        sequence=task.sequence,
        relative_path="",
        sha256=sha256_file(temporary_path),
        file_size=temporary_path.stat().st_size,
        decoded_frame_count=30,
        encoded_frame_count=30,
        audio_frame_count=48,
        start_pts=task.sequence * 30,
        end_pts=task.sequence * 30 + 30,
        start_sample=task.sequence * 48000,
        audio_samples=48000,
    )


def test_budget_denials_retries_isolation_and_reassignment(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    tasks = make_tasks(root)
    scheduler = BudgetScheduler(
        budget=BudgetConfig(max_workers=3, memory_budget_mb=180, cpu_budget_ms=180),
        retry=RetryConfig(max_retries=2, base_delay_ms=1),
        fault=FaultConfig(task_key="720p:00001", failures=3),
        root=root,
        sleep_fn=lambda _: None,
    )
    result = scheduler.run(tasks, fake_executor)

    assert len(result.outcomes) == 4
    assert [outcome.task.key for outcome in result.outcomes] == sorted(
        (task.key for task in tasks), key=lambda key: (key.startswith("720p"), key)
    )
    assert all(outcome.artifact.relative_path for outcome in result.outcomes)
    assert all((root / outcome.artifact.relative_path).exists() for outcome in result.outcomes)
    assert result.budget_denials
    assert all(item["code"] == "E_BUDGET_DENIED" for item in result.budget_denials)
    assert result.degradations
    assert all(item["before_parallelism"] >= item["after_parallelism"] for item in result.degradations)
    assert any(item["kind"] == "retry_exhausted" for item in result.retry_events)
    assert len(result.isolation_events) == 1
    assert len(result.reassignment_events) == 1
    retry = next(outcome for outcome in result.outcomes if outcome.task.key == "720p:00001")
    assert retry.attempts >= 4
    assert retry.units_tried[0] != retry.units_tried[-1]
    assert [event["delay_ms"] for event in result.retry_events if event["kind"] == "retry_backoff"] == [1, 2]


def test_duplicate_task_key_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "duplicate"
    root.mkdir()
    tasks = make_tasks(root, sequences=1)
    tasks.append(tasks[0])
    scheduler = BudgetScheduler(
        budget=BudgetConfig(max_workers=2, memory_budget_mb=300, cpu_budget_ms=300),
        retry=RetryConfig(max_retries=0),
        fault=FaultConfig(),
        root=root,
        sleep_fn=lambda _: None,
    )
    with pytest.raises(WorkbenchError) as caught:
        scheduler.run(tasks, fake_executor)
    assert caught.value.code == "E_TASK_DUPLICATE"


def test_budget_below_single_task_falls_back_to_traceable_serial(tmp_path: Path) -> None:
    root = tmp_path / "serial-fallback"
    root.mkdir()
    tasks = make_tasks(root, sequences=1)
    scheduler = BudgetScheduler(
        budget=BudgetConfig(max_workers=2, memory_budget_mb=1, cpu_budget_ms=1),
        retry=RetryConfig(max_retries=0),
        fault=FaultConfig(),
        root=root,
        sleep_fn=lambda _: None,
    )
    result = scheduler.run(tasks, fake_executor)
    assert len(result.outcomes) == 2
    assert result.effective_parallelism == 1
    assert all((root / outcome.artifact.relative_path).exists() for outcome in result.outcomes)
    assert all(item["code"] == "E_BUDGET_DENIED" for item in result.budget_denials)
    assert any(item["forced_serial_override"] for item in result.degradations)
