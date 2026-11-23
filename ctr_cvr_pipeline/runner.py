from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .cache import ArtifactCache
from .canonical import fingerprint
from .commands import apply_command, load_commands
from .config import FeatureConfig, TrainingConfig, load_feature_config, load_training_config
from .pipeline import analyze_state, report_core
from .samples import load_samples_jsonl
from .snapshot import SnapshotStore
from .state import PipelineState, initial_state


@dataclass(frozen=True)
class FixturePaths:
    samples: str
    feature_config: str
    training_config: str
    commands: str


@dataclass
class LoadedFixtures:
    feature_config: FeatureConfig
    training_config: TrainingConfig
    base_samples: list[Any]
    base_failures: list[Any]
    commands: list[dict[str, Any]]

    def fresh_state(self) -> PipelineState:
        return initial_state(
            self.base_samples,
            self.feature_config,
            self.training_config,
            self.base_failures,
        )


def load_fixtures(paths: FixturePaths) -> LoadedFixtures:
    feature_config = load_feature_config(paths.feature_config)
    training_config = load_training_config(paths.training_config)
    samples, failures = load_samples_jsonl(
        paths.samples,
        feature_config,
        event_time_ms=training_config.initial_event_time_ms,
    )
    commands = load_commands(paths.commands)
    return LoadedFixtures(
        feature_config=feature_config,
        training_config=training_config,
        base_samples=samples,
        base_failures=failures,
        commands=commands,
    )


def _command_metadata(commands: list[dict[str, Any]], index: int) -> tuple[int, str | None]:
    if index < 0 or index >= len(commands):
        return -1, None
    return index, commands[index].get("type")


def replay_incremental(
    fixtures: LoadedFixtures,
    output_dir: str | Path,
    *,
    workers: int | None = None,
    clean: bool = True,
    write_snapshots: bool = True,
) -> dict[str, Any]:
    output = Path(output_dir)
    store = SnapshotStore(output, clean=clean) if write_snapshots else None
    cache = ArtifactCache(output / "cache", collect_ledger=True)
    state = fixtures.fresh_state()
    reports: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index in range(len(fixtures.commands)):
        apply_command(state, fixtures.commands[index], index, fixtures.feature_config)
        command_index, command_type = _command_metadata(fixtures.commands, index)
        report = analyze_state(
            state,
            fixtures.feature_config,
            fixtures.training_config,
            cache,
            mode="incremental",
            command_index=command_index,
            command_type=command_type,
            workers=workers,
        )
        reports.append(report)
        if store is not None:
            store.commit(index, report)
    if len(fixtures.commands) == 0:
        report = analyze_state(
            state,
            fixtures.feature_config,
            fixtures.training_config,
            cache,
            mode="incremental",
            command_index=-1,
            command_type=None,
            workers=workers,
        )
        reports.append(report)
        if store is not None:
            store.commit(0, report)
    elapsed = time.perf_counter() - started
    comparison = {
        "mode": "incremental",
        "step_count": len(reports),
        "elapsed_seconds": elapsed,
        "core_digests": [report["deterministic_core_digest"] for report in reports],
        "workers": workers if workers is not None else fixtures.training_config.thread_pool_workers,
    }
    if store is not None:
        store.finalize({"run": comparison})
    return {"reports": reports, "summary": comparison, "output_dir": str(output)}


def replay_full(
    fixtures: LoadedFixtures,
    output_dir: str | Path,
    *,
    workers: int | None = None,
    clean: bool = True,
    write_snapshots: bool = True,
) -> dict[str, Any]:
    output = Path(output_dir)
    store = SnapshotStore(output, clean=clean) if write_snapshots else None
    reports: list[dict[str, Any]] = []
    started = time.perf_counter()
    steps = max(1, len(fixtures.commands))
    for step in range(steps):
        state = fixtures.fresh_state()
        last_index = step if fixtures.commands else -1
        for index in range(0, last_index + 1):
            apply_command(state, fixtures.commands[index], index, fixtures.feature_config)
        with tempfile.TemporaryDirectory(prefix="ctr-cvr-full-cache-") as temp_cache:
            report = analyze_state(
                state,
                fixtures.feature_config,
                fixtures.training_config,
                ArtifactCache(temp_cache, collect_ledger=True),
                mode="full",
                command_index=last_index,
                command_type=fixtures.commands[last_index].get("type") if last_index >= 0 else None,
                workers=workers,
            )
        reports.append(report)
        if store is not None:
            store.commit(step, report)
    elapsed = time.perf_counter() - started
    comparison = {
        "mode": "full",
        "step_count": len(reports),
        "elapsed_seconds": elapsed,
        "core_digests": [report["deterministic_core_digest"] for report in reports],
        "workers": workers if workers is not None else fixtures.training_config.thread_pool_workers,
    }
    if store is not None:
        store.finalize({"run": comparison})
    return {"reports": reports, "summary": comparison, "output_dir": str(output)}


def verify_consistency(
    fixtures: LoadedFixtures,
    output_dir: str | Path,
    *,
    clean: bool = True,
) -> dict[str, Any]:
    output = Path(output_dir)
    if clean and output.exists():
        import shutil

        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    incremental = replay_incremental(
        fixtures, output / "incremental", clean=True, write_snapshots=True
    )
    full = replay_full(fixtures, output / "full", clean=True, write_snapshots=True)
    serial = replay_incremental(
        fixtures,
        output / "serial_workers_1",
        workers=1,
        clean=True,
        write_snapshots=False,
    )
    concurrent = replay_incremental(
        fixtures,
        output / "concurrent_workers",
        workers=fixtures.training_config.thread_pool_workers,
        clean=True,
        write_snapshots=False,
    )
    steps = min(
        len(incremental["reports"]),
        len(full["reports"]),
        len(serial["reports"]),
        len(concurrent["reports"]),
    )
    lengths = {
        "incremental": len(incremental["reports"]),
        "full": len(full["reports"]),
        "serial_workers_1": len(serial["reports"]),
        "concurrent_workers": len(concurrent["reports"]),
    }
    mismatches: list[dict[str, Any]] = []
    for index in range(steps):
        candidates = {
            "incremental": incremental["reports"][index],
            "full": full["reports"][index],
            "serial_workers_1": serial["reports"][index],
            "concurrent_workers": concurrent["reports"][index],
        }
        signatures = {name: fingerprint(report_core(report)) for name, report in candidates.items()}
        if len(set(signatures.values())) != 1:
            mismatches.append({"step": index, "signatures": signatures})
    serial_concurrent_same = all(
        serial["reports"][index]["deterministic_core_digest"]
        == concurrent["reports"][index]["deterministic_core_digest"]
        for index in range(steps)
    )
    summary = {
        "consistent": not mismatches and len(set(lengths.values())) == 1,
        "step_count": steps,
        "step_counts": lengths,
        "mismatches": mismatches,
        "serial_concurrent_consistent": serial_concurrent_same,
        "incremental_digests": incremental["summary"]["core_digests"],
        "full_digests": full["summary"]["core_digests"],
        "final_core_digest": incremental["reports"][-1]["deterministic_core_digest"] if steps else None,
        "final_reused_count": incremental["reports"][-1]["cache"]["reused_count"] if steps else 0,
        "final_recomputed_count": incremental["reports"][-1]["cache"]["recomputed_count"] if steps else 0,
        "final_failures": incremental["reports"][-1]["failures"] if steps else [],
        "final_stage_summary": incremental["reports"][-1]["stage_summary"] if steps else {},
        "final_quantization_flip_count": len(incremental["reports"][-1]["quantization_flips"]) if steps else 0,
        "final_reference_flip_count": len(incremental["reports"][-1]["reference_flips"]) if steps else 0,
        "elapsed_seconds": {
            "incremental": incremental["summary"]["elapsed_seconds"],
            "full": full["summary"]["elapsed_seconds"],
            "serial_workers_1": serial["summary"]["elapsed_seconds"],
            "concurrent_workers": concurrent["summary"]["elapsed_seconds"],
        },
        "output_dir": str(output),
    }
    from .io_utils import atomic_write_json

    atomic_write_json(output / "verification.json", summary)
    return summary
