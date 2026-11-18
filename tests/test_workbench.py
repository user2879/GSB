from __future__ import annotations

import hashlib
import json
import os
import random
import signal
import subprocess
import sys
import time
from dataclasses import replace
from decimal import Decimal, ROUND_HALF_EVEN
from pathlib import Path

import pytest
import yaml

from feature_workbench.canonical import canonical_json, decimal_text
from feature_workbench.cli import (
    build_full_from_final,
    compare_projections,
    execute_incremental,
)
from feature_workbench.fixedpoint import decimal_to_fixed
from feature_workbench.models import failure_sort_key


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def run_cli(*args: str | Path, timeout: float = 40.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [PYTHON, "-m", "feature_workbench", *[str(item) for item in args]],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )


def generate_sample(root: Path, decimal_shift: int = 0) -> Path:
    sample_dir = root / "samples"
    completed = subprocess.run(
        [
            PYTHON,
            "scripts/generate_samples.py",
            "--output-dir",
            str(sample_dir),
            "--decimal-shift",
            str(decimal_shift),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stderr
    return sample_dir


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sample_paths(sample_dir: Path) -> dict[str, Path]:
    return {
        "definitions": sample_dir / "definitions.json",
        "events": sample_dir / "events.jsonl",
        "config": sample_dir / "scoring.yaml",
        "edits": sample_dir / "edits.json",
    }


def run_sample(
    sample_dir: Path,
    root: Path,
    *,
    config: Path | None = None,
    edits: Path | None = None,
    name: str = "run",
    extra: list[str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], dict]:
    paths = sample_paths(sample_dir)
    report_path = root / f"{name}_report.json"
    state_dir = root / f"{name}_state"
    args: list[str | Path] = [
        "run",
        "--definitions",
        paths["definitions"],
        "--events",
        paths["events"],
        "--config",
        config or paths["config"],
        "--edits",
        edits or paths["edits"],
        "--state-dir",
        state_dir,
        "--report",
        report_path,
    ]
    if extra:
        args.extend(extra)
    completed = run_cli(*args)
    return completed, load_json(report_path)


def final_events(sample_dir: Path) -> list[dict]:
    events = [
        json.loads(line)
        for line in (sample_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {item["event_id"]: item for item in events}
    plan = load_json(sample_dir / "edits.json")
    for batch in plan["batches"]:
        for event in batch.get("appends", []):
            events.append(event)
            by_id[event["event_id"]] = event
        for backfill in batch.get("backfills", []):
            event_id = backfill.get("event_id")
            if event_id is None:
                continue
            old = by_id[event_id]
            replacement = json.loads(json.dumps(old))
            for key, value in backfill.get("patch", {}).items():
                if key == "context":
                    replacement.setdefault("context", {}).update(value)
                else:
                    replacement[key] = value
            by_id[event_id] = replacement
            events = [replacement if item["event_id"] == event_id else item for item in events]
    return sorted(events, key=lambda item: (item["timestamp_ms"], item["event_id"]))


def final_definitions(sample_dir: Path) -> dict[str, dict]:
    definitions = {
        item["name"]: item
        for item in load_json(sample_dir / "definitions.json")["features"]
    }
    plan = load_json(sample_dir / "edits.json")
    for batch in plan["batches"]:
        for change in batch.get("definition_changes", []):
            if change.get("op", "upsert") == "upsert":
                definition = change["definition"]
                definitions[definition["name"]] = definition
            elif change.get("op") == "remove":
                definitions.pop(change.get("name"), None)
    return definitions


def expected_fixed(value: Decimal, decimal_places: int, quantum_text: str) -> int:
    quantum = Decimal(quantum_text)
    units = (value / quantum).to_integral_value(rounding=ROUND_HALF_EVEN)
    return int(units * quantum.scaleb(decimal_places))


@pytest.mark.parametrize("decimal_shift", [0, 2])
def test_cli_diagnose_full_incremental_equivalence(tmp_path: Path, decimal_shift: int) -> None:
    sample_dir = generate_sample(tmp_path, decimal_shift)
    report_path = tmp_path / "diagnostic.json"
    completed = run_cli(
        "diagnose",
        "--sample-dir",
        sample_dir,
        "--work-dir",
        tmp_path / "work",
        "--report",
        report_path,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    report = load_json(report_path)
    assert report["all_checks_passed"] is True
    assert report["equivalence"]["equal"] is True
    assert report["equivalence"]["diff_paths"] == []
    assert report["equivalence"]["incremental_fingerprint"] == report["equivalence"]["full_fingerprint"]
    assert {item["concurrency"] for item in report["concurrency_equivalence"]} == {1, 4, 8}
    assert all(item["equal_to_full"] for item in report["concurrency_equivalence"])
    assert report["injections"]["ran"] is True
    assert report["injections"]["all_hit"] is True
    assert all(item["hit"] for item in report["injections"]["hits"])
    assert len(report["sample"]["decimal_places_seen"]) >= 3
    assert report["incremental"]["pending_count"] == 0
    assert report["full_rebuild"]["pending_count"] == 0


def test_precise_invalidation_scope_and_fixed_point_values(tmp_path: Path) -> None:
    sample_dir = generate_sample(tmp_path)
    completed, report = run_sample(sample_dir, tmp_path)
    assert completed.returncode == 0, completed.stderr
    assert report["status"] == "ok"
    assert report["failures"] == []
    assert report["pending"] == []
    assert report["event_count"] == 49
    assert len(report["feature_graph"]) == 8
    assert report["feature_graph"]["user_value_composite_10m"]["upstream_features"] == [
        "user_amount_sum_10m",
        "user_risk_latest_5m",
    ]

    summaries = {item["batch_id"]: item for item in report["batch_summaries"]}
    backfill = summaries["backfill-amount"]
    assert backfill["force_all"] is False
    assert backfill["recomputed_feature_count"] == 8
    assert backfill["reused_feature_count"] == 48
    assert backfill["recomputed_score_count"] == 2
    assert backfill["reused_score_count"] == 12
    assert all(
        unit.endswith("|u-001") or unit.endswith("|i-001")
        for unit in backfill["recomputed_features"]
    )
    definition_change = summaries["change-composite-precision"]
    assert definition_change["recomputed_feature_count"] == 8
    assert definition_change["reused_feature_count"] == 48
    assert definition_change["invalidated_features"] == ["user_value_composite_10m"]

    events = final_events(sample_dir)
    definitions = final_definitions(sample_dir)
    cursor = events[-1]["timestamp_ms"]
    features = {
        (item["feature_name"], item["side"], item["entity_id"]): item
        for item in report["features"]
    }
    user_id = "u-001"
    count_definition = definitions["user_event_count_5m"]
    count_events = [
        event
        for event in events
        if event["user_id"] == user_id
        and cursor - count_definition["window_ms"] <= event["timestamp_ms"] <= cursor
    ]
    count_record = features[("user_event_count_5m", "user", user_id)]
    assert count_record["event_ids"] == [event["event_id"] for event in count_events]
    assert count_record["value_fixed"] == expected_fixed(
        Decimal(len(count_events)),
        count_definition["decimal_places"],
        count_definition["quantization_step"],
    )

    sum_definition = definitions["user_amount_sum_10m"]
    sum_events = [
        event
        for event in events
        if event["user_id"] == user_id
        and cursor - sum_definition["window_ms"] <= event["timestamp_ms"] <= cursor
    ]
    amount_sum = sum(Decimal(str(event["context"]["amount"])) for event in sum_events)
    sum_record = features[("user_amount_sum_10m", "user", user_id)]
    assert sum_record["value_fixed"] == expected_fixed(
        amount_sum,
        sum_definition["decimal_places"],
        sum_definition["quantization_step"],
    )
    assert sum_record["event_ids"] == [event["event_id"] for event in sum_events]

    # Scalar rounding follows deterministic half-even; ties round to an even quantum.
    assert decimal_to_fixed(Decimal("0.00005"), 4, Decimal("0.0001")) == 0
    assert decimal_to_fixed(Decimal("0.00015"), 4, Decimal("0.0001")) == 2

    feature_by_entity: dict[tuple[str, str], list[dict]] = {}
    for item in report["features"]:
        feature_by_entity.setdefault((item["side"], item["entity_id"]), []).append(item)
    for score in report["scores"]:
        entity_values = sorted(
            feature_by_entity[(score["entity_type"], score["entity_id"])],
            key=lambda item: item["feature_name"],
        )
        assert [item["feature_name"] for item in score["contributions"]] == [
            item["feature_name"] for item in entity_values
        ]
        total = Decimal(score["bias_contribution"])
        for contribution in score["contributions"]:
            total += Decimal(contribution["contribution_decimal"])
        total = min(max(total, Decimal("-50")), Decimal("50"))
        expected = expected_fixed(total, score["score_decimal_places"], score["score_quantum"])
        assert score["score_fixed"] == expected
        event_ids = {
            event_id
            for item in entity_values
            for event_id in item["event_ids"]
        }
        assert score["participating_event_count"] == len(event_ids)
    ranking = [
        (-item["score_fixed"], -item["participating_event_count"], item["entity_id"], item["entity_type"])
        for item in report["scores"]
    ]
    assert ranking == sorted(ranking)


def test_snapshot_resume_and_config_mismatch(tmp_path: Path) -> None:
    sample_dir = generate_sample(tmp_path)
    first, first_report = run_sample(sample_dir, tmp_path, name="resume")
    assert first.returncode == 0
    assert first_report["resumed"] is False
    assert first_report["applied_batches"] == 4
    state_dir = tmp_path / "resume_state"
    manifest = load_json(state_dir / "manifest.json")
    stage_dir = state_dir / manifest["snapshot_dir"]
    for name in ("load", "features", "scores", "progress"):
        path = stage_dir / f"{name}.json"
        assert path.exists()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["artifact_hashes"][name]
    progress = load_json(stage_dir / "progress.json")
    assert progress["global_version"] == manifest["global_version"]
    assert progress["applied_batches"] == 4
    assert progress["event_cursor"] == first_report["event_cursor"]

    second, second_report = run_sample(sample_dir, tmp_path, name="resume")
    assert second.returncode == 0
    assert second_report["resumed"] is True
    assert second_report["applied_batches"] == 4
    assert second_report["features"] == first_report["features"]
    assert second_report["scores"] == first_report["scores"]

    config_path = sample_dir / "scoring.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["concurrency"] = 7
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    third, third_report = run_sample(sample_dir, tmp_path, name="resume")
    assert third.returncode == 0
    assert third_report["resumed"] is False
    startup_failures = third_report["startup_failures"]
    assert startup_failures
    assert startup_failures[0]["code"] == "E_CONFIG_MISMATCH"
    assert "config_hash" in startup_failures[0]["detail"]["mismatched_fields"]
    assert any(item["code"] == "E_CONFIG_MISMATCH" for item in third_report["failures"])


def test_budget_pending_is_retained_and_catches_up(tmp_path: Path) -> None:
    sample_dir = generate_sample(tmp_path)
    plan = load_json(sample_dir / "edits.json")
    for index in range(30):
        plan["batches"].append(
            {
                "batch_id": f"catchup-{index + 1:02d}",
                "appends": [],
                "backfills": [],
                "definition_changes": [],
            }
        )
    extended_plan = tmp_path / "extended_plan.json"
    extended_plan.write_text(json.dumps(plan), encoding="utf-8")
    completed, report = run_sample(
        sample_dir,
        tmp_path,
        config=sample_dir / "scoring_low_budget.yaml",
        edits=extended_plan,
        name="budget",
    )
    assert completed.returncode == 0, completed.stderr
    assert report["pending"] == []
    assert report["status"] == "ok"
    assert any(
        item["pending_feature_count"] > 0 or item["pending_score_count"] > 0
        for item in report["batch_summaries"]
    )
    assert any(
        item["pending_feature_count"] > 0 for item in report["batch_summaries"]
    )

    # Compare the caught-up budgeted run against a one-shot full rebuild in-process.
    paths = sample_paths(sample_dir)
    _, workbench, config = execute_incremental(
        definitions_path=paths["definitions"],
        events_path=paths["events"],
        config_path=sample_dir / "scoring_low_budget.yaml",
        edits_path=extended_plan,
        state_dir=tmp_path / "budget_compare_state",
        report_path=tmp_path / "budget_compare_report.json",
    )
    assert workbench is not None and config is not None
    full = build_full_from_final(
        workbench, replace(config, point_budget_per_batch=10**9)
    )
    comparison = compare_projections(workbench, full)
    assert comparison["equal"] is True
    assert comparison["diff_paths"] == []


def _wait_for_manifest(state_dir: Path, timeout: float = 10.0) -> dict:
    manifest_path = state_dir / "manifest.json"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if manifest_path.exists():
            try:
                manifest = load_json(manifest_path)
            except (OSError, json.JSONDecodeError):
                pass
            else:
                if isinstance(manifest.get("applied_batches"), int) and manifest["applied_batches"] >= 1:
                    return manifest
        time.sleep(0.03)
    raise AssertionError("snapshot manifest did not reach at least one committed batch")


def _run_process(sample_dir: Path, root: Path, *, name: str, delay_ms: int) -> tuple[subprocess.Popen[str], Path, Path]:
    paths = sample_paths(sample_dir)
    report_path = root / f"{name}_report.json"
    state_dir = root / f"{name}_state"
    process = subprocess.Popen(
        [
            PYTHON,
            "-m",
            "feature_workbench",
            "run",
            "--definitions",
            str(paths["definitions"]),
            "--events",
            str(paths["events"]),
            "--config",
            str(paths["config"]),
            "--edits",
            str(paths["edits"]),
            "--state-dir",
            str(state_dir),
            "--report",
            str(report_path),
            "--batch-delay-ms",
            str(delay_ms),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return process, report_path, state_dir


def test_graceful_signal_finishes_batch_and_exits_zero(tmp_path: Path) -> None:
    sample_dir = generate_sample(tmp_path)
    process, report_path, state_dir = _run_process(
        sample_dir, tmp_path, name="signal", delay_ms=700
    )
    try:
        _wait_for_manifest(state_dir)
        process.send_signal(signal.SIGINT)
        stdout, stderr = process.communicate(timeout=20)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
    assert process.returncode == 0, stderr + stdout
    report = load_json(report_path)
    assert report["status"] == "stopped"
    assert report["signals_received"] == [signal.SIGINT]
    assert report["applied_batches"] >= 1
    manifest = load_json(state_dir / "manifest.json")
    assert manifest["status"] == "stopped"
    assert manifest["applied_batches"] == report["applied_batches"]
    progress = load_json(state_dir / manifest["snapshot_dir"] / "progress.json")
    assert progress["applied_batches"] == manifest["applied_batches"]


def test_forced_kill_recovers_only_from_complete_snapshot(tmp_path: Path) -> None:
    sample_dir = generate_sample(tmp_path)
    process, _, state_dir = _run_process(
        sample_dir, tmp_path, name="kill", delay_ms=700
    )
    _wait_for_manifest(state_dir)
    process.kill()
    process.communicate(timeout=10)
    manifest_before = load_json(state_dir / "manifest.json")
    stage_dir = state_dir / manifest_before["snapshot_dir"]
    for name in ("load", "features", "scores", "progress"):
        path = stage_dir / f"{name}.json"
        assert path.exists()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest_before["artifact_hashes"][name]

    resumed, resumed_report = run_sample(sample_dir, tmp_path, name="kill")
    assert resumed.returncode == 0, resumed.stderr
    assert resumed_report["resumed"] is True
    assert resumed_report["applied_batches"] == 4
    assert resumed_report["pending"] == []
    assert resumed_report["feature_count"] == 56
    assert resumed_report["score_count"] == 14


def test_injection_isolation_and_business_codes(tmp_path: Path) -> None:
    sample_dir = generate_sample(tmp_path)
    paths = sample_paths(sample_dir)
    injection_dir = sample_dir / "injections"
    report_path = tmp_path / "injection_report.json"
    completed = run_cli(
        "run",
        "--definitions",
        injection_dir / "definitions.json",
        "--events",
        injection_dir / "events.jsonl",
        "--config",
        paths["config"],
        "--edits",
        injection_dir / "edits.json",
        "--state-dir",
        tmp_path / "injection_state",
        "--report",
        report_path,
        "--injection-config",
        injection_dir / "injections.json",
    )
    assert completed.returncode == 0, completed.stderr
    report = load_json(report_path)
    assert report["startup_failures"] == []
    hits = report["injection_hits"]
    assert len(hits) == 7
    assert all(item["hit"] for item in hits)
    assert {item["id"]: item["expected_code"] for item in hits} == {
        "INJ-001": "E_EVENT_INVALID",
        "INJ-002": "E_EVENT_INVALID",
        "INJ-003": "E_EVENT_INVALID",
        "INJ-004": "E_DEFINITION_INVALID",
        "INJ-005": "E_DEFINITION_INVALID",
        "INJ-006": "E_FEATURE_COMPUTE",
        "INJ-007": "E_SCORE_COMPUTE",
    }
    assert report["event_count"] == 19
    failures = report["failures"]
    assert failures == sorted(failures, key=failure_sort_key)
    feature_faults = [item for item in failures if item["code"] == "E_FEATURE_COMPUTE"]
    score_faults = [item for item in failures if item["code"] == "E_SCORE_COMPUTE"]
    assert feature_faults and score_faults
    assert all(item["stage"] == "feature" and item["exception_type"] for item in feature_faults)
    assert all(item["feature_name"] == "user_amount_sum_10m" for item in feature_faults)
    assert score_faults[0]["stage"] == "scoring"
    assert score_faults[0]["entity_type"] == "item"
    assert score_faults[0]["entity_id"] == "i-001"
    assert score_faults[0]["feature_name"] == "item_risk_mean_10m"
    assert any(item["code"] == "E_UPSTREAM_UNAVAILABLE" for item in failures)
    # Invalid entries do not prevent valid entries from loading/computing.
    assert report["feature_count"] > 0
    assert any(
        item["feature_name"] == "item_amount_sum_10m"
        for item in report["features"]
    )


def test_sample_generation_is_byte_deterministic(tmp_path: Path) -> None:
    first = generate_sample(tmp_path / "first")
    second = generate_sample(tmp_path / "second")
    first_hashes = {
        str(path.relative_to(first)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(first.rglob("*"))
        if path.is_file()
    }
    second_hashes = {
        str(path.relative_to(second)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(second.rglob("*"))
        if path.is_file()
    }
    assert first_hashes == second_hashes


def test_report_projection_is_canonical_json_serializable(tmp_path: Path) -> None:
    sample_dir = generate_sample(tmp_path)
    completed, report = run_sample(sample_dir, tmp_path)
    assert completed.returncode == 0
    serialized = canonical_json(
        {
            "features": report["features"],
            "scores": report["scores"],
            "failures": report["failures"],
            "pending": report["pending"],
        }
    )
    assert isinstance(serialized, str) and serialized.startswith("{")


@pytest.mark.parametrize(("sequence_seed", "concurrency", "decimal_shift"), [
    (11, 1, 0),
    (12, 4, 1),
    (13, 8, 2),
])
def test_randomized_valid_edit_sequences_match_full_rebuild(
    tmp_path: Path, sequence_seed: int, concurrency: int, decimal_shift: int
) -> None:
    sample_dir = generate_sample(tmp_path, decimal_shift)
    events_path = sample_dir / "events.jsonl"
    base_events = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    definitions_path = sample_dir / "definitions.json"
    definitions = {
        item["name"]: item for item in load_json(definitions_path)["features"]
    }
    rng = random.Random(sequence_seed)
    cursor = base_events[-1]["timestamp_ms"]
    next_event_number = 20_000
    existing_ids = [item["event_id"] for item in base_events]
    plan = {"seed": sequence_seed, "batches": []}
    for batch_index in range(6):
        batch: dict[str, list] = {
            "batch_id": f"random-{batch_index}",
            "appends": [],
            "backfills": [],
            "definition_changes": [],
        }
        for _ in range(rng.randint(0, 2)):
            cursor += rng.randint(1000, 15_000)
            next_event_number += 1
            event_id = f"evt-r{next_event_number:06d}"
            batch["appends"].append(
                {
                    "event_id": event_id,
                    "user_id": rng.choice([f"u-{index:03d}" for index in range(1, 9)]),
                    "item_id": rng.choice([f"i-{index:03d}" for index in range(1, 7)]),
                    "behavior": rng.choice(["view", "click", "cart", "purchase"]),
                    "timestamp_ms": cursor,
                    "context": {
                        "amount": round(rng.uniform(1.0, 900.0), 6),
                        "dwell_ms": rng.randint(500, 90_000),
                        "risk_score": round(rng.uniform(0.0, 1.0), 6),
                        "session_depth": rng.randint(1, 20),
                    },
                }
            )
            existing_ids.append(event_id)
        if rng.random() < 0.75:
            batch["backfills"].append(
                {
                    "event_id": rng.choice(existing_ids),
                    "patch": {
                        "context": {
                            "amount": round(rng.uniform(1.0, 900.0), 6),
                            "risk_score": round(rng.uniform(0.0, 1.0), 6),
                        }
                    },
                }
            )
        if rng.random() < 0.75:
            name = rng.choice(sorted(definitions))
            definition = dict(definitions[name])
            definition["window_ms"] = rng.choice([300_000, 600_000, 900_000])
            new_places = rng.choice([2, 4, 6, 8])
            definition["decimal_places"] = new_places
            definition["quantization_step"] = decimal_text(Decimal(1).scaleb(-new_places))
            definitions[name] = definition
            batch["definition_changes"].append(
                {"op": "upsert", "definition": definition}
            )
        plan["batches"].append(batch)

    custom_definitions = tmp_path / "random_definitions.json"
    custom_events = tmp_path / "random_events.jsonl"
    custom_plan = tmp_path / "random_plan.json"
    custom_config = tmp_path / "random_config.yaml"
    custom_definitions.write_text(
        json.dumps({"features": list(definitions.values())}), encoding="utf-8"
    )
    custom_events.write_text(
        "".join(json.dumps(item) + "\n" for item in base_events), encoding="utf-8"
    )
    custom_plan.write_text(json.dumps(plan), encoding="utf-8")
    config = yaml.safe_load((sample_dir / "scoring.yaml").read_text(encoding="utf-8"))
    config["concurrency"] = concurrency
    custom_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    _, workbench, loaded_config = execute_incremental(
        definitions_path=custom_definitions,
        events_path=custom_events,
        config_path=custom_config,
        edits_path=custom_plan,
        state_dir=tmp_path / "random_state",
        report_path=tmp_path / "random_report.json",
    )
    assert workbench is not None and loaded_config is not None
    assert workbench.pending_public() == []
    full = build_full_from_final(
        workbench, replace(loaded_config, point_budget_per_batch=10**9)
    )
    comparison = compare_projections(workbench, full)
    assert comparison["equal"] is True, comparison["diff_paths"]
