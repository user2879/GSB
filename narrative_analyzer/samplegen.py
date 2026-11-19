"""Programmatic generation of the deterministic demonstration corpus."""

from __future__ import annotations

import copy
from typing import Any

import yaml

from . import __version__
from .canonical import pretty_json_bytes, sha256_bytes, sha256_value
from .engine import replay
from .errors import BusinessError, ERROR_CODES
from .inject import parse_injection, run_injection
from .pathcheck import analyze_paths
from .sensitivity import ScanConfig, scan_sensitivity
from .snapshot import build_snapshot, verify_snapshot_value
from .spec import parse_spec


def _yaml_bytes(value: Any) -> bytes:
    text = yaml.safe_dump(
        value,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
        width=1000,
    )
    return text.encode("utf-8")


def _json_bytes(value: Any) -> bytes:
    return pretty_json_bytes(value)


def sample_spec_value() -> dict[str, Any]:
    return {
        "spec_version": 1,
        "seed": 20261119,
        "start": "start",
        "comparison": "strict_greater",
        "boundary_epsilon": 1.0e-9,
        "path_tolerance": 1.0e-18,
        "numeric": {
            "coefficient_relative_error": 1.0e-12,
            "initial_relative_error": 1.0e-12,
            "unit_roundoff": 1.1102230246251565e-16,
            "max_abs_error": 1.0e-3,
        },
        "state": {
            "components": ["trust", "reputation", "fear"],
            "initial": {"trust": 0.0, "reputation": 0.1, "fear": 0.0},
        },
        "conditions": [
            {"id": "c_trust_high", "component": "trust", "threshold": 0.3, "direction": ">"},
        ],
        "nodes": [
            {"id": "start", "type": "scene"},
            {"id": "branch_a", "type": "scene"},
            {"id": "branch_b", "type": "scene"},
            {"id": "merge", "type": "choice"},
            {"id": "ending_good", "type": "ending", "outcome": "good"},
            {"id": "ending_bad", "type": "ending", "outcome": "bad"},
        ],
        "edges": [
            {
                "id": "e_start_a",
                "source": "start",
                "target": "branch_a",
                "choice": "path_a",
                "effects": [
                    "trust += 0.1",
                    "reputation += 0.05",
                    "fear += 0.02",
                ],
            },
            {
                "id": "e_start_b",
                "source": "start",
                "target": "branch_b",
                "choice": "path_b",
                "effects": [
                    "trust += 0.3",
                    "reputation += 0.05",
                    "fear += 0.03",
                ],
            },
            {
                "id": "e_a_join",
                "source": "branch_a",
                "target": "merge",
                "choice": "join",
                "effects": [
                    "trust += 0.2",
                    "fear += 0.01",
                ],
            },
            {
                "id": "e_b_join",
                "source": "branch_b",
                "target": "merge",
                "choice": "join",
                "effects": [],
            },
            {
                "id": "e_good",
                "source": "merge",
                "target": "ending_good",
                "choice": "continue",
                "when": ["c_trust_high"],
                "effects": [],
            },
            {
                "id": "e_bad",
                "source": "merge",
                "target": "ending_bad",
                "choice": "continue",
                "effects": [],
            },
        ],
    }


def _sequence_value(selections: list[str]) -> dict[str, Any]:
    return {"version": 1, "selections": selections}


def _scan_value() -> dict[str, Any]:
    return {
        "version": 1,
        "direction": "increase",
        "relative_deltas": [1.0e-15, 1.0e-12, 1.0e-9, 1.0e-6, 1.0e-4, 1.0e-2, 0.1],
    }


def _injection_value() -> dict[str, Any]:
    return {
        "version": 1,
        "name": "threshold-nudge-flips-fallback-to-good-ending",
        "coefficient_perturbations": [],
        "threshold_offsets": [{"condition": "c_trust_high", "delta": -1.0e-9}],
        "initial_offsets": [],
    }


def _capture_error(func: Any) -> dict[str, Any]:
    try:
        func()
    except BusinessError as exc:
        return exc.to_dict()
    return {
        "business_code": "NO_ERROR",
        "message": "expected a business error but the operation succeeded",
        "details": {},
    }


def build_demo_artifacts() -> dict[str, bytes]:
    spec_value = sample_spec_value()
    spec_bytes = _yaml_bytes(spec_value)
    spec = parse_spec(spec_bytes, "spec.yaml")
    primary_selections = ("path_a", "join", "continue")
    branch_b_selections = ("path_b", "join", "continue")
    primary = replay(spec, primary_selections)
    branch_b = replay(spec, branch_b_selections)
    paths = analyze_paths(spec)
    injection_bytes = _yaml_bytes(_injection_value())
    injection_plan = parse_injection(injection_bytes, "inject_threshold_flip.yaml")
    injected = run_injection(spec, branch_b_selections, injection_plan)
    sensitivity = scan_sensitivity(
        spec,
        branch_b_selections,
        ScanConfig(relative_deltas=(1.0e-15, 1.0e-12, 1.0e-9, 1.0e-6, 1.0e-4, 1.0e-2, 0.1), direction=1),
    )

    unknown_spec = copy.deepcopy(spec_value)
    unknown_spec["edges"][0]["effects"][0] = "trust += unknown_drive"
    cycle_spec = copy.deepcopy(spec_value)
    cycle_spec["edges"].append(
        {
            "id": "e_cycle",
            "source": "merge",
            "target": "start",
            "choice": "loop",
            "effects": [],
        }
    )
    missing_threshold_spec = copy.deepcopy(spec_value)
    del missing_threshold_spec["conditions"][0]["threshold"]
    precision_spec = copy.deepcopy(spec_value)
    precision_spec["numeric"]["max_abs_error"] = 1.0e-16
    invalid_injection = {
        "version": 1,
        "coefficient_perturbations": [
            {"coefficient": "does/not/exist/coef[0]", "relative": 1.0e-9}
        ],
        "threshold_offsets": [],
        "initial_offsets": [],
    }

    def trigger_precision_bound() -> None:
        precision_parsed = parse_spec(_yaml_bytes(precision_spec), "invalid_precision_bound.yaml")
        replay(precision_parsed, primary_selections)

    def trigger_inject_config() -> None:
        parsed_plan = parse_injection(_yaml_bytes(invalid_injection), "invalid_inject.yaml")
        run_injection(spec, primary_selections, parsed_plan)

    fake_snapshot = {
        "version": 1,
        "engine_version": __version__,
        "inputs": [],
        "artifact_count": 1,
        "artifacts": [{"path": "spec.yaml", "sha256": "0" * 64, "size": len(spec_bytes)}],
    }
    fake_snapshot["content_sha256"] = sha256_value(fake_snapshot)

    error_cases = [
        _capture_error(lambda: parse_spec(_yaml_bytes(unknown_spec), "invalid_unknown.yaml")),
        _capture_error(lambda: parse_spec(_yaml_bytes(cycle_spec), "invalid_cycle.yaml")),
        _capture_error(lambda: replay(spec, ("not_a_choice",))),
        _capture_error(trigger_precision_bound),
        _capture_error(lambda: analyze_paths(spec, raise_on_diverge=True)),
        _capture_error(trigger_inject_config),
        _capture_error(
            lambda: verify_snapshot_value(fake_snapshot, lambda name: spec_bytes)
        ),
    ]
    seen_codes = [case["business_code"] for case in error_cases]
    if tuple(seen_codes) != ERROR_CODES:
        raise RuntimeError(f"demo error chain mismatch: {seen_codes!r}")

    boundary_decisions = [
        decision
        for decision in primary["decisions"]
        if decision["boundary"] or decision["uncertain"]
    ]
    summary = {
        "business_code": "OK",
        "description": "deterministic narrative precision and causal analysis sample set",
        "sample_set": {
            "normal_path": "sequence_primary.yaml",
            "boundary_path": "sequence_boundary.yaml",
            "branch_merge_divergence": "paths.json",
            "injection_flip": "replay_injected.json",
            "sensitivity": "sensitivity.json",
        },
        "spec_sha256": spec.source_sha256,
        "primary": {
            "selections": list(primary_selections),
            "outcome": primary["outcome"],
            "final_state_bits": primary["final_state"]["bits"],
            "final_state_values": primary["final_state"]["values"],
            "edge_sequence_hash": primary["edge_sequence_hash"],
            "content_hash": primary["content_hash"],
        },
        "boundary_judgments": [
            {
                "condition_id": decision["condition_id"],
                "step_index": decision["step_index"],
                "edge_id": decision["edge_id"],
                "signed_margin": decision["signed_margin"],
                "absolute_margin": decision["absolute_margin"],
                "margin_error_upper": decision["margin_error_upper"],
                "reachable": decision["reachable"],
                "uncertain": decision["uncertain"],
                "boundary": decision["boundary"],
            }
            for decision in boundary_decisions
        ],
        "branch_paths": {
            "business_code": paths["business_code"],
            "first_divergence": paths["first_divergence"],
        },
        "injection_flip": {
            "outcome_changed": injected["outcome_changed"],
            "outcome_before": injected["baseline"]["outcome"],
            "outcome_after": injected["injected"]["outcome"],
            "first_flip": injected["first_flip"],
        },
        "sensitivity": {
            "critical_count": sensitivity["critical_count"],
            "top": sensitivity["ranking"][:5],
        },
        "error_code_coverage": seen_codes,
    }

    artifacts: dict[str, bytes] = {
        "spec.yaml": spec_bytes,
        "sequence_primary.yaml": _yaml_bytes(_sequence_value(list(primary_selections))),
        "sequence_boundary.yaml": _yaml_bytes(_sequence_value(["path_a", "join", "continue"])),
        "sequence_branch_b.yaml": _yaml_bytes(_sequence_value(list(branch_b_selections))),
        "inject_threshold_flip.yaml": injection_bytes,
        "sensitivity_config.yaml": _yaml_bytes(_scan_value()),
        "invalid_unknown.yaml": _yaml_bytes(unknown_spec),
        "invalid_cycle.yaml": _yaml_bytes(cycle_spec),
        "invalid_missing_threshold.yaml": _yaml_bytes(missing_threshold_spec),
        "invalid_precision_bound.yaml": _yaml_bytes(precision_spec),
        "invalid_inject.yaml": _yaml_bytes(invalid_injection),
        "replay_primary.json": _json_bytes(primary),
        "replay_branch_b.json": _json_bytes(branch_b),
        "replay_injected.json": _json_bytes(injected),
        "paths.json": _json_bytes(paths),
        "sensitivity.json": _json_bytes(sensitivity),
        "error_cases.json": _json_bytes({"business_code": "OK", "cases": error_cases}),
        "demo_summary.json": _json_bytes(summary),
    }
    return artifacts


def build_demo_bundle() -> dict[str, bytes]:
    first = build_demo_artifacts()
    second = build_demo_artifacts()
    mismatches = sorted(
        name for name in set(first) | set(second) if first.get(name) != second.get(name)
    )
    if mismatches:
        raise RuntimeError(f"demo build is not byte-reproducible: {mismatches}")
    reproducibility = {
        "business_code": "OK",
        "identical": True,
        "compared_artifact_count": len(first),
        "artifacts": [
            {"path": name, "sha256": sha256_bytes(first[name]), "size": len(first[name])}
            for name in sorted(first)
        ],
        "manifest_sha256": sha256_value(
            {name: sha256_bytes(first[name]) for name in sorted(first)}
        ),
    }
    all_artifacts = dict(first)
    all_artifacts["reproducibility.json"] = _json_bytes(reproducibility)
    input_names = (
        "spec.yaml",
        "sequence_primary.yaml",
        "sequence_boundary.yaml",
        "sequence_branch_b.yaml",
        "inject_threshold_flip.yaml",
        "sensitivity_config.yaml",
    )
    snapshot = build_snapshot(all_artifacts, input_names)
    verify_snapshot_value(snapshot, lambda name: all_artifacts[name])
    all_artifacts["snapshot.json"] = _json_bytes(snapshot)
    return all_artifacts
