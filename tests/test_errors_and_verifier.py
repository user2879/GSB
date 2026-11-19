from __future__ import annotations

import json
from pathlib import Path

import pytest

from mesh_repair.boundary import VertexProposal, resolve_boundary_proposals
from mesh_repair.errors import EXIT_CODES, ErrorCode
from mesh_repair.hashing import write_json
from mesh_repair.io_ply import read_ply
from mesh_repair.pipeline import _cross_check_quality_and_verifier
from mesh_repair.quality import COMMON_CHECK_NAMES
from mesh_repair.verification import validate_files

from .conftest import PROJECT_ROOT


def test_required_error_catalog_and_exit_codes():
    required = {
        ErrorCode.E_MESH_FORMAT,
        ErrorCode.E_BLOCK_KEY,
        ErrorCode.E_BOUNDARY_CONFLICT,
        ErrorCode.E_BUDGET_DENIED,
        ErrorCode.E_VOLUME_DEVIATE,
        ErrorCode.E_RESIDUAL_TOPOLOGY,
        ErrorCode.E_VERIFIER_MISMATCH,
    }
    assert required.issubset(set(ErrorCode))
    assert {EXIT_CODES[code] for code in required} == {10, 11, 12, 13, 14, 15, 16}


def test_mesh_format_error(default_run, tmp_path):
    bad = tmp_path / "bad.ply"
    bad.write_bytes(b"this is not a ply file\n")
    with pytest.raises(Exception) as exc_info:
        read_ply(bad)
    assert getattr(exc_info.value, "code", None) == ErrorCode.E_MESH_FORMAT
    report = validate_files(bad, default_run.output_dir / "ground_truth.json")
    assert report["status"] == "fail"
    assert report["business_code"] == "E_MESH_FORMAT"


def test_residual_topology_is_not_just_printed(default_run):
    report = validate_files(
        default_run.output_dir / "defective.ply",
        default_run.output_dir / "ground_truth.json",
    )
    assert report["status"] == "fail"
    assert report["business_code"] == "E_RESIDUAL_TOPOLOGY"
    assert report["first_failure"]["business_code"] == "E_RESIDUAL_TOPOLOGY"


def test_volume_deviation_business_code(default_run, tmp_path):
    truth = json.loads((default_run.output_dir / "ground_truth.json").read_text(encoding="utf-8"))
    truth["baseline"]["volume"] *= 1.10
    tampered = tmp_path / "tampered_truth.json"
    write_json(tampered, truth)
    report = validate_files(default_run.output_dir / "baseline.ply", tampered)
    assert report["status"] == "fail"
    assert report["business_code"] == "E_VOLUME_DEVIATE"
    assert report["first_failure"]["name"] == "volume_within_tolerance"


def test_boundary_arbiter_rejects_duplicate_registration():
    with pytest.raises(Exception) as exc_info:
        resolve_boundary_proposals(
            [
                VertexProposal("C00000-P00000", 0, (0.0, 0.0, 0.0), True),
                VertexProposal("C00000-P00000", 0, (0.0, 0.0, 0.0), True),
            ],
            tolerance=1.0e-9,
            vertex_count=1,
        )
    assert getattr(exc_info.value, "code", None) == ErrorCode.E_BOUNDARY_CONFLICT


def test_boundary_position_conflict_is_resolved_by_block_key_order():
    high = VertexProposal("C00000-P00001", 0, (1.0, 0.0, 0.0), False)
    low = VertexProposal("C00000-P00000", 0, (0.0, 0.0, 0.0), True)
    first = resolve_boundary_proposals([high, low], tolerance=1.0e-9, vertex_count=1)
    second = resolve_boundary_proposals([low, high], tolerance=1.0e-9, vertex_count=1)
    assert first.to_dict() == second.to_dict()
    assert first.positions[0] == (0.0, 0.0, 0.0)
    assert first.keep[0] is True
    assert first.position_conflicts[0].resolved_by == "C00000-P00000"
    assert first.retention_conflicts[0].criterion == "retention_disagreement_resolved_by_logical_or"


def test_pipeline_verifier_mismatch_detection():
    passing_checks = {
        name: {"name": name, "status": "pass", "location": {}}
        for name in COMMON_CHECK_NAMES
    }
    quality = {
        "status": "pass",
        "business_code": "OK",
        "checks_by_name": passing_checks,
        "metrics": {
            "volume": 10.0,
            "area": 20.0,
            "component_count": 3,
            "face_count": 100,
            "nonmanifold_edge_count": 0,
            "boundary_edge_count": 0,
            "inconsistent_winding_edge_count": 0,
            "self_intersection_count": 0,
        },
    }
    verifier_checks = {name: dict(value) for name, value in passing_checks.items()}
    verifier_checks["self_intersections_zero"] = {
        "name": "self_intersections_zero",
        "status": "fail",
        "location": {"faces": [7, 9], "criterion": "triangle_intersection"},
    }
    verifier = {
        "status": "fail",
        "business_code": "E_RESIDUAL_TOPOLOGY",
        "checks_by_name": verifier_checks,
        "first_failure": verifier_checks["self_intersections_zero"],
        "metrics": dict(quality["metrics"]),
    }
    cross_check = _cross_check_quality_and_verifier(quality, verifier)
    assert cross_check["status"] == "fail"
    assert cross_check["first_mismatch"]["criterion"] == "check_status_mismatch"
    assert cross_check["first_mismatch"]["location"]["faces"] == [7, 9]
