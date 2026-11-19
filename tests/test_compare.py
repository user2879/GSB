from __future__ import annotations

import json
from pathlib import Path

import pytest

from npr_editor.binary_format import decode_binary, encode_binary
from npr_editor.compare import ComparisonStore, VersionTrace, compare_versions
from npr_editor.editing import Editor
from npr_editor.migration import migrate_document
from npr_editor.model import project_from_dict
from npr_editor.render import RenderConfig


CFG = RenderConfig(width=180, height=120, supersample=1)


def test_migration_difference_is_attributed(sample_project, v1_document):
    migrated = migrate_document(v1_document)
    left_trace = VersionTrace(
        version_stamp=sample_project.version_stamp,
        source_schema_version=1,
        migration_actions=[action.to_dict() for action in migrated.actions],
    )
    right_trace = VersionTrace(version_stamp=sample_project.version_stamp)
    comparison = compare_versions(migrated.project, sample_project, left_trace, right_trace, config=CFG)
    format_diffs = [item for item in comparison.result["differences"] if item["kind"] == "format_version"]
    assert format_diffs
    assert all("migration" in item["sources"] for item in format_diffs)
    assert comparison.result["suspicious_items"] == []


def test_edited_control_point_is_attributed_to_edit(sample_project):
    left_trace = VersionTrace(version_stamp=sample_project.version_stamp)
    editor = Editor(sample_project, config=CFG)
    editor.edit_control_point("tree_crown", 0, 0, 1, 151.0, 83.0)
    right_project = editor.project
    right_trace = VersionTrace(
        version_stamp=right_project.version_stamp,
        edited_shapes=("tree_crown",),
        edit_actions=[{"shape": "tree_crown", "field": "control_point"}],
    )
    comparison = compare_versions(sample_project, right_project, left_trace, right_trace, config=CFG)
    control_diffs = [item for item in comparison.result["differences"] if item["kind"] == "control_point"]
    assert control_diffs
    assert all("edit" in item["sources"] for item in control_diffs)
    assert comparison.result["suspicious_items"] == []


def test_binary_representation_change_is_attributed_to_format_conversion(sample_project):
    encoded = encode_binary(
        sample_project,
        byte_order="big",
        index_width=2,
        include_project_metadata=False,
    )
    imported = decode_binary(encoded.data).project
    document = imported.to_dict()
    document["project_id"] = sample_project.project_id
    document["version_stamp"] = "000000000002"
    imported = project_from_dict(document)
    left_trace = VersionTrace(
        version_stamp=sample_project.version_stamp,
        binary_format_version=2,
    )
    right_trace = VersionTrace(
        version_stamp=imported.version_stamp,
        binary_format_version=2,
        conversion_actions=[{"operation": "json_to_skeleton_to_json"}],
    )
    comparison = compare_versions(sample_project, imported, left_trace, right_trace, config=CFG)
    control_entries = [
        item for item in comparison.result["differences"] if item["kind"] in ("control_structure", "control_point")
    ]
    assert control_entries
    assert all("format_conversion" in item["sources"] for item in control_entries)
    assert comparison.result["suspicious_items"] == []


def test_unattributed_geometry_difference_is_suspicious(sample_project):
    document = sample_project.to_dict()
    document["version_stamp"] = "000000000002"
    document["shapes"][3]["contours"][0]["segments"][0]["points"][1][0] = 151.0
    changed = project_from_dict(document)
    left_trace = VersionTrace(version_stamp=sample_project.version_stamp)
    right_trace = VersionTrace(version_stamp=changed.version_stamp)
    comparison = compare_versions(sample_project, changed, left_trace, right_trace, config=CFG)
    assert comparison.result["suspicious_items"]
    suspicious = [
        item for item in comparison.result["differences"] if item["suspicious"]
    ]
    assert suspicious
    assert all(item["sources"] == [] for item in suspicious)


def test_comparison_is_persisted_and_recomputable(sample_project, tmp_path):
    editor = Editor(sample_project, config=CFG)
    editor.edit_control_point("tree_crown", 0, 0, 1, 151.0, 83.0)
    right = editor.project
    left_trace = VersionTrace(version_stamp=sample_project.version_stamp)
    right_trace = VersionTrace(
        version_stamp=right.version_stamp,
        edited_shapes=("tree_crown",),
        edit_actions=[{"shape": "tree_crown", "field": "control_point"}],
    )
    store = ComparisonStore(tmp_path)
    saved, path = store.save(sample_project, right, left_trace, right_trace, config=CFG)
    assert (path / "side_by_side.png").is_file()
    assert (path / "COMMITTED").is_file()
    recomputed = store.recompute(sample_project.version_stamp, right.version_stamp)
    assert recomputed == saved
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["comparison_hash"] == saved["comparison_hash"]
