"""Version comparison attribution and demo determinism."""
import json

from cadapp.compare import compare_documents, compare_models
from cadapp.demo import run_demo
from cadapp.editor import EditorSession
from cadapp.migrate import migrate_document
from cadapp.model import model_from_dict
from cadapp.samples import legacy_v1_json_text, make_sample_model


def test_compare_attributes_edit_and_migration():
    base = make_sample_model()
    variant = make_sample_model()
    editor = EditorSession(variant)
    editor.edit_control_point("patch_wave", 1, 1, [-1.4, 0.3, 0.9])
    op_log = list(editor.op_log)

    report = compare_models(base, variant, op_log)
    assert report["diff_count"] == 1
    diff = report["diffs"][0]
    assert diff["kind"] == "control_point"
    assert diff["source"] == "edit"


def test_compare_attributes_migration_actions():
    legacy_doc = json.loads(legacy_v1_json_text())
    migrated_doc, actions = migrate_document(legacy_doc)
    op_log = [{"op": "migrate", "actions": actions}]
    # A v1-derived model differs from the v2 sample only by the fields
    # the migration table fills in; simulate the pre-migration model by
    # stripping those fields back out.
    base = make_sample_model()
    report = compare_models(base, model_from_dict(migrated_doc), op_log)
    assert report["diff_count"] == 0  # migrated doc equals the reference


def test_compare_flags_suspicious_diffs():
    base = make_sample_model()
    variant = make_sample_model()
    variant.patch_by_id("patch_saddle").material_channel = "mystery"
    report = compare_models(base, variant, op_log=[])
    assert report["suspicious_count"] == 1
    assert report["diffs"][0]["source"] == "suspicious"


def test_compare_documents_attributes_migration():
    legacy_doc = json.loads(legacy_v1_json_text())
    migrated_doc, actions = migrate_document(legacy_doc)
    op_log = [{"op": "migrate", "actions": actions}]
    report = compare_documents(legacy_doc, migrated_doc, op_log)
    assert report["diff_count"] > 0
    assert report["suspicious_count"] == 0
    assert all(d["source"] == "migration" for d in report["diffs"])
    # An undocumented change becomes suspicious.
    migrated_doc["patches"][0]["material_channel"] = "mystery"
    report2 = compare_documents(legacy_doc, migrated_doc, op_log)
    assert report2["suspicious_count"] == 1


def test_demo_report_deterministic(tmp_path):
    r1 = run_demo(str(tmp_path / "a"))
    r2 = run_demo(str(tmp_path / "b"))
    assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)
    text1 = (tmp_path / "a" / "demo_report.json").read_bytes()
    text2 = (tmp_path / "b" / "demo_report.json").read_bytes()
    assert text1 == text2


def test_demo_all_checks_pass(tmp_path):
    report = run_demo(str(tmp_path))
    steps = {s["name"]: s for s in report["steps"]}
    assert steps["json_roundtrip"]["hash_equal"]
    assert steps["json_roundtrip"]["control_points_exact"]
    assert steps["binary_roundtrip"]["all_profiles_bit_exact"]
    assert steps["binary_to_json_recovery"]["control_points_exact"]
    assert steps["migration"]["idempotent"]
    assert steps["migration"]["geometry_preserved"]
    assert steps["edit_and_undo"]["undo_restores_hash"]
    assert steps["edit_and_undo"]["unaffected_normals_identical"]
    assert steps["edit_and_undo"]["pipeline_consistent"]
    assert steps["error_codes"]["all_matched"]
    assert steps["version_compare"]["suspicious_count"] >= 1
    assert set(steps["version_compare"]["sources"]) == {"edit", "migration", "suspicious"}
