from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from npr_editor.errors import NprError
from npr_editor.migration import assert_idempotent, migrate_document
from npr_editor.model import (
    canonical_json,
    dump_project,
    load_project,
    parse_json_project,
    project_from_dict,
)
from npr_editor.render import RenderConfig, RenderEngine


def test_json_roundtrip_preserves_float32_control_points(sample_project):
    document = sample_project.to_dict()
    document["shapes"][0]["contours"][0]["segments"][0]["points"][0] = [0.1, 123.456789]
    project = project_from_dict(document)
    encoded = dump_project(project)
    decoded = parse_json_project(encoded)
    point = decoded.shapes[0].contours[0].segments[0].points[0]
    assert point == tuple(float(v) for v in np.asarray([0.1, 123.456789], dtype=np.float32))
    assert decoded.content_hash() == project.content_hash()
    assert canonical_json(decoded.to_dict()) == canonical_json(project.to_dict())


def test_json_content_hash_is_stable_after_pretty_reload(sample_project):
    encoded = dump_project(sample_project, indent=2)
    parsed = load_project(json.loads(encoded))
    assert parsed.content_hash() == sample_project.content_hash()


def test_future_schema_is_rejected(current_document):
    raw = copy.deepcopy(current_document)
    raw["schema_version"] = 4
    with pytest.raises(NprError) as caught:
        load_project(raw)
    assert caught.value.code == "E_SCHEMA_VERSION"


def test_v1_migration_adds_fixed_defaults(v1_document):
    result = migrate_document(v1_document)
    assert result.source_version == 1
    assert [action.field for action in result.actions] == [
        "style",
        "stroke_sample",
        "layer_order",
    ]
    assert result.project.schema_version == 3
    assert result.project.layer_order == ["background", "midground", "ink"]


def test_v2_migration_adds_sampling_and_layer_order(v2_document):
    result = migrate_document(v2_document)
    assert result.source_version == 2
    assert [action.field for action in result.actions] == ["stroke_sample", "layer_order"]
    assert result.project.schema_version == 3


@pytest.mark.parametrize("legacy_fixture", ["v1_document", "v2_document"])
def test_migration_preserves_lineart_render(legacy_fixture, request, sample_project):
    legacy = request.getfixturevalue(legacy_fixture)
    migrated = migrate_document(legacy).project
    config = RenderConfig(width=240, height=160, supersample=1)
    before = RenderEngine(sample_project, config).render("lineart")
    after = RenderEngine(migrated, config).render("lineart")
    assert before.pixel_sha256 == after.pixel_sha256
    assert before.png_sha256 == after.png_sha256
    assert before.geometry_hashes == after.geometry_hashes


@pytest.mark.parametrize("legacy_fixture", ["v1_document", "v2_document"])
def test_migration_is_idempotent(legacy_fixture, request):
    legacy = request.getfixturevalue(legacy_fixture)
    first = migrate_document(legacy)
    second = migrate_document(first.project.to_dict())
    assert assert_idempotent(legacy) is True
    assert second.actions == []
    assert first.content_hash_after == second.content_hash_after


def test_migration_rejects_out_of_range_versions(current_document):
    for version in (0, 4, 99):
        raw = copy.deepcopy(current_document)
        raw["schema_version"] = version
        with pytest.raises(NprError) as caught:
            migrate_document(raw)
        assert caught.value.code == "E_SCHEMA_VERSION"


def test_migration_rejects_unrecoverable_legacy_layer_order(v1_document):
    raw = copy.deepcopy(v1_document)
    raw["shapes"][0].pop("layer")
    with pytest.raises(NprError) as caught:
        migrate_document(raw)
    assert caught.value.code == "E_SCHEMA_VERSION"


def test_invalid_sampling_has_business_error(current_document):
    raw = copy.deepcopy(current_document)
    raw["stroke_sample"]["curve_steps"] = 0
    with pytest.raises(NprError) as caught:
        project_from_dict(raw)
    assert caught.value.code == "E_STROKE_SAMPLE"
