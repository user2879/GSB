from __future__ import annotations

from pathlib import Path

from image_edit_workbench.codec_v1 import LINE_WIDTH, encode_v1, parse_v1
from image_edit_workbench.codec_v2 import encode_v2, parse_v2
from image_edit_workbench.convert import downgrade_v2, restore_v1_with_sidecar, v1_to_v2_regions
from image_edit_workbench.util import read_json


def test_v1_v2_parse_and_fixed_width(workspace):
    images = workspace["images"]
    sample_dir = workspace["sample_dir"]
    keys = {region.region_key for region in workspace["regions"]}
    v1_text = (sample_dir / "exchange_v1.txt").read_text(encoding="utf-8")
    v2_text = (sample_dir / "exchange_v2.json").read_text(encoding="utf-8")
    assert all(len(line) == LINE_WIDTH for line in v1_text.splitlines())
    v1 = parse_v1(v1_text, images=images, manifest_keys=keys)
    v2 = parse_v2(v2_text, images=images, manifest_keys=keys)
    assert len(v1.regions) == len(v2.regions) == 6
    assert not v1.failures and not v2.failures
    assert int(v1.regions[0].style_strength * 1000) == v1.regions[0].style_permille
    assert v1.regions[0].rotation_deg == 0
    assert v2.regions[0].rotation_deg != 0
    assert v1.document_fingerprint and v2.document_fingerprint


def test_v1_roundtrip_is_field_exact(workspace):
    images = workspace["images"]
    keys = {region.region_key for region in workspace["regions"]}
    parsed = parse_v1(
        (workspace["sample_dir"] / "exchange_v1.txt").read_text(encoding="utf-8"),
        images=images,
        manifest_keys=keys,
    )
    upgraded = v1_to_v2_regions(parsed.regions)
    roundtrip = parse_v2(encode_v2(upgraded), images=images, manifest_keys=keys)
    for direct, parsed_back in zip(upgraded, roundtrip.regions):
        assert direct.image_name == parsed_back.image_name
        assert direct.region_index == parsed_back.region_index
        assert direct.rect == parsed_back.rect
        assert direct.style_strength == parsed_back.style_strength
        assert direct.target_attribute == parsed_back.target_attribute
        assert direct.repair_method == parsed_back.repair_method
        assert direct.rotation_deg == parsed_back.rotation_deg
        assert direct.feather_radius == parsed_back.feather_radius


def test_downgrade_sidecar_restores_exact_v2(workspace):
    images = workspace["images"]
    keys = {region.region_key for region in workspace["regions"]}
    v2 = parse_v2(
        (workspace["sample_dir"] / "exchange_v2.json").read_text(encoding="utf-8"),
        images=images,
        manifest_keys=keys,
    )
    converted = downgrade_v2(v2.regions)
    assert any(any(value != "0" for value in record["coordinate_residual"].values()) for record in converted.sidecar["records"])
    assert any(record["rotation_deg"] != "0" for record in converted.sidecar["records"])
    low = parse_v1(converted.text, images=images, manifest_keys=keys)
    restored = restore_v1_with_sidecar(low.regions, converted.sidecar)
    assert restored.sidecar_hits == 6
    assert restored.sidecar_misses == 0
    assert [region.canonical_dict() for region in restored.regions] == [
        region.canonical_dict() for region in v2.regions
    ]


def test_business_error_codes(workspace):
    images = workspace["images"]
    keys = {region.region_key for region in workspace["regions"]}
    root = workspace["root"]
    truncated = parse_v1(
        (root / "injections" / "truncated_v1.txt").read_text(encoding="utf-8"),
        images=images,
        manifest_keys=keys,
    )
    assert [failure.code for failure in truncated.failures] == ["E_FIELD_TRUNCATED"]
    structure = parse_v2(
        (root / "injections" / "invalid_structure_v2.json").read_text(encoding="utf-8"),
        images=images,
        manifest_keys=keys,
    )
    assert structure.failures[0].code == "E_STRUCTURE_INVALID"
    assert len(structure.regions) == 5
    out_of_range = parse_v2(
        (root / "injections" / "out_of_range_v2.json").read_text(encoding="utf-8"),
        images=images,
        manifest_keys=keys,
    )
    assert out_of_range.failures[0].code == "E_REGION_OUT_OF_RANGE"
    assert len(out_of_range.regions) == 5
