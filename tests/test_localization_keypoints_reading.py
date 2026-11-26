from __future__ import annotations

import io
from dataclasses import replace

import numpy as np
from PIL import Image

from gsb.localize import sliding_candidates
from gsb.reading import assemble_reading_order


def _sample_report(result, sample_id: str) -> dict:
    return next(item for item in result.report["samples"] if item["sample_id"] == sample_id)


def _image(record) -> np.ndarray:
    image = Image.open(io.BytesIO(record.image_bytes))
    image.load()
    return np.asarray(image.convert("L"), dtype=np.uint8)


def test_components_exactly_match_truth_bboxes(project, baseline):
    for sample_id in ("page_a", "page_b"):
        sample = _sample_report(baseline, sample_id)
        truth = sorted(
            keypoint.bbox.to_list()
            for keypoint in project.dataset.samples[sample_id].manifest.keypoints
        )
        observed = sorted(component["bbox"] for component in sample["components"])
        assert observed == truth
        assert len(observed) == len(truth)


def test_candidate_fields_are_stable_across_worker_counts(project):
    record = project.dataset.samples["page_a"]
    image = _image(record)
    templates = project.require_templates()

    def execute(workers: int):
        config = project.config.with_workers(workers)
        execution = sliding_candidates(
            image,
            config=config,
            image_fingerprint=record.image_fingerprint,
            parameter_fingerprint=config.stage_parameter_fingerprint("candidate"),
            template_fingerprint=templates.content_fingerprint,
            file_name="page_a.png",
        )
        return execution, [item.to_dict() for item in execution.results]

    serial_execution, serial = execute(1)
    concurrent_execution, concurrent = execute(4)

    assert serial == concurrent
    assert serial_execution.max_observed == 1
    assert concurrent_execution.max_observed <= 4
    assert all(item["fingerprint"] == item["candidate_id"] or len(item["fingerprint"]) == 64 for item in serial)


def test_keypoints_have_fixed_point_audit_fields_and_valid_references(baseline):
    for sample in baseline.report["samples"]:
        candidate_ids = {item["candidate_id"] for item in sample["candidates"]}
        component_ids = {item["component_id"] for item in sample["components"]}
        for keypoint in sample["keypoints"]:
            assert isinstance(keypoint["x"], int)
            assert isinstance(keypoint["y"], int)
            assert all(isinstance(value, int) for value in keypoint["raw_fixed"])
            assert all(isinstance(value, int) for value in keypoint["quantized_fixed"])
            assert keypoint["candidate_id"] in candidate_ids
            assert keypoint["component_id"] in component_ids
            assert keypoint["rounding"]["regression"]["x"] in {"UP", "DOWN", "EXACT"}
            assert keypoint["rounding"]["regression"]["y"] in {"UP", "DOWN", "EXACT"}
            assert keypoint["cache_key"]
            assert keypoint["sort_key"]


def test_reading_order_is_right_to_left_and_top_to_bottom(project, baseline):
    for sample in baseline.report["samples"]:
        manifest = project.dataset.samples[sample["sample_id"]].manifest
        truth_by_id = manifest.keypoint_by_id()
        assert sample["conflicts"] == []
        assert [entry["position"] for entry in sample["reading_order"]] == list(
            range(1, len(manifest.keypoints) + 1)
        )
        for entry in sample["reading_order"]:
            assert entry["position"] == truth_by_id[entry["keypoint_id"]].reading_index + 1


def _two_keypoints(baseline):
    keypoints = {
        item.keypoint_id: item
        for item in baseline.cache.keypoints.values()
        if item.keypoint_id.startswith("page_a-")
    }
    return [keypoints["page_a-c0-r0"], keypoints["page_a-c0-r1"]]


def test_reading_conflict_priority_cross_column_misorder_missing(project, baseline):
    first, second = _two_keypoints(baseline)
    columns = project.dataset.samples["page_a"].manifest.columns
    config = project.config

    def assemble(modified):
        return assemble_reading_order(
            [first, modified],
            columns=columns,
            config=config,
            image_fingerprint="image",
            parameter_fingerprint="parameters",
            template_fingerprint="templates",
            file_name="page_a.png",
        )

    cross = assemble(replace(second, column_truth=1, reading_index_truth=4))
    assert cross.conflicts[0].conflict_type == "CROSS_COLUMN"
    assert set(cross.conflicts[0].types) == {"CROSS_COLUMN", "MISSING"}
    assert cross.conflicts[0].priority == 1

    misorder = assemble(replace(second, reading_index_truth=0))
    assert misorder.conflicts[0].conflict_type == "MISORDER"
    assert set(misorder.conflicts[0].types) == {"MISORDER", "MISSING"}
    assert misorder.conflicts[0].priority == 2

    missing = assemble(replace(second, reading_index_truth=2))
    assert missing.conflicts[0].conflict_type == "MISSING"
    assert missing.conflicts[0].types == ("MISSING",)
    assert missing.conflicts[0].priority == 3
