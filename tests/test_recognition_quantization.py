from __future__ import annotations

from gsb.pipeline import run_pipeline
from gsb.trace import analyze_attribution
from gsb.verification import compare_core_results, truth_accuracy


def _recognition_map(report: dict) -> dict[str, dict]:
    return {
        item["keypoint_id"]: item
        for sample in report["samples"]
        for item in sample["recognitions"]
    }


def test_default_recognitions_are_correct_at_all_quantization_levels(project, baseline):
    assert truth_accuracy(baseline.report, project.dataset)["all_recognitions_correct"]
    for level in ("Q4", "Q6", "Q10"):
        result = run_pipeline(
            project.dataset,
            config=project.config.with_quantization(level).with_budget(10**12),
            templates=project.require_templates(),
        )
        accuracy = truth_accuracy(result.report, project.dataset)
        assert accuracy["all_keypoints_correct"]
        assert accuracy["all_recognitions_correct"]
        assert accuracy["all_reading_correct"]


def test_quantization_changes_score_confidence_and_cache_keys(project, baseline):
    q4 = run_pipeline(
        project.dataset,
        config=project.config.with_quantization("Q4").with_budget(10**12),
        templates=project.require_templates(),
    )
    q8 = _recognition_map(baseline.report)
    q4_map = _recognition_map(q4.report)
    changed_confidence = 0
    changed_score_scale = 0
    changed_keypoint_cache = 0
    for keypoint_id, q8_recognition in q8.items():
        q4_recognition = q4_map[keypoint_id]
        changed_confidence += q4_recognition["confidence"] != q8_recognition["confidence"]
        changed_score_scale += q4_recognition["score"] != q8_recognition["score"]
        changed_keypoint_cache += q4_recognition["cache_key"] != q8_recognition["cache_key"]
    assert changed_confidence > 0
    assert changed_score_scale > 0
    assert changed_keypoint_cache == len(q8)
    assert q4.report["fixed_bits"] == 4
    assert baseline.report["fixed_bits"] == 8


def test_quantization_flip_fixture_and_attribution(flip_project, flip_q8):
    templates = flip_project.require_templates()
    q4 = run_pipeline(
        flip_project.dataset,
        config=flip_project.config.with_quantization("Q4"),
        templates=templates,
    )
    q4_recognition = _recognition_map(q4.report)["quantization_flip-c0-r0"]
    q8_recognition = _recognition_map(flip_q8.report)["quantization_flip-c0-r0"]

    assert q8_recognition["char"] == "曰"
    assert q4_recognition["char"] == "己"
    assert q4_recognition["confidence"] != q8_recognition["confidence"]
    assert q4_recognition["block_bbox"] == q8_recognition["block_bbox"]
    q4_tied = sorted(q4_recognition["template_scores"], key=lambda item: -item["score"])
    assert q4_tied[0]["score"] == q4_tied[1]["score"]
    assert q4_recognition["char"] == min(q4_tied[0]["char"], q4_tied[1]["char"], key=ord)

    attribution = analyze_attribution(
        flip_project.dataset,
        config=flip_project.config.with_quantization("Q8"),
        templates=templates,
        base_result=flip_q8,
        alternate_levels=("Q4",),
    )
    assert attribution["flip_count"] == 1
    assert attribution["flips"][0]["source"] == "template_match_score_rounding"
    assert attribution["per_character"][0]["first_exceeding_stage"] is None


def test_point_budget_preserves_pending_and_matches_full_run(project, baseline):
    config = project.config.with_budget(15000)
    templates = project.require_templates()
    cache = None
    prior: dict = {}
    pending_ids: tuple[str, ...] = ()
    pending_counts: list[int] = []
    result = None
    for _ in range(12):
        result = run_pipeline(
            project.dataset,
            config=config,
            templates=templates,
            cache=cache,
            prior_recognition_results=prior,
            pending_ids=pending_ids,
        )
        cache = result.cache
        prior = result.recognition_results
        pending_ids = result.pending_ids
        pending_counts.append(len(pending_ids))
        assert result.report["budget"]["consumed"] <= 15000
        assert all(
            item["code"] == "E_POINT_BUDGET_PENDING"
            for item in result.report["pending"]
        )
        if not pending_ids:
            break

    assert result is not None
    assert pending_counts == [22, 8, 0]
    assert result.report["budget"]["consumed"] == 8232
    assert compare_core_results(result.report, baseline.report)["equal"]
