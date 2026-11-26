"""Core-result projections and acceptance checks."""

from __future__ import annotations

from typing import Any

from gsb.models import DatasetState


def core_projection(report: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "samples": [],
        "pending": list(report.get("pending", [])),
        "failures": [
            {
                key: value
                for key, value in failure.items()
                if key not in {"injection_id"}
            }
            for failure in report.get("failures", [])
        ],
        "template_failures": list(report.get("template_failures", [])),
    }
    for sample in report.get("samples", []):
        summary["samples"].append(
            {
                "sample_id": sample.get("sample_id"),
                "status": sample.get("status"),
                "image_fingerprint": sample.get("image_fingerprint"),
                "annotation_version": sample.get("annotation_version"),
                "candidate_set_key": sample.get("candidate_set_key"),
                "component_set_key": sample.get("component_set_key"),
                "candidates": [
                    {
                        "candidate_id": item.get("candidate_id"),
                        "fingerprint": item.get("fingerprint"),
                        "sort_key": item.get("sort_key"),
                        "window": item.get("window"),
                        "metrics": item.get("metrics"),
                        "score": item.get("score"),
                        "selected": item.get("selected"),
                    }
                    for item in sample.get("candidates", [])
                ],
                "components": [
                    {
                        "component_id": item.get("component_id"),
                        "fingerprint": item.get("fingerprint"),
                        "sort_key": item.get("sort_key"),
                        "candidate_id": item.get("candidate_id"),
                        "candidate_fingerprint": item.get("candidate_fingerprint"),
                        "bbox": item.get("bbox"),
                        "area": item.get("area"),
                        "fragment_ids": item.get("fragment_ids"),
                        "merge_basis": item.get("merge_basis"),
                        "split_basis": item.get("split_basis"),
                    }
                    for item in sample.get("components", [])
                ],
                "keypoints": [
                    {
                        "keypoint_id": item.get("keypoint_id"),
                        "x": item.get("x"),
                        "y": item.get("y"),
                        "bbox": item.get("bbox"),
                        "raw_fixed": item.get("raw_fixed"),
                        "quantized_fixed": item.get("quantized_fixed"),
                        "rounding": item.get("rounding"),
                        "residual_fixed": item.get("residual_fixed"),
                        "cache_key": item.get("cache_key"),
                        "sort_key": item.get("sort_key"),
                        "candidate_id": item.get("candidate_id"),
                        "component_id": item.get("component_id"),
                    }
                    for item in sample.get("keypoints", [])
                ],
                "recognitions": [
                    {
                        "keypoint_id": item.get("keypoint_id"),
                        "char": item.get("char"),
                        "score": item.get("score"),
                        "score_ppm": item.get("score_ppm"),
                        "confidence": item.get("confidence"),
                        "cache_key": item.get("cache_key"),
                        "status": item.get("status"),
                        "sort_key": item.get("sort_key"),
                    }
                    for item in sample.get("recognitions", [])
                ],
                "reading_order": list(sample.get("reading_order", [])),
                "reading_order_cache_key": sample.get("reading_order_cache_key"),
                "conflicts": list(sample.get("conflicts", [])),
            }
        )
    return summary


def compare_core_results(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_projection = core_projection(left)
    right_projection = core_projection(right)
    return {
        "equal": left_projection == right_projection,
        "left": left_projection,
        "right": right_projection,
    }


def truth_accuracy(report: dict[str, Any], dataset: DatasetState) -> dict[str, Any]:
    keypoint_correct = 0
    keypoint_total = 0
    recognition_correct = 0
    recognition_total = 0
    reading_correct = 0
    reading_total = 0
    per_sample: list[dict[str, Any]] = []
    for sample in report.get("samples", []):
        sample_id = sample["sample_id"]
        record = dataset.samples.get(sample_id)
        if record is None:
            continue
        truth_by_id = record.manifest.keypoint_by_id()
        sample_keypoint_correct = 0
        for keypoint in sample.get("keypoints", []):
            truth = truth_by_id.get(keypoint["keypoint_id"])
            if truth is None:
                continue
            keypoint_total += 1
            if keypoint["x"] == truth.x and keypoint["y"] == truth.y:
                keypoint_correct += 1
                sample_keypoint_correct += 1
        sample_recognition_correct = 0
        for recognition in sample.get("recognitions", []):
            truth = truth_by_id.get(recognition["keypoint_id"])
            if truth is None:
                continue
            recognition_total += 1
            if recognition.get("char") == truth.text:
                recognition_correct += 1
                sample_recognition_correct += 1
        sample_reading_correct = 0
        for entry in sample.get("reading_order", []):
            truth = truth_by_id.get(entry["keypoint_id"])
            if truth is None:
                continue
            reading_total += 1
            if entry["position"] == truth.reading_index + 1:
                reading_correct += 1
                sample_reading_correct += 1
        per_sample.append(
            {
                "sample_id": sample_id,
                "keypoint_correct": sample_keypoint_correct,
                "keypoint_total": len(truth_by_id),
                "recognition_correct": sample_recognition_correct,
                "recognition_total": len(truth_by_id),
                "reading_correct": sample_reading_correct,
                "reading_total": len(truth_by_id),
                "conflicts": len(sample.get("conflicts", [])),
            }
        )
    return {
        "keypoint_correct": keypoint_correct,
        "keypoint_total": keypoint_total,
        "recognition_correct": recognition_correct,
        "recognition_total": recognition_total,
        "reading_correct": reading_correct,
        "reading_total": reading_total,
        "all_keypoints_correct": keypoint_correct == keypoint_total,
        "all_recognitions_correct": recognition_correct == recognition_total,
        "all_reading_correct": reading_correct == reading_total,
        "per_sample": per_sample,
    }
