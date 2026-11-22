"""Canonical analysis artifacts and byte-stable output files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .canonical import (
    canonical_lines,
    sha256_bytes,
    sha256_obj,
    write_json,
    write_jsonl,
)
from .config import Declarations
from .errors import state_error
from .records import NormalizedRecord, model_digest, provenance_rows


def build_artifact(
    declarations: Declarations,
    records: Iterable[NormalizedRecord],
    analysis: dict[str, Any],
    *,
    parameter_version: int | None = None,
    execution_id: str | None = None,
    view_diff: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    records = list(records)
    conclusions = analysis["conclusions"]
    group_conclusions = analysis["group_conclusions"]
    conflicts = analysis["conflicts"]
    conclusions_digest = sha256_bytes(canonical_lines(conclusions))
    group_conclusions_digest = sha256_bytes(canonical_lines(group_conclusions))
    conflicts_digest = sha256_bytes(canonical_lines(conflicts))
    stats_digest = sha256_obj(analysis["stats"])
    digest_payload = {
        "schema_version": 1,
        "parameter_version": parameter_version,
        "parameter_fingerprint": declarations.parameter_fingerprint(),
        "format_versions": list(declarations.format_versions),
        "model_digest": model_digest(records),
        "conclusions_digest": conclusions_digest,
        "group_conclusions_digest": group_conclusions_digest,
        "conflicts_digest": conflicts_digest,
        "stats_digest": stats_digest,
    }
    artifact = {
        "schema_version": 1,
        "parameter_version": parameter_version,
        "execution_id": execution_id,
        "parameter_fingerprint": declarations.parameter_fingerprint(),
        "format_versions": list(declarations.format_versions),
        "model_digest": digest_payload["model_digest"],
        "conclusions_digest": conclusions_digest,
        "group_conclusions_digest": group_conclusions_digest,
        "conflicts_digest": conflicts_digest,
        "stats_digest": stats_digest,
        "analysis_digest": sha256_obj(digest_payload),
        "record_count": analysis["record_count"],
        "format_counts": analysis["format_counts"],
        "verdict_counts": analysis["verdict_counts"],
        "conflict_counts": analysis["conflict_counts"],
        "conclusions": conclusions,
        "group_conclusions": group_conclusions,
        "conflicts": conflicts,
        "stats": analysis["stats"],
        "view_diff": list(view_diff or []),
    }
    return artifact


def summary_from_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": artifact["schema_version"],
        "parameter_version": artifact["parameter_version"],
        "execution_id": artifact["execution_id"],
        "parameter_fingerprint": artifact["parameter_fingerprint"],
        "format_versions": artifact["format_versions"],
        "model_digest": artifact["model_digest"],
        "conclusions_digest": artifact["conclusions_digest"],
        "group_conclusions_digest": artifact["group_conclusions_digest"],
        "conflicts_digest": artifact["conflicts_digest"],
        "stats_digest": artifact["stats_digest"],
        "analysis_digest": artifact["analysis_digest"],
        "record_count": artifact["record_count"],
        "format_counts": artifact["format_counts"],
        "verdict_counts": artifact["verdict_counts"],
        "conflict_counts": artifact["conflict_counts"],
        "topic_ranking": artifact["stats"]["topic_ranking"],
        "comparison_affected_pairs": artifact["stats"]["comparison_affected_pairs"],
        "view_diff_counts": _view_diff_counts(artifact.get("view_diff", [])),
    }


def _view_diff_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts = {"added": 0, "deleted": 0, "changed": 0}
    for row in rows:
        counts[row["kind"]] = counts.get(row["kind"], 0) + 1
    return dict(sorted(counts.items()))


def write_analysis_outputs(
    out_dir: str | Path,
    artifact: dict[str, Any],
    records: Iterable[NormalizedRecord],
) -> dict[str, str]:
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    files = {
        "artifact": target / "artifact.json",
        "summary": target / "summary.json",
        "conclusions": target / "conclusions.jsonl",
        "group_conclusions": target / "group_conclusions.jsonl",
        "conflicts": target / "conflicts.jsonl",
        "stats": target / "stats.json",
        "model_digest": target / "model_digest.txt",
        "normalized_records": target / "normalized_records.jsonl",
    }
    write_json(files["artifact"], artifact)
    write_json(files["summary"], summary_from_artifact(artifact))
    write_jsonl(files["conclusions"], artifact["conclusions"])
    write_jsonl(files["group_conclusions"], artifact["group_conclusions"])
    write_jsonl(files["conflicts"], artifact["conflicts"])
    write_json(files["stats"], artifact["stats"])
    files["model_digest"].write_text(artifact["model_digest"] + "\n", encoding="utf-8", newline="\n")
    write_jsonl(files["normalized_records"], provenance_rows(records))
    return {key: str(value) for key, value in files.items()}


def compare_expected_summary(artifact: dict[str, Any], expected: dict[str, Any]) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    fields = (
        "model_digest",
        "conclusions_digest",
        "group_conclusions_digest",
        "conflicts_digest",
        "record_count",
        "format_counts",
        "verdict_counts",
        "conflict_counts",
    )
    for field in fields:
        if field not in expected:
            continue
        if artifact.get(field) != expected[field]:
            mismatches.append(
                {"field": field, "expected": expected[field], "actual": artifact.get(field)}
            )
    expected_conclusions = expected.get("conclusions")
    if expected_conclusions is not None:
        actual = {
            row["message_id"]: {
                "message_id": row["message_id"],
                "verdict": row["verdict"],
                "source_formats": row["basis"]["source_formats"],
                "source_groups": row["basis"]["source_groups"],
            }
            for row in artifact["conclusions"]
        }
        wanted = {row["message_id"]: row for row in expected_conclusions}
        for message_id in sorted(set(actual) | set(wanted)):
            if actual.get(message_id) != wanted.get(message_id):
                mismatches.append(
                    {
                        "field": f"conclusions[{message_id}]",
                        "expected": wanted.get(message_id),
                        "actual": actual.get(message_id),
                    }
                )
    return mismatches


def assert_expected_summary(artifact: dict[str, Any], expected: dict[str, Any]) -> None:
    mismatches = compare_expected_summary(artifact, expected)
    if mismatches:
        raise state_error(
            "artifact does not match the expected summary",
            first_mismatch=mismatches[0],
            mismatch_count=len(mismatches),
        )
