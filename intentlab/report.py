"""Stable JSON and Markdown diagnostic report output."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Mapping


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with temp.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def write_report(report: Mapping[str, Any], root: str | Path) -> tuple[Path, Path, Path]:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    run_id = str(report.get("run_id") or "uncommitted")
    json_path = root_path / f"{run_id}.json"
    md_path = root_path / f"{run_id}.md"
    latest_json = root_path / "latest.json"
    latest_md = root_path / "latest.md"
    payload = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2)
    _atomic_write_text(json_path, payload)
    _atomic_write_text(latest_json, payload)
    markdown = render_markdown(report)
    _atomic_write_text(md_path, markdown)
    _atomic_write_text(latest_md, markdown)
    return json_path, md_path, latest_json


def render_markdown(report: Mapping[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        f"# IntentLab Diagnostic Report `{report.get('run_id', 'uncommitted')}`",
        "",
        "## Summary",
        "",
        f"- Queries loaded/processed/pending: `{summary.get('queries_loaded', 0)}` / `{summary.get('queries_processed', 0)}` / `{summary.get('queries_pending', 0)}`",
        f"- Merged segments: `{summary.get('segments', 0)}`",
        f"- Boundary intents: `{summary.get('boundary_intents', 0)}`",
        f"- Cache reuse hits: `{summary.get('reuse_hits', 0)}`",
        f"- Business codes: `{json.dumps(summary.get('business_code_counts', {}), ensure_ascii=False, sort_keys=True)}`",
        f"- Snapshot generation: `{report.get('snapshot', {}).get('generation')}`",
        "",
        "## Invalidation by revision",
        "",
        "| revision | match | score | merge | total |",
        "|---|---:|---:|---:|---:|",
    ]
    invalidations = report.get("invalidation_by_revision", {})
    if invalidations:
        for revision_id in sorted(invalidations):
            item = invalidations[revision_id]
            match = int(item.get("match", 0))
            score = int(item.get("score", 0))
            merge = int(item.get("merge", 0))
            lines.append(f"| `{revision_id}` | {match} | {score} | {merge} | {match + score + merge} |")
    else:
        lines.append("| _none_ | 0 | 0 | 0 | 0 |")

    lines.extend(
        [
            "",
            "## Query conclusions",
            "",
            "| query | source | intent | score | margin | boundary | high-width | first error stage | segment |",
            "|---|---|---|---:|---:|---|---|---|---|",
        ]
    )
    for item in report.get("queries", []):
        score = item.get("score", {})
        diagnostics = item.get("diagnostics", {})
        flip = diagnostics.get("flip", {})
        score_text = "unknown"
        margin_text = ""
        for candidate in score.get("scores", []):
            if candidate.get("intent_id") == score.get("predicted_intent"):
                score_text = str(candidate.get("score"))
                margin_text = str(candidate.get("margin"))
                break
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{item.get('query_id')}`",
                    str(item.get("source", "")),
                    f"`{score.get('predicted_intent')}`",
                    score_text,
                    margin_text,
                    str(score.get("boundary")),
                    f"`{flip.get('high_width_intent')}`",
                    str(diagnostics.get("first_exceeding_stage")),
                    f"`{item.get('segment_id')}`",
                ]
            )
            + " |"
        )

    lines.extend(["", "## Segments", ""])
    for segment in report.get("segments", []):
        lines.append(
            f"- `{segment.get('segment_id')}` session `{segment.get('session_id')}` "
            f"intent `{segment.get('intent')}` queries `{', '.join(segment.get('query_ids', []))}` "
            f"slots `{len(segment.get('aligned_slots', []))}` conflicts `{len(segment.get('conflicts', []))}`"
        )

    lines.extend(["", "## Pending queue", ""])
    pending = report.get("pending", [])
    if pending:
        for item in pending:
            lines.append(f"- `{item.get('query_id')}`: `{item.get('code')}`")
    else:
        lines.append("- _empty_")

    lines.extend(["", "## Business issues", ""])
    issues = report.get("issues", [])
    if issues:
        for issue in issues:
            lines.append(
                f"- `{issue.get('code')}` [{issue.get('scope')}/{issue.get('entity_id')}]: "
                f"{issue.get('message')}"
            )
    else:
        lines.append("- _none_")
    lines.append("")
    return "\n".join(lines)

