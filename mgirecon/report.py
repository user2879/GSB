"""Machine-readable and console diagnostics."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def build_report(result: dict[str, Any], *, project_dir: str | Path | None = None) -> dict[str, Any]:
    views: list[dict[str, Any]] = []
    matches_by_view: dict[str, list[str]] = {}
    for pair in result.get("match_pairs", []):
        if pair.get("selected"):
            matches_by_view.setdefault(pair["left"], []).append(pair["pair_id"])
            matches_by_view.setdefault(pair["right"], []).append(pair["pair_id"])
    for view_id in sorted(result.get("features", {})):
        feature_map = result["features"][view_id]
        depth_map = result.get("depth_maps", {}).get(view_id, {})
        views.append(
            {
                "view": view_id,
                "features": feature_map.get("feature_count", 0),
                "depth_pixels": depth_map.get("populated_pixels", 0),
                "boundary_pixels": depth_map.get("boundary_pixels", 0),
                "matching_pairs": sorted(matches_by_view.get(view_id, [])),
            }
        )
    selected_pairs = [
        {
            "pair": pair["pair_id"],
            "inliers": pair["inlier_count"],
            "selected": pair["selected"],
            "tied": pair["tied"],
            "tie_winner": pair["tie_winner"],
            "rank": pair["decision_rank"],
        }
        for pair in result.get("match_pairs", [])
    ]
    reused_counts = Counter(entry["stage"] for entry in result.get("ledger", {}).get("reused", []))
    invalidated_counts = Counter(entry["stage"] for entry in result.get("ledger", {}).get("invalidated", []))
    relation_counts: Counter[str] = Counter()
    for entry in result.get("ledger", {}).get("invalidated", []):
        relation_counts.update(entry.get("relations", []))
    error_codes = sorted({error["code"] for error in result.get("errors", [])})
    report = {
        "project_dir": str(project_dir) if project_dir is not None else None,
        "status": result.get("status"),
        "error_codes": error_codes,
        "errors": result.get("errors", []),
        "strategy": result.get("strategy", {}),
        "views": views,
        "match_pairs": selected_pairs,
        "voxel": result.get("voxel_stats", {}),
        "budget": result.get("budget", {}),
        "pending": result.get("pending", []),
        "ledger": {
            "reused_by_stage": dict(sorted(reused_counts.items())),
            "invalidated_by_stage": dict(sorted(invalidated_counts.items())),
            "invalidated_by_relation": dict(sorted(relation_counts.items())),
            "reused_total": sum(reused_counts.values()),
            "invalidated_total": sum(invalidated_counts.values()),
        },
        "scheduler": result.get("scheduler", {}),
        "depth_reuse": result.get("depth_reuse", {}),
        "snapshot": result.get("manifest", {}),
    }
    return report


def write_report(report: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def console_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"status: {report['status']}")
    lines.append(f"error codes: {', '.join(report['error_codes']) if report['error_codes'] else '-'}")
    strategy = report.get("strategy", {})
    lines.append(
        "strategy: {name} block={block_size} depth={effective_max_depth} split={split_order} estimated_children={estimated_children}".format(
            name=strategy.get("name"),
            block_size=strategy.get("block_size"),
            effective_max_depth=strategy.get("effective_max_depth"),
            split_order=strategy.get("split_order"),
            estimated_children=strategy.get("estimated_children"),
        )
    )
    lines.append("views:")
    for view in report["views"]:
        lines.append(
            f"  {view['view']}: features={view['features']} depth_pixels={view['depth_pixels']} "
            f"boundary={view['boundary_pixels']} pairs={','.join(view['matching_pairs']) or '-'}"
        )
    lines.append("matching pairs:")
    for pair in report["match_pairs"]:
        lines.append(
            f"  {pair['pair']}: inliers={pair['inliers']} selected={pair['selected']} "
            f"rank={pair['rank']} tied={pair['tied']} tie_winner={pair['tie_winner']}"
        )
    voxel = report.get("voxel", {})
    lines.append(
        f"voxels: total={voxel.get('voxel_count', 0)} leaves={voxel.get('leaf_count', 0)} "
        f"subdivided={voxel.get('subdivided_count', 0)} max_depth={voxel.get('max_depth_used', 0)} "
        f"adjacency={voxel.get('adjacency_edges', 0)}"
    )
    lines.append(f"subdivision depth distribution: {voxel.get('depth_distribution', {})}")
    ledger = report.get("ledger", {})
    lines.append(
        f"ledger: reused={ledger.get('reused_total', 0)} invalidated={ledger.get('invalidated_total', 0)} "
        f"relations={ledger.get('invalidated_by_relation', {})}"
    )
    budget = report.get("budget", {})
    lines.append(
        f"budget: consumed={budget.get('global_consumed', 0)}/{budget.get('global_limit', 0)} "
        f"pending={len(report.get('pending', []))} quota={budget.get('per_view_quota', 0)}"
    )
    for pending in report.get("pending", [])[:10]:
        lines.append(f"  pending {pending.get('kind')} {pending.get('key')} reason={pending.get('reason')}")
    if report["errors"]:
        lines.append("diagnostics:")
        for error in report["errors"]:
            location = error.get("view") or "-"
            lines.append(f"  [{error['code']}] stage={error.get('stage')} view={location}: {error['message']}")
    return "\n".join(lines)
