"""Stable, deterministic diagnostic report rendering."""

from __future__ import annotations

import json
from typing import Any, Optional

from .canonical import short_hexdigest
from .ledger import Ledger


def render_report(ledger: Ledger, run_id: Optional[int] = None) -> str:
    model = ledger.view_model(run_id)
    if not model.get("run"):
        return "PIPELINE DIAGNOSTIC WORKBENCH\n(no runs recorded)\n"
    run = model["run"]
    evolution = json.loads(run["evolution_json"])
    active = ledger.active_run()
    recovered = "yes" if run.get("recovery_of") else "no"
    active_id = active["id"] if active is not None else "-"
    lines = [
        "PIPELINE DIAGNOSTIC WORKBENCH",
        f"RUN run={run['id']} version={run['fingerprint_version']} "
        f"status={run['status']} recovered={recovered} active_run={active_id}",
        f"FINGERPRINT value={run['spec_fingerprint']} "
        f"version={run['fingerprint_version']} "
        f"samples={run['samples_digest']}",
        f"EVOLUTION baseline={evolution.get('baseline_run_id')} "
        f"changed={'yes' if evolution.get('changed_stage_ids') else 'no'} "
        f"sample_changed={'yes' if evolution.get('sample_changed') else 'no'} "
        f"affected={','.join(evolution.get('affected_stage_ids', [])) or '-'} "
        f"unchanged={','.join(evolution.get('unchanged_stage_ids', [])) or '-'} "
        f"invalidated_shards={evolution.get('invalidated_shards', 0)} "
        f"reasons={','.join(evolution.get('reasons', []))}",
    ]
    for change in evolution.get("stage_changes", []):
        lines.append(
            f"CHANGE stage={change['stage_id']} old={change['old_fingerprint']} "
            f"new={change['new_fingerprint']}")
    if evolution.get("unchanged_stage_ids"):
        lines.append(
            "PRESERVED stages=" +
            ",".join(evolution["unchanged_stage_ids"]))
    lines.append("STAGES")
    for stage in model["stages"]:
        lines.extend(_render_stage(stage))
    lines.append("SHARDS")
    if model["shards"]:
        for shard in model["shards"]:
            lines.append(
                f"  [{shard['shard_index']:04d}] stage={shard['stage_id']} "
                f"rows=[{shard['row_start']},{shard['row_end']}) "
                f"count={shard['row_count']} "
                f"fp={short_hexdigest(shard['shard_fingerprint'])} "
                f"source={short_hexdigest(shard['source_stage_digest'])}")
    else:
        lines.append("  none")
    lines.append("EVENTS")
    if model["events"]:
        for event in model["events"]:
            lines.append(
                f"  seq={event['sequence']} code={event['code']} "
                f"stage={event['stage_id'] or '-'} "
                f"path={event['field_path']} message={event['message']}")
    else:
        lines.append("  none")
    return "\n".join(lines) + "\n"


def _render_stage(stage: dict[str, Any]) -> list[str]:
    digest = stage["output_digest"] or "-"
    head = (
        f"  [{stage['ordinal'] + 1:02d}] id={stage['stage_id']} "
        f"type={stage['stage_type']} depends_on={stage['depends_on']} "
        f"status={stage['status']} rows={stage['rows_in']}->"
        f"{stage['rows_out']} digest={short_hexdigest(digest) if digest != '-' else digest}")
    rendered = [head]
    for column in sorted(stage["column_stats"]):
        stats = stage["column_stats"][column]
        if "min_raw" in stats:
            rendered.append(
                f"       column={column} missing={stats['missing_count']} "
                f"min_raw={stats['min_raw']} max_raw={stats['max_raw']} "
                f"zero={stats['zero_count']}")
        else:
            rendered.append(
                f"       column={column} missing={stats['missing_count']} "
                f"sum_fixed={stats['sum_fixed']} "
                f"min_fixed={stats['min_fixed']} "
                f"max_fixed={stats['max_fixed']} zero={stats['zero_count']}")
    if stage["extra"]:
        rendered.append(
            "       extra=" + json.dumps(stage["extra"], sort_keys=True,
                                         separators=(",", ":")))
    return rendered
