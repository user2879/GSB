"""Stable, deterministic text rendering of the full diagnostic view."""


def render_diagnostic_report(view: dict) -> str:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("PIPELINE CACHE LEDGER - DIAGNOSTIC REPORT")
    lines.append("=" * 72)

    lines.append("")
    lines.append("[stages and artifacts]")
    for stage in view["stages"]:
        lines.append(
            f"  {stage['order']}. {stage['stage_id']}: "
            f"{', '.join(stage['artifacts'])}"
        )

    lines.append("")
    lines.append("[entries / hit conclusions]")
    header = (
        f"  {'key':<14} {'stage':<17} {'ver':>3} {'src':>3} "
        f"{'status':<11} {'outcome':<10} reason"
    )
    lines.append(header)
    for art in view["artifacts"]:
        src = view["source_versions"].get(art["key"], "-")
        lines.append(
            f"  {art['key']:<14} {art['stage_id']:<17} {art['version']:>3} "
            f"{str(src):>3} {art['status']:<11} "
            f"{str(art['hit_outcome']):<10} {art['hit_reason']}"
        )

    lines.append("")
    lines.append("[backfilled fields after cross-version normalization]")
    if view["backfilled_fields"]:
        for key in sorted(view["backfilled_fields"]):
            fields = ", ".join(view["backfilled_fields"][key])
            lines.append(f"  {key}: {fields}")
    else:
        lines.append("  (none)")

    lines.append("")
    lines.append("[invalidation propagation trails]")
    if view["trails"]:
        for root in sorted(view["trails"]):
            lines.append(f"  root={root} -> " + " -> ".join(view["trails"][root]))
    else:
        lines.append("  (none)")

    lines.append("")
    lines.append("[concurrency slot occupancy]")
    for slot in view["slot_timeline"]:
        parts = [
            f"{occ['key']}@t{occ['start_tick']}-t{occ['end_tick']}"
            for occ in slot["occupancy"]
        ]
        lines.append(f"  slot {slot['slot']}: " + ", ".join(parts))

    lines.append("")
    lines.append("[snapshots]")
    if view["snapshots"]:
        for snap in view["snapshots"]:
            ro = "readonly" if snap["readonly"] else "writable"
            lines.append(
                f"  {snap['name']} fmt={snap['snapshot_format']} "
                f"pipeline_v={snap['pipeline_version']} "
                f"graph_v={snap['graph_version']} {ro} "
                f"digest={snap['payload_digest'][:18]}..."
            )
    else:
        lines.append("  (none)")
    lines.append("")
    return "\n".join(lines)


def render_upgrade_report(report: dict) -> str:
    lines = ["", "[cache open upgrade]"]
    lines.append(
        f"  total entries={report['total']} processed="
        f"{len(report['processed'])} upgraded={len(report['upgraded'])}"
    )
    for item in report["upgraded"]:
        lines.append(
            f"  - {item['key']}: v{item['from']} -> v{item['to']} "
            f"backfilled={','.join(item['backfilled']) or '-'}"
        )
    return "\n".join(lines)
