"""与渲染无关的诊断视图模型：所有界面数据按稳定顺序在此生成。"""

from .graph import topological_order


class DiagnosticViewModel:
    def __init__(self, ledger, version, invalid_stage_ids=None, snapshots=None):
        self.ledger = ledger
        self.version = version
        self.invalid_stage_ids = list(invalid_stage_ids or [])
        self.snapshots = list(snapshots or [])

    def build(self):
        plan, events = self.ledger.load_plan_events(self.version)
        topo = topological_order(plan.stages)
        outcomes = {item["stage_id"]: item for item in self.ledger.active_outcomes(self.version)}
        rows = []
        for stage_id in topo:
            stage = plan.stage_map[stage_id]
            outcome = outcomes.get(stage_id, {})
            rows.append(
                {
                    "stage_id": stage_id,
                    "order": stage.order,
                    "prerequisites": list(stage.prerequisites),
                    "state": outcome.get("state", "pending"),
                    "reason": outcome.get("reason"),
                    "skip_trace": outcome.get("skip_trace", []),
                    "plan_version": outcome.get("plan_version"),
                    "unknown": outcome.get("state") == "unknown",
                    "stale": bool(outcome.get("stale", 0)),
                }
            )
        timeline = [
            {
                "event_id": event.event_id,
                "kind": event.kind,
                "offset_ms": event.offset_ms,
                "target_stage": event.target_stage,
            }
            for event in sorted(events.events, key=lambda item: (item.offset_ms, item.event_id))
        ]
        return {
            "plan_id": plan.plan_id,
            "plan_version": plan.version,
            "stage_rows": rows,
            "invalid_stage_ids": list(self.invalid_stage_ids),
            "event_timeline": timeline,
            "stats": self.ledger.stats(self.version),
            "snapshots": list(self.snapshots),
            "stale_versions": [v for v in self.ledger.list_versions() if v != self.version],
        }


def render_text(view):
    lines = [
        f"预案: {view['plan_id']}  版本: v{view['plan_version']}",
        "阶段（稳定拓扑序）:",
    ]
    for row in view["stage_rows"]:
        flags = []
        if row["unknown"]:
            flags.append("UNKNOWN")
        if row["stale"]:
            flags.append("STALE")
        flag_text = f" [{' '.join(flags)}]" if flags else ""
        prereq = ",".join(row["prerequisites"]) or "-"
        trace = " -> ".join(row["skip_trace"]) if row["skip_trace"] else ""
        reason = f" reason={row['reason']}" if row["reason"] else ""
        lines.append(
            f"  {row['order']}. {row['stage_id']} state={row['state']} prereq=[{prereq}]"
            f" v{row['plan_version']}{reason}{flag_text} {trace}".rstrip()
        )
    lines.append("本轮失效集合: " + (", ".join(view["invalid_stage_ids"]) or "-"))
    lines.append("注入事件时间轴:")
    for event in view["event_timeline"]:
        lines.append(
            f"  {event['offset_ms']:>6}ms {event['event_id']} {event['kind']} -> {event['target_stage']}"
        )
    lines.append(
        "统计: total={total} succeeded={succeeded} failed={failed} skipped={skipped} unknown={unknown}".format(
            **view["stats"]
        )
    )
    lines.append("陈旧版本: " + (", ".join(f"v{v}" for v in view["stale_versions"]) or "-"))
    lines.append("快照列表: " + (", ".join(view["snapshots"]) or "-"))
    return "\n".join(lines)
