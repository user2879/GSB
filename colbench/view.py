"""诊断工作台视图模型与稳定顺序文本渲染。

GUI（Tkinter）与无头演示共用 :class:`DashboardModel`，保证渲染顺序完全一致；
无显示环境或缺少 tkinter 时由 :mod:`colbench.gui` 降级为文本渲染。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .planner import ScanPlan
from .scheduler import ScheduleResult
from .versioning import ParameterVersion, ReplacementReport


@dataclass
class DashboardModel:
    plan: ScanPlan
    schedule: ScheduleResult
    version: ParameterVersion
    block_rows: list[dict] = field(default_factory=list)
    segment_quota: list[dict] = field(default_factory=list)
    business_codes: list[str] = field(default_factory=list)
    replacement: ReplacementReport | None = None
    snapshots: list[str] = field(default_factory=list)
    verify_report: dict | None = None

    # ------------------------------------------------------------------
    def section_titles(self) -> list[str]:
        return ["参数版本与业务码", "列裁剪与谓词下推", "块编码与压缩比",
                "任务调度时间线", "每段配额占用", "扫描统计",
                "整体替换与快照列表", "权威重算校验"]

    def render_text(self) -> str:
        lines: list[str] = []
        lines.append("=" * 72)
        lines.append("列式存储压缩与扫描诊断工作台")
        lines.append("=" * 72)

        lines.append("[1] 参数版本与业务码")
        lines.append(f"  版本 budget={self.version.budget} "
                     f"encoding={self.version.encoding} "
                     f"format={self.version.format}")
        lines.append("  业务码: " + (", ".join(self.business_codes)
                                     if self.business_codes else "OK"))

        lines.append("[2] 列裁剪与谓词下推")
        summary = self.plan.summary()
        lines.append(f"  计划摘要: {summary['plan_digest']}")
        lines.append("  投影列: " + ", ".join(summary["projected_columns"]))
        lines.append("  裁掉列: " + ", ".join(summary["pruned_columns"]))
        lines.append("  下推条件:")
        for predicate in summary["pushed"]:
            lines.append("    - " + json_predicate(predicate) + " [PUSHED]")
        for predicate in summary["retained"]:
            lines.append("    - " + json_predicate(predicate) + " [RETAINED]")
        lines.append(f"  候选块 {summary['candidate_block_count']} 个，"
                     f"跳过块 {summary['skipped_block_count']} 个")
        for reason in summary["reasons"]:
            lines.append(f"    依据: {reason}")

        lines.append("[3] 块编码与压缩比")
        for row in self.block_rows:
            ratio = (row["pre_bytes"] / row["post_bytes"]
                     if row["post_bytes"] else 0.0)
            lines.append(
                f"  {row['segment_id']} rg={row['row_group']} "
                f"{row['column']:<9} {row['encoding']:<4} "
                f"压缩={'Y' if row['compressed'] else 'N'} "
                f"前={row['pre_bytes']:>4}B 后={row['post_bytes']:>4}B "
                f"比值={ratio:.2f}")

        lines.append("[4] 任务调度时间线（单调时钟毫秒）")
        for task in self.schedule.tasks:
            lines.append(
                f"  {task.task_id} {task.segment_id} "
                f"rg={task.row_group} {task.column:<9} "
                f"状态={task.status:<9} 排队={task.wait_ms:>3}ms "
                f"运行={task.run_ms:>3}ms 内存={task.memory_cost:>4}B")

        lines.append("[5] 每段配额占用")
        for row in self.segment_quota:
            lines.append(
                f"  {row['segment_id']} 峰值并发={row['peak_running']} "
                f"配额={row['quota']} 任务数={row['tasks']}")

        lines.append("[6] 扫描统计")
        for key, value in self.schedule.statistics().items():
            lines.append(f"  {key}: {value}")

        lines.append("[7] 整体替换与快照列表")
        if self.replacement:
            report = self.replacement.to_dict()
            lines.append(f"  替换: 清空轨迹 {report['cleared_trajectory_count']} "
                         f"条，受影响块 {len(report['affected_block_keys'])} 个，"
                         f"陈旧 {report['stale_count']} 条")
            for key in report["affected_block_keys"]:
                lines.append(f"    受影响块: {key}")
        else:
            lines.append("  尚未发生版本替换")
        for snapshot in self.snapshots:
            lines.append(f"  快照: {snapshot}")

        lines.append("[8] 权威重算校验")
        if self.verify_report:
            lines.append(f"  {self.verify_report.get('status')}: "
                         f"比较块 {self.verify_report.get('blocks_compared')} 个")
        else:
            lines.append("  未执行")
        lines.append("=" * 72)
        return "\n".join(lines)


def json_predicate(predicate: dict) -> str:
    return f"{predicate['column']} {predicate['kind']} {predicate['values']}"


def build_block_rows(segments) -> list[dict]:
    rows = []
    for segment_id in sorted(segments):
        segment = segments[segment_id]
        for key in sorted(segment.headers):
            header = segment.headers[key]
            rows.append({
                "segment_id": segment_id, "row_group": header.row_group,
                "column": header.column,
                "block_index": header.block_index,
                "encoding": header.encoding,
                "compressed": header.compressed,
                "pre_bytes": header.pre_bytes,
                "post_bytes": header.post_bytes,
                "min_value": header.min_value,
                "max_value": header.max_value,
                "distinct_count": header.distinct_count,
            })
    return rows


def build_segment_quota(schedule: ScheduleResult,
                        per_segment_concurrency: int) -> list[dict]:
    counts: dict[str, int] = {}
    for task in schedule.tasks:
        counts[task.segment_id] = counts.get(task.segment_id, 0) + 1
    return [
        {"segment_id": segment_id, "tasks": counts[segment_id],
         "peak_running": schedule.max_segment_running.get(segment_id, 0),
         "quota": per_segment_concurrency}
        for segment_id in sorted(counts)]
