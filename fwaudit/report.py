"""结构化 JSON 审计报告与人工可读分区映射图数据。"""

import os

from .errors import E_OK
from .storage import atomic_write_json


def build_graph(image_info, patch_results):
    """构造分区映射图：分区节点、槽位标记、越权链边。"""
    nodes = []
    for part in image_info.partitions:
        nodes.append({
            "id": part.name,
            "label": part.name,
            "slot": part.slot,
            "part_type": part.part_type,
            "payload_region": [part.payload_offset, part.payload_end],
            "record_offset": part.record_offset,
            "findings": [
                finding for finding in image_info.findings
                if finding.get("partition") == part.name
                or part.name in finding.get("partitions", [])
            ],
        })

    edges = []
    for result in patch_results:
        for chain in result.get("write_chains", []):
            edges.append({
                "source": result["path"],
                "target": chain["name"],
                "edge_type": "unauthorized_write_chain",
                "operation_seqs": chain["operation_seqs"],
                "merged_absolute_region": chain["merged_absolute_region"],
            })
        for finding in result.get("findings", []):
            if finding["code"] == "E_SLOT_ESCALATION":
                edges.append({
                    "source": result["path"],
                    "target": finding["name"],
                    "edge_type": "slot_escalation",
                    "seq": finding["seq"],
                })
    overlap_edges = [
        {
            "source": pair[0],
            "target": pair[1],
            "edge_type": "overlap",
            "relation": finding["relation"],
        }
        for finding in image_info.findings
        if finding["code"] == "E_OVERLAP"
        for pair in [finding["partitions"]]
    ]
    edges.extend(overlap_edges)
    return {
        "graph_version": 1,
        "base_image_id": image_info.base_image_id,
        "base_slot": image_info.base_slot,
        "slot_legend": {0: "slot-A", 1: "slot-B"},
        "nodes": nodes,
        "edges": edges,
    }


def _patch_summary(result):
    codes = result.get("codes") or [E_OK]
    return {
        "path": result["path"],
        "status": result["status"],
        "codes": codes,
        "stale": result.get("stale", False),
        "degraded": result.get("degraded", False),
        "coverage_ratio": result.get("coverage_ratio", 1.0),
        "uncovered_ops": result.get("uncovered_ops", []),
        "op_count": result.get("op_count"),
        "elapsed_seconds": result.get("elapsed_seconds", 0.0),
        "operations": result.get("operations", []),
        "findings": result.get("findings", []),
        "write_chains": result.get("write_chains", []),
    }


def build_report(image_info, engine_output, budget, samples_meta=None):
    image_codes = sorted({f["code"] for f in image_info.findings}) or [E_OK]
    results = engine_output.get("results", [])
    report = {
        "report_version": 1,
        "budget": budget.as_dict(),
        "image": {
            "path": image_info.path,
            "status": "pass" if image_info.ok else "fail",
            "codes": image_codes,
            "version": image_info.version,
            "base_image_id": image_info.base_image_id,
            "base_slot": image_info.base_slot,
            "declared_length": image_info.declared_length,
            "table_region": [image_info.table_offset, image_info.table_end],
            "partitions": [
                {
                    **part.as_dict(),
                    "mapping_status": (
                        "conflict"
                        if any(
                            part.name in f.get("partitions", [])
                            or f.get("partition") == part.name
                            for f in image_info.findings
                        )
                        else "mapped"
                    ),
                }
                for part in image_info.partitions
            ],
            "findings": image_info.findings,
        },
        "patches": [_patch_summary(r) for r in results],
        "parse_errors": engine_output.get("parse_errors", []),
        "samples_meta": samples_meta or {},
        "graph": build_graph(image_info, results),
    }
    report["summary"] = _summarize(report)
    return report


def _summarize(report):
    patch_total = len(report["patches"])
    patch_failed = sum(1 for p in report["patches"] if p["status"] != "pass")
    parse_failed = len(report["parse_errors"])
    overall = (
        "pass"
        if report["image"]["status"] == "pass"
        and patch_failed == 0 and parse_failed == 0
        else "fail"
    )
    return {
        "overall_status": overall,
        "image_ok": report["image"]["status"] == "pass",
        "patch_total": patch_total,
        "patch_failed": patch_failed,
        "parse_errors": parse_failed,
        "code_counts": _code_counts(report),
    }


def _code_counts(report):
    counts = {}
    for finding in report["image"]["findings"]:
        counts[finding["code"]] = counts.get(finding["code"], 0) + 1
    for patch in report["patches"]:
        for code in patch["codes"]:
            counts[code] = counts.get(code, 0) + 1
    for item in report["parse_errors"]:
        code = item["error"]["code"]
        counts[code] = counts.get(code, 0) + 1
    return dict(sorted(counts.items()))


def write_outputs(report, report_path, graph_path):
    atomic_write_json(report_path, report)
    atomic_write_json(graph_path, report["graph"])
    return report_path, graph_path


def render_graph_text(graph):
    """人工可读的分区映射图（纯文本，确定性输出）。"""
    lines = [
        "固件分区映射图 (base_image_id=%d, base_slot=%d)"
        % (graph["base_image_id"], graph["base_slot"]),
        "节点:",
    ]
    for node in graph["nodes"]:
        start, end = node["payload_region"]
        lines.append(
            "  - %s [slot=%d type=%d] 载荷 [%d, %d) 长度=%d 状态=%s"
            % (
                node["label"], node["slot"], node["part_type"],
                start, end, end - start,
                "conflict" if node["findings"] else "mapped",
            )
        )
    lines.append("边:")
    if not graph["edges"]:
        lines.append("  (无)")
    for edge in graph["edges"]:
        lines.append(
            "  - %s --[%s]--> %s"
            % (edge["source"], edge["edge_type"], edge["target"])
        )
    return os.linesep.join(lines) + os.linesep
