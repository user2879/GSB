"""One-command deterministic demonstration and report generation."""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Any

from .canonical import numpy_available, pretty_json, sha256_json, sha256_text, write_utf8
from .dependency import IncrementalDependencyGraph
from .injection import (
    InjectionConfig,
    apply_injections,
    locate_first_flip,
    parse_injection_config,
)
from .merge import check_all_merge_nodes
from .samples import generate_samples
from .spec import load_spec
from .timeline import NarrativeSession, TimelineResult


REPORT_SCHEMA = "narrative-workbench/demo-report/v1"


def _load_json_config(path: Path) -> InjectionConfig:
    return parse_injection_config(path.read_text(encoding="utf-8"))


def _timeline_payload(result: TimelineResult) -> dict[str, Any]:
    return result.as_dict()


def _all_arbitrations(result: TimelineResult) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for snapshot in result.snapshots:
        for arbitration in snapshot.arbitrations:
            records.append(
                {
                    "state_version": snapshot.state_version,
                    "version_stamp": snapshot.version_stamp,
                    **arbitration.as_dict(),
                }
            )
    return records


def _outgoing_projection_arbitrations(result: TimelineResult) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for snapshot in result.snapshots:
        projection = snapshot.outgoing_projection
        for arbitration in projection.get("arbitrations", []):
            records.append(
                {
                    "state_version": snapshot.state_version,
                    "version_stamp": snapshot.version_stamp,
                    "node_id": snapshot.node_id,
                    **arbitration,
                }
            )
    return records


def _boundary_manifests(result: TimelineResult) -> list[dict[str, Any]]:
    return [
        snapshot.as_dict()["boundary_manifest"]
        for snapshot in result.snapshots
        if snapshot.boundary_judgments
    ]


def _hash_manifest(
    *,
    specs: dict[str, dict[str, Any]],
    timelines: dict[str, TimelineResult],
    merge_report: dict[str, Any],
    dependency_reports: dict[str, Any],
    injection_flip: dict[str, Any],
    injection_suite: dict[str, Any],
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema": "narrative-workbench/hash-manifest/v1",
        "specs": {
            key: {"spec_hash": value["spec_hash"], "file": value["file"], "sequence": value["sequence"]}
            for key, value in sorted(specs.items())
        },
        "timeline_nodes": {
            key: {
                "edge_sequence_hash": result.edge_sequence_hash,
                "state_hash": result.state_hash,
                "replay_hash": result.replay_hash,
                "snapshot_hashes": {
                    f"snapshot:{snapshot.state_version:04d}": snapshot.content_hash
                    for snapshot in result.snapshots
                },
            }
            for key, result in sorted(timelines.items())
        },
        "merge_nodes": {
            check["target_node"]: check["report_hash"] for check in merge_report["checks"]
        },
        "dependency_nodes": {
            section: report["invalidation"]["node_hashes"]
            for section, report in sorted(dependency_reports.items())
        },
        "injection": {
            "flip_report_hash": injection_flip["report_hash"],
            "suite_report_hash": injection_suite["report_hash"],
        },
    }
    manifest["manifest_hash"] = sha256_json(manifest)
    return manifest


def build_demo_payload(samples_dir: str | Path) -> dict[str, Any]:
    sample_root = Path(samples_dir)
    normal_spec = load_spec(sample_root / "normal.yaml")
    merge_spec = load_spec(sample_root / "merge.yaml")
    boundary_spec = load_spec(sample_root / "boundary.yaml")
    injection_spec = load_spec(sample_root / "injection.yaml")

    normal_timeline = NarrativeSession(normal_spec, "normal_path").run()
    merge_timeline = NarrativeSession(merge_spec, "merge_left").run()
    boundary_timeline = NarrativeSession(boundary_spec, "boundary_path").run()
    injection_baseline = NarrativeSession(injection_spec, "injection_path").run()

    flip_config = _load_json_config(sample_root / "injection_flip.json")
    injected_spec, flip_records = apply_injections(injection_spec, flip_config)
    injection_injected = NarrativeSession(injected_spec, "injection_path").run()
    flip_report = locate_first_flip(
        injection_baseline,
        injection_injected,
        injection_records=flip_records,
        injection_config=flip_config,
    )

    suite_config = _load_json_config(sample_root / "injection_suite.json")
    suite_spec, suite_records = apply_injections(normal_spec, suite_config)
    suite_timeline = NarrativeSession(suite_spec, "normal_path").run()
    suite_report = {
        "schema": "narrative-workbench/injection-suite/v1",
        "injection_config": suite_config.as_dict(),
        "injection_records": suite_records,
        "operation_kinds": [operation["kind"] for operation in suite_config.operations],
        "baseline_ending": normal_timeline.ending,
        "injected_ending": suite_timeline.ending,
        "injected_timeline": suite_timeline.as_dict(),
        "injected_spec_hash": suite_spec.spec_hash,
    }
    suite_report["report_hash"] = sha256_json(suite_report)

    merge_report = check_all_merge_nodes(merge_spec)

    coefficient_graph = IncrementalDependencyGraph(normal_spec, "normal_path")
    coefficient_key = "e_intro:f_intro_rep:coef:0"
    coefficient_before = {
        node.id: coefficient_graph.cache[node.id].content_hash for node in coefficient_graph.topology
    }
    coefficient_invalidation = coefficient_graph.set_coefficient(coefficient_key, 0.25)
    coefficient_after = {
        node.id: coefficient_graph.cache[node.id].content_hash for node in coefficient_graph.topology
    }
    coefficient_outside = set(coefficient_before) - set(coefficient_invalidation["closure_nodes"])
    coefficient_report = {
        "baseline_node_hashes": coefficient_before,
        "invalidation": coefficient_invalidation,
        "outside_closure_hashes_preserved": all(
            coefficient_before[node_id] == coefficient_after[node_id] for node_id in coefficient_outside
        ),
        "final_snapshots": coefficient_graph.snapshot_payloads(),
        "ending": coefficient_graph.cache["ending:normal_path"].value,
    }

    threshold_graph = IncrementalDependencyGraph(normal_spec, "normal_path")
    threshold_key = "e_ally:c_trust_ally:threshold"
    threshold_before = {
        node.id: threshold_graph.cache[node.id].content_hash for node in threshold_graph.topology
    }
    threshold_invalidation = threshold_graph.set_threshold(threshold_key, 0.5)
    threshold_after = {
        node.id: threshold_graph.cache[node.id].content_hash for node in threshold_graph.topology
    }
    threshold_outside = set(threshold_before) - set(threshold_invalidation["closure_nodes"])
    threshold_report = {
        "baseline_node_hashes": threshold_before,
        "invalidation": threshold_invalidation,
        "outside_closure_hashes_preserved": all(
            threshold_before[node_id] == threshold_after[node_id] for node_id in threshold_outside
        ),
        "final_snapshots": threshold_graph.snapshot_payloads(),
        "ending": threshold_graph.cache["ending:normal_path"].value,
    }

    timelines = {
        "normal": normal_timeline,
        "merge": merge_timeline,
        "boundary": boundary_timeline,
        "injection_baseline": injection_baseline,
        "injection_injected": injection_injected,
    }
    specs = {
        "normal": {"spec_hash": normal_spec.spec_hash, "file": "normal.yaml", "sequence": "normal_path"},
        "merge": {"spec_hash": merge_spec.spec_hash, "file": "merge.yaml", "sequence": "merge_left"},
        "boundary": {"spec_hash": boundary_spec.spec_hash, "file": "boundary.yaml", "sequence": "boundary_path"},
        "injection": {"spec_hash": injection_spec.spec_hash, "file": "injection.yaml", "sequence": "injection_path"},
    }

    dependency_reports = {
        "coefficient_edit": {
            "spec_id": normal_spec.id,
            "sequence_id": "normal_path",
            **coefficient_report,
        },
        "threshold_edit": {
            "spec_id": normal_spec.id,
            "sequence_id": "normal_path",
            **threshold_report,
        },
    }

    manifest = _hash_manifest(
        specs=specs,
        timelines=timelines,
        merge_report=merge_report,
        dependency_reports=dependency_reports,
        injection_flip=flip_report,
        injection_suite=suite_report,
    )

    payload: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "numpy_available": numpy_available(),
            "deterministic_sources": True,
        },
        "samples": {
            "directory": "samples",
            "files": [
                "normal.yaml",
                "merge.yaml",
                "boundary.yaml",
                "injection.yaml",
                "injection_flip.json",
                "injection_suite.json",
                "index.json",
            ],
        },
        "specs": specs,
        "normal": {
            "timeline": normal_timeline.as_dict(),
            "edge_arbitrations": _all_arbitrations(normal_timeline),
            "outgoing_projection_arbitrations": _outgoing_projection_arbitrations(normal_timeline),
        },
        "merge": {
            "timeline": merge_timeline.as_dict(),
            "merge_checks": merge_report,
            "multi_path_ok_status": next(
                check["status"] for check in merge_report["checks"] if check["target_node"] == "m_ok"
            ),
            "multi_path_bad_status": next(
                check["status"] for check in merge_report["checks"] if check["target_node"] == "m_bad"
            ),
            "bad_first_divergence": next(
                check["first_divergence"]
                for check in merge_report["checks"]
                if check["target_node"] == "m_bad"
            ),
        },
        "boundary": {
            "timeline": boundary_timeline.as_dict(),
            "boundary_manifests": _boundary_manifests(boundary_timeline),
        },
        "injection_flip": flip_report,
        "injection_suite": suite_report,
        "dependency_invalidation": dependency_reports,
        "hash_manifest": manifest,
    }
    payload["report_payload_hash"] = sha256_json(payload)
    return payload


def render_markdown(payload: dict[str, Any]) -> str:
    normal = payload["normal"]["timeline"]
    merge = payload["merge"]
    boundary = payload["boundary"]
    flip = payload["injection_flip"]
    coefficient = payload["dependency_invalidation"]["coefficient_edit"]["invalidation"]
    threshold = payload["dependency_invalidation"]["threshold_edit"]["invalidation"]
    bad = merge["bad_first_divergence"]
    boundary_manifest = boundary["boundary_manifests"][0] if boundary["boundary_manifests"] else None
    lines = [
        "# 叙事分支因果工作台演示报告",
        "",
        "本报告由 `python -m story_workbench demo` 在固定样本上确定性生成，不含墙钟时间。",
        "",
        "## 摘要",
        "",
        f"- 正常路线结局：`{normal['ending']}`，状态哈希 `{normal['state_hash'][:16]}…`，重放哈希 `{normal['replay_hash'][:16]}…`。",
        f"- 多路径一致汇合 `m_ok`：`{merge['multi_path_ok_status']}`。",
        f"- 多路径分歧汇合 `m_bad`：`{merge['multi_path_bad_status']}`；首个分歧分量 `{bad['component']}`，差异 `{bad['absolute_difference']!r}`，容差 `{bad['tolerance']!r}`。",
        f"- 注入结局变化：`{flip['baseline']['ending']}` -> `{flip['injected']['ending']}`；首个翻转判定 `{flip['first_decision_flip']['condition_id']}`。",
        "",
        "## 因果与裁决",
        "",
        f"- 正常路线逐边裁决记录：{len(payload['normal']['edge_arbitrations'])} 条。",
        f"- 出边投影裁决记录：{len(payload['normal']['outgoing_projection_arbitrations'])} 条。",
        "- 同边重复写入采用“预执行状态求值 + 分量级最后写者”规则；被覆盖效果仍保留在裁决账本。",
        "",
        "## 判定裕度与边界",
        "",
    ]
    if boundary_manifest is not None:
        judgment = boundary_manifest["judgments"][0]
        lines.extend(
            [
                f"- 边界判定 `{judgment['condition_id']}`：当前值 `{judgment['value']!r}`，阈值 `{judgment['threshold']!r}`，有符号裕度 `{judgment['signed_margin']!r}`。",
                f"- 版本戳 `{boundary_manifest['version_stamp']}`，内容哈希 `{boundary_manifest['content_hash'][:16]}…`。",
                "- 边界判定以独立状态输出，不直接归入可达或不可达。",
            ]
        )
    lines.extend(
        [
            "",
            "## 增量失效闭包",
            "",
            f"- 系数键修改：闭包 {coefficient['closure_size']} 个节点，重算集合与闭包一致：`{coefficient['actual_recomputed_equals_closure']}`。",
            f"- 阈值键修改：闭包 {threshold['closure_size']} 个节点，重算集合与闭包一致：`{threshold['actual_recomputed_equals_closure']}`。",
            f"- 系数修改闭包外哈希保持：`{payload['dependency_invalidation']['coefficient_edit']['outside_closure_hashes_preserved']}`。",
            f"- 阈值修改闭包外哈希保持：`{payload['dependency_invalidation']['threshold_edit']['outside_closure_hashes_preserved']}`。",
            "",
            "## 可重复性",
            "",
            f"- 两次内存构建逐字节一致：`{payload['reproducibility']['byte_identical']}`。",
            f"- 运行 A 哈希：`{payload['reproducibility']['run_a_sha256']}`。",
            f"- 运行 B 哈希：`{payload['reproducibility']['run_b_sha256']}`。",
            "",
            "完整逐节点哈希、状态向量、边际、因果链与注入片段见 `report.json` 与 `node_hashes.json`。",
            "",
        ]
    )
    return "\n".join(lines)


def run_demo(output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    sample_dir = output / "samples"
    output.mkdir(parents=True, exist_ok=True)
    generate_samples(sample_dir)

    payload_a = build_demo_payload(sample_dir)
    text_a = pretty_json(payload_a)
    payload_b = build_demo_payload(sample_dir)
    text_b = pretty_json(payload_b)
    byte_identical = text_a.encode("utf-8") == text_b.encode("utf-8")
    payload_a["reproducibility"] = {
        "byte_identical": byte_identical,
        "run_a_sha256": sha256_text(text_a),
        "run_b_sha256": sha256_text(text_b),
        "run_a_bytes": len(text_a.encode("utf-8")),
        "run_b_bytes": len(text_b.encode("utf-8")),
        "comparison_scope": "canonical JSON report payload before reproducibility metadata",
    }
    final_text = pretty_json(payload_a)
    write_utf8(output / "report.json", final_text)
    write_utf8(output / "node_hashes.json", pretty_json(payload_a["hash_manifest"]))
    write_utf8(output / "report.md", render_markdown(payload_a))
    return {
        "output_dir": str(output),
        "samples_dir": str(sample_dir),
        "report": str(output / "report.json"),
        "node_hashes": str(output / "node_hashes.json"),
        "markdown": str(output / "report.md"),
        "byte_identical": byte_identical,
        "run_a_sha256": payload_a["reproducibility"]["run_a_sha256"],
        "run_b_sha256": payload_a["reproducibility"]["run_b_sha256"],
        "normal_ending": payload_a["normal"]["timeline"]["ending"],
        "merge_ok": payload_a["merge"]["multi_path_ok_status"],
        "merge_bad": payload_a["merge"]["multi_path_bad_status"],
        "flip_ending_before": payload_a["injection_flip"]["baseline"]["ending"],
        "flip_ending_after": payload_a["injection_flip"]["injected"]["ending"],
        "flip_condition": payload_a["injection_flip"]["first_decision_flip"]["condition_id"],
        "coefficient_closure_size": payload_a["dependency_invalidation"]["coefficient_edit"]["invalidation"]["closure_size"],
        "threshold_closure_size": payload_a["dependency_invalidation"]["threshold_edit"]["invalidation"]["closure_size"],
    }
