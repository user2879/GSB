"""Deterministic local sample specifications and injection configs."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

from .canonical import canonical_json, pretty_json, write_utf8


SCHEMA = "narrative-workbench/spec/v1"
_SPEC_FILES = ("normal.yaml", "merge.yaml", "boundary.yaml", "injection.yaml")


def _common(components: list[str] | None = None) -> list[str]:
    return components or ["trust", "reputation", "fear"]


def _spec(
    *,
    spec_id: str,
    title: str,
    seed: int,
    boundary_tolerance: float,
    merge_tolerance: float,
    start: str,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    components: list[str] | None = None,
    initial_state: dict[str, float] | None = None,
) -> dict[str, Any]:
    components = _common(components)
    return {
        "schema": SCHEMA,
        "id": spec_id,
        "title": title,
        "seed": seed,
        "components": components,
        "initial_state": initial_state or {component: 0.0 for component in components},
        "boundary_tolerance": boundary_tolerance,
        "merge_tolerance": merge_tolerance,
        "start": start,
        "nodes": nodes,
        "edges": edges,
        "samples": samples,
    }


def normal_spec() -> dict[str, Any]:
    return _spec(
        spec_id="normal",
        title="夜港领事馆",
        seed=20260101,
        boundary_tolerance=1e-9,
        merge_tolerance=1e-12,
        start="s_start",
        nodes=[
            {"id": "s_start", "kind": "scene", "title": "夜港"},
            {"id": "s_choice", "kind": "choice", "title": "领事馆门前"},
            {"id": "s_ally", "kind": "scene", "title": "结盟", "ending": "alliance"},
            {"id": "s_wait", "kind": "scene", "title": "等待", "ending": "wait"},
            {"id": "s_fail", "kind": "scene", "title": "失约", "ending": "failure"},
        ],
        edges=[
            {
                "id": "e_intro",
                "from": "s_start",
                "to": "s_choice",
                "label": "穿过雨幕",
                "effects": [
                    {"id": "f_intro_trust_shadow", "target": "trust", "expr": "0.1"},
                    {"id": "f_intro_trust", "target": "trust", "expr": "0.2"},
                    {"id": "f_intro_rep_shadow", "target": "reputation", "expr": "0.05"},
                    {"id": "f_intro_rep", "target": "reputation", "expr": "0.1 + 0.2"},
                    {"id": "f_intro_fear", "target": "fear", "expr": "0.05 * (trust + 1.0)"},
                ],
                "conditions": [],
            },
            {
                "id": "e_ally",
                "from": "s_choice",
                "to": "s_ally",
                "label": "接受盟约",
                "effects": [
                    {"id": "f_ally_trust", "target": "trust", "expr": "0.1"},
                    {"id": "f_ally_rep", "target": "reputation", "expr": "0.2"},
                    {"id": "f_ally_fear", "target": "fear", "expr": "0.05"},
                ],
                "conditions": [
                    {"id": "c_trust_ally", "component": "trust", "op": ">=", "threshold": 0.1}
                ],
            },
            {
                "id": "e_wait",
                "from": "s_choice",
                "to": "s_wait",
                "label": "继续等待",
                "effects": [
                    {"id": "f_wait_fear", "target": "fear", "expr": "0.4"},
                    {"id": "f_wait_trust", "target": "trust", "expr": "0.05"},
                ],
                "conditions": [
                    {"id": "c_trust_wait", "component": "trust", "op": ">=", "threshold": 1.0}
                ],
            },
            {
                "id": "e_fail",
                "from": "s_choice",
                "to": "s_fail",
                "label": "独自离开",
                "effects": [
                    {"id": "f_fail_rep", "target": "reputation", "expr": "0.0"},
                    {"id": "f_fail_fear_shadow", "target": "fear", "expr": "0.1"},
                    {"id": "f_fail_fear", "target": "fear", "expr": "0.2"},
                ],
                "conditions": [],
            },
        ],
        samples=[
            {
                "id": "normal_path",
                "description": "结盟路线",
                "edges": ["e_intro", "e_ally"],
            }
        ],
    )


def merge_spec() -> dict[str, Any]:
    return _spec(
        spec_id="merge",
        title="双路汇合与数值漂移",
        seed=20260102,
        boundary_tolerance=1e-9,
        merge_tolerance=1e-9,
        start="m_start",
        nodes=[
            {"id": "m_start", "kind": "scene", "title": "岔路口"},
            {"id": "m_left", "kind": "scene", "title": "左廊"},
            {"id": "m_right", "kind": "scene", "title": "右廊"},
            {"id": "m_ok", "kind": "scene", "title": "一致汇合", "ending": "merge-ok"},
            {"id": "m_a", "kind": "scene", "title": "甲路"},
            {"id": "m_b", "kind": "scene", "title": "乙路"},
            {"id": "m_bad", "kind": "scene", "title": "分歧汇合", "ending": "merge-bad"},
        ],
        edges=[
            {
                "id": "e_left_start",
                "from": "m_start",
                "to": "m_left",
                "label": "走左廊",
                "effects": [{"id": "f_left_start", "target": "trust", "expr": "0.1"}],
                "conditions": [],
            },
            {
                "id": "e_left_end",
                "from": "m_left",
                "to": "m_ok",
                "label": "左廊尽头",
                "effects": [{"id": "f_left_end", "target": "trust", "expr": "0.2"}],
                "conditions": [],
            },
            {
                "id": "e_right_start",
                "from": "m_start",
                "to": "m_right",
                "label": "走右廊",
                "effects": [{"id": "f_right_start", "target": "trust", "expr": "0.3"}],
                "conditions": [],
            },
            {
                "id": "e_right_end",
                "from": "m_right",
                "to": "m_ok",
                "label": "右廊尽头",
                "effects": [{"id": "f_right_end", "target": "trust", "expr": "0.0"}],
                "conditions": [],
            },
            {
                "id": "e_a_start",
                "from": "m_start",
                "to": "m_a",
                "label": "甲路出发",
                "effects": [{"id": "f_a_start", "target": "trust", "expr": "0.1"}],
                "conditions": [],
            },
            {
                "id": "e_a_end",
                "from": "m_a",
                "to": "m_bad",
                "label": "甲路抵达",
                "effects": [{"id": "f_a_end", "target": "trust", "expr": "0.3"}],
                "conditions": [],
            },
            {
                "id": "e_b_start",
                "from": "m_start",
                "to": "m_b",
                "label": "乙路出发",
                "effects": [{"id": "f_b_start", "target": "trust", "expr": "0.2"}],
                "conditions": [],
            },
            {
                "id": "e_b_end",
                "from": "m_b",
                "to": "m_bad",
                "label": "乙路抵达",
                "effects": [{"id": "f_b_end", "target": "trust", "expr": "0.3"}],
                "conditions": [],
            },
        ],
        samples=[
            {
                "id": "merge_left",
                "description": "左路汇合",
                "edges": ["e_left_start", "e_left_end"],
            },
            {
                "id": "merge_right",
                "description": "右路汇合",
                "edges": ["e_right_start", "e_right_end"],
            },
        ],
    )


def boundary_spec() -> dict[str, Any]:
    return _spec(
        spec_id="boundary",
        title="阈值边界",
        seed=20260103,
        boundary_tolerance=1e-9,
        merge_tolerance=1e-12,
        start="b_start",
        nodes=[
            {"id": "b_start", "kind": "scene", "title": "雨夜"},
            {"id": "b_mid", "kind": "scene", "title": "石阶"},
            {"id": "b_gate", "kind": "choice", "title": "边门"},
            {"id": "b_boundary", "kind": "scene", "title": "边界承诺", "ending": "boundary-ending"},
            {"id": "b_clear", "kind": "scene", "title": "明确路径", "ending": "clear-ending"},
        ],
        edges=[
            {
                "id": "e_b_add_a",
                "from": "b_start",
                "to": "b_mid",
                "label": "第一枚筹码",
                "effects": [{"id": "f_b_a", "target": "trust", "expr": "0.1"}],
                "conditions": [],
            },
            {
                "id": "e_b_add_b",
                "from": "b_mid",
                "to": "b_gate",
                "label": "第二枚筹码",
                "effects": [{"id": "f_b_b", "target": "trust", "expr": "0.2"}],
                "conditions": [],
            },
            {
                "id": "e_b_boundary",
                "from": "b_gate",
                "to": "b_boundary",
                "label": "恰好抵达",
                "effects": [{"id": "f_b_boundary", "target": "fear", "expr": "0.1"}],
                "conditions": [
                    {"id": "c_b_boundary", "component": "trust", "op": ">=", "threshold": 0.3}
                ],
            },
            {
                "id": "e_b_clear",
                "from": "b_gate",
                "to": "b_clear",
                "label": "明确通过",
                "effects": [{"id": "f_b_clear", "target": "reputation", "expr": "0.1"}],
                "conditions": [
                    {"id": "c_b_clear", "component": "trust", "op": ">=", "threshold": 0.2}
                ],
            },
        ],
        samples=[
            {
                "id": "boundary_path",
                "description": "保留边界候选并选择明确分支",
                "edges": ["e_b_add_a", "e_b_add_b", "e_b_clear"],
            }
        ],
    )


def injection_spec() -> dict[str, Any]:
    return _spec(
        spec_id="injection",
        title="阈值注入翻转",
        seed=20260104,
        boundary_tolerance=1e-6,
        merge_tolerance=1e-12,
        start="i_start",
        nodes=[
            {"id": "i_start", "kind": "scene", "title": "提案"},
            {"id": "i_gate", "kind": "choice", "title": "表决"},
            {"id": "i_good", "kind": "scene", "title": "通过", "ending": "good"},
            {"id": "i_alt", "kind": "scene", "title": "搁置", "ending": "quiet"},
        ],
        edges=[
            {
                "id": "e_open",
                "from": "i_start",
                "to": "i_gate",
                "label": "争取支持",
                "effects": [{"id": "f_open_trust", "target": "trust", "expr": "0.6"}],
                "conditions": [],
            },
            {
                "id": "e_good",
                "from": "i_gate",
                "to": "i_good",
                "label": "提交表决",
                "effects": [{"id": "f_good_rep", "target": "reputation", "expr": "0.2"}],
                "conditions": [
                    {"id": "c_trust_good", "component": "trust", "op": ">=", "threshold": 0.5}
                ],
            },
            {
                "id": "e_alt",
                "from": "i_gate",
                "to": "i_alt",
                "label": "撤回提案",
                "effects": [{"id": "f_alt_fear", "target": "fear", "expr": "0.1"}],
                "conditions": [
                    {"id": "c_trust_alt", "component": "trust", "op": "<", "threshold": 0.5}
                ],
            },
        ],
        samples=[
            {
                "id": "injection_path",
                "description": "阈值翻转路线",
                "edges": ["e_open", "e_good"],
            }
        ],
    )


def injection_flip_config() -> dict[str, Any]:
    return {
        "schema": "narrative-workbench/injections/v1",
        "id": "flip-threshold",
        "operations": [
            {
                "kind": "threshold_shift",
                "edge_id": "e_good",
                "condition_id": "c_trust_good",
                "delta": 0.11,
            }
        ],
    }


def injection_suite_config() -> dict[str, Any]:
    return {
        "schema": "narrative-workbench/injections/v1",
        "id": "all-kinds",
        "operations": [
            {
                "kind": "coefficient_perturb",
                "edge_id": "e_intro",
                "effect_id": "f_intro_rep",
                "coefficient_index": 0,
                "mode": "add",
                "value": 0.05,
            },
            {
                "kind": "threshold_shift",
                "edge_id": "e_wait",
                "condition_id": "c_trust_wait",
                "delta": -0.5,
            },
            {"kind": "initial_offset", "component": "fear", "delta": 0.01},
            {"kind": "remove_effects", "edge_id": "e_fail"},
        ],
    }


def sample_documents() -> dict[str, dict[str, Any]]:
    return {
        "normal.yaml": normal_spec(),
        "merge.yaml": merge_spec(),
        "boundary.yaml": boundary_spec(),
        "injection.yaml": injection_spec(),
    }


def generate_samples(directory: str | Path) -> list[Path]:
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    documents = sample_documents()
    for filename in _SPEC_FILES:
        document = documents[filename]
        text = yaml.safe_dump(
            document,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
            width=120,
        )
        path = output / filename
        write_utf8(path, text)
        written.append(path)
    flip_path = output / "injection_flip.json"
    write_utf8(flip_path, pretty_json(injection_flip_config()))
    written.append(flip_path)
    suite_path = output / "injection_suite.json"
    write_utf8(suite_path, pretty_json(injection_suite_config()))
    written.append(suite_path)
    index = {
        "schema": "narrative-workbench/sample-index/v1",
        "files": [path.name for path in written],
        "specs": {
            "normal": {"file": "normal.yaml", "sequence": "normal_path"},
            "merge": {"file": "merge.yaml", "sequence": "merge_left"},
            "boundary": {"file": "boundary.yaml", "sequence": "boundary_path"},
            "injection": {"file": "injection.yaml", "sequence": "injection_path"},
        },
        "injections": {
            "flip": "injection_flip.json",
            "suite": "injection_suite.json",
        },
        "sample_digest": hashlib.sha256(
            "\n".join(canonical_json(documents[filename]) for filename in _SPEC_FILES).encode("utf-8")
        ).hexdigest(),
    }
    index_path = output / "index.json"
    write_utf8(index_path, pretty_json(index))
    written.append(index_path)
    return written
