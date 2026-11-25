from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from edge_ledger.loader import write_yaml
from edge_ledger.samplegen import generate_workspace


BASE_CONFIG: dict[str, Any] = {
    "schema_version": 1,
    "dedupe": {"threshold_permille": 800, "subword_size": 3},
    "quality": {
        "weights_permille": {
            "completeness": 400,
            "template_coverage": 300,
            "answer_length": 300,
        },
        "answer_soft_limit_chars": 24,
        "answer_decay_per_char": 4_000,
        "answer_floor": 150_000,
    },
    "fixed_point": {"decimal_places": 4, "quantization": "Q1000"},
    "scheduler": {
        "max_workers": 4,
        "render_point_budget": 10_000,
        "source_quota_points": 10_000,
        "points_per_render": 1,
    },
}


def make_sample_workspace(root: Path) -> Path:
    generate_workspace(root)
    return root


def write_custom_workspace(
    root: Path,
    *,
    templates: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
    injections: list[dict[str, Any]] | None = None,
    sample_batches: list[dict[str, Any]] | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    write_yaml(root / "data" / "templates.yaml", {"schema_version": 1, "templates": templates})
    write_yaml(root / "config" / "pipeline.yaml", copy.deepcopy(config or BASE_CONFIG))
    write_yaml(
        root / "data" / "injections.yaml",
        {"schema_version": 1, "injections": injections or []},
    )
    batches = sample_batches or [{"batch_id": "batch-001", "samples": samples}]
    for index, batch in enumerate(batches, start=1):
        write_yaml(root / "data" / "samples" / f"batch-{index:03d}.yaml", batch)
    return root


def simple_template(
    *,
    template_id: str = "tpl",
    revision: int = 1,
    parent: int | None = None,
    instruction: str = "Do {topic}",
    answer: str = "Did {topic}",
    slots: list[dict[str, Any]] | None = None,
    constraints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = {
        "template_id": template_id,
        "revision": revision,
        "parent_revision": parent,
        "instruction_template": instruction,
        "answer_template": answer,
        "slots": slots if slots is not None else [{"name": "topic"}],
    }
    if constraints:
        result["constraints"] = constraints
    return result


def simple_sample(
    sample_id: str = "S1",
    *,
    template_id: str = "tpl",
    revision: int = 1,
    shard: str = "shard-a",
    fills: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "template_id": template_id,
        "template_revision": revision,
        "answer_template_id": template_id,
        "answer_template_revision": revision,
        "source_shard": shard,
        "slot_fills": fills if fills is not None else [{"name": "topic", "value": "alpha"}],
    }

