from __future__ import annotations

from pathlib import Path
from typing import Any

from .loader import write_yaml


def _sample(
    sample_id: str,
    template_id: str,
    template_revision: int,
    shard: str,
    fills: list[tuple[str, Any] | tuple[str, Any, int]],
    *,
    answer_template_id: str | None = None,
    answer_template_revision: int | None = None,
) -> dict[str, Any]:
    slot_fills = []
    for fill in fills:
        if len(fill) == 2:
            name, value = fill
            priority = 0
        else:
            name, value, priority = fill
        slot_fills.append({"name": name, "value": value, "priority": priority})
    if answer_template_id is None:
        if template_id == "tpl_summary":
            answer_template_id = "tpl_summary_answer"
            answer_template_revision = 1
        else:
            answer_template_id = template_id
            answer_template_revision = template_revision
    elif answer_template_revision is None:
        answer_template_revision = template_revision
    return {
        "sample_id": sample_id,
        "template_id": template_id,
        "template_revision": template_revision,
        "answer_template_id": answer_template_id,
        "answer_template_revision": answer_template_revision,
        "source_shard": shard,
        "slot_fills": slot_fills,
    }


def generate_workspace(root: Path) -> dict[str, Any]:
    root = Path(root)
    sample_dir = root / "data" / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    for stale in sample_dir.glob("*.yaml"):
        stale.unlink()
    templates = {
        "schema_version": 1,
        "templates": [
            {
                "template_id": "tpl_summary",
                "revision": 1,
                "parent_revision": None,
                "instruction_template": "Summarize {topic} with a {style} tone.",
                "answer_template": "Summary of {topic}: {style}.",
                "slots": [
                    {"name": "topic", "min_length": 3, "max_length": 120},
                    {"name": "style", "allowed_values": ["concise", "detailed"]},
                ],
            },
            {
                "template_id": "tpl_summary",
                "revision": 2,
                "parent_revision": 1,
                "instruction_template": "Summarize {topic} with a {style} tone for {audience}.",
                "answer_template": "Summary for {audience}: {topic} ({style}).",
                "slots": [
                    {"name": "audience", "min_length": 3, "max_length": 60},
                ],
                "constraints": {
                    "style": {"allowed_values": ["concise", "detailed"]},
                },
            },
            {
                "template_id": "tpl_summary",
                "revision": 3,
                "parent_revision": 2,
                "instruction_template": "Give a {style} summary of {topic} for {audience}.",
                "answer_template": "For {audience}, {topic} in {style} form.",
                "slots": [],
            },
            {
                "template_id": "tpl_summary_answer",
                "revision": 1,
                "parent_revision": None,
                "instruction_template": "Answer scaffold for {topic}.",
                "answer_template": "Answer on {topic} in a {style} register.",
                "slots": [{"name": "topic"}, {"name": "style"}],
            },
            {
                "template_id": "tpl_summary_answer",
                "revision": 2,
                "parent_revision": 1,
                "instruction_template": "Answer scaffold for {topic} and {audience}.",
                "answer_template": "Answer on {topic} for {audience} in a {style} register.",
                "slots": [{"name": "audience"}],
            },
            {
                "template_id": "tpl_task",
                "revision": 1,
                "parent_revision": None,
                "instruction_template": "Complete task {task_name} for {owner} by {due}.",
                "answer_template": "Task {task_name} is assigned to {owner} ({due}).",
                "slots": [{"name": "task_name"}, {"name": "owner"}, {"name": "due"}],
            },
            {
                "template_id": "tpl_task",
                "revision": 2,
                "parent_revision": 1,
                "instruction_template": "Complete {priority} task {task_name} for {owner} by {due}.",
                "answer_template": "{priority} task {task_name}: {owner}, due {due}.",
                "slots": [{"name": "priority", "allowed_values": ["normal", "urgent"]}],
            },
            {
                "template_id": "tpl_note",
                "revision": 1,
                "parent_revision": None,
                "instruction_template": "Write a note about {topic}: {body}",
                "answer_template": "Note [{topic}] {body}",
                "slots": [{"name": "topic", "min_length": 2}, {"name": "body", "min_length": 2}],
            },
            {
                "template_id": "tpl_broken",
                "revision": 1,
                "parent_revision": None,
                "instruction_template": "Diagnose {subject}.",
                "answer_template": "Diagnosis: {subject}.",
                "slots": [{"name": "subject"}],
            },
            {
                "template_id": "tpl_broken",
                "revision": 2,
                "parent_revision": 1,
                "instruction_template": "Diagnose {subject} with detail {detail}.",
                "answer_template": "Diagnosis for {subject}: {detail}.",
                "slots": [{"name": "detail"}],
            },
        ],
    }

    batch_001 = {
        "batch_id": "batch-001",
        "samples": [
            _sample("S0001", "tpl_summary", 2, "shard-a", [
                ("topic", "incremental cache validation"),
                ("style", "concise"),
                ("audience", "release engineering"),
            ]),
            _sample("S0002", "tpl_summary", 2, "shard-a", [
                ("topic", "incremental cache validation"),
                ("style", "concise"),
                ("audience", "release engineering"),
            ]),
            _sample("S0003", "tpl_summary", 2, "shard-a", [
                ("topic", "incremental cache verification"),
                ("style", "concise"),
                ("audience", "release engineering"),
            ]),
            _sample("S0004", "tpl_summary", 1, "shard-a", [
                ("topic", "duplicate slot priority"),
                ("style", "detailed"),
                ("style", "concise", 5),
            ]),
            _sample("S0005", "tpl_summary", 3, "shard-a", [
                ("topic", "template rollback"),
                ("style", "detailed"),
                ("audience", "platform team"),
            ]),
            _sample("S0006", "tpl_task", 2, "shard-b", [
                ("task_name", "snapshot recovery"),
                ("owner", "ops"),
                ("due", "2026-12-01"),
                ("priority", "urgent"),
            ]),
            _sample("S0007", "tpl_task", 1, "shard-b", [
                ("task_name", "ledger compaction"),
                ("owner", "data"),
                ("due", "2026-12-03"),
            ]),
            _sample("S0008", "tpl_task", 1, "shard-b", [
                ("task_name", "browser smoke test"),
                ("owner", "qa"),
                ("due", "2026-12-04"),
            ]),
            _sample("S0009", "tpl_note", 1, "shard-b", [
                ("topic", "quota ledger"),
                ("body", "record shard quota decisions"),
            ]),
            _sample("S0010", "tpl_note", 1, "shard-b", [
                ("topic", "render budget"),
                ("body", "record pending queue priority"),
            ]),
        ],
    }
    batch_002 = {
        "batch_id": "batch-002",
        "samples": [
            _sample("S0011", "tpl_summary", 1, "shard-c", [
                ("topic", "fixed point residual"),
                ("style", "detailed"),
            ]),
            _sample("S0012", "tpl_summary", 1, "shard-c", [
                ("topic", "fixed point quantization"),
                ("style", "detailed"),
            ]),
            _sample("S0013", "tpl_summary", 3, "shard-c", [
                ("topic", "atomic snapshot commit"),
                ("style", "concise"),
                ("audience", "storage team"),
            ]),
            _sample("S0014", "tpl_task", 2, "shard-c", [
                ("task_name", "deterministic merge"),
                ("owner", "runtime"),
                ("due", "2026-12-08"),
                ("priority", "normal"),
            ]),
            _sample("S0015", "tpl_note", 1, "shard-c", [
                ("topic", "injection isolation"),
                ("body", "each injected failure remains local"),
            ]),
            _sample("S0016", "tpl_summary", 2, "shard-c", [
                ("topic", "cache key derivation"),
                ("style", "detailed"),
                ("audience", "review board"),
            ]),
            _sample("S-FAIL-CYCLE", "tpl_broken", 2, "shard-failure", [
                ("subject", "cycle"),
                ("detail", "injected template cycle"),
            ]),
            _sample("S-FAIL-MISSING", "tpl_note", 1, "shard-failure", [
                ("topic", "missing slot"),
                ("body", "this body will be removed by injection"),
            ]),
            _sample("S-FAIL-REV", "tpl_summary", 1, "shard-failure", [
                ("topic", "missing revision"),
                ("style", "concise"),
            ]),
            _sample("S-FAIL-NONFINITE", "tpl_summary", 2, "shard-failure", [
                ("topic", "non-finite quality"),
                ("style", "concise"),
                ("audience", "quality lab"),
            ]),
        ],
    }
    config = {
        "schema_version": 1,
        "dedupe": {"threshold_permille": 720, "subword_size": 3},
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
    injections = {
        "schema_version": 1,
        "injections": [
            {
                "id": "INJ-CYCLE-001",
                "type": "template_cycle",
                "template_id": "tpl_broken",
                "revision": 2,
                "parent_revision": 2,
            },
            {
                "id": "INJ-MISSING-002",
                "type": "missing_slot",
                "sample_id": "S-FAIL-MISSING",
                "slot": "body",
            },
            {
                "id": "INJ-REVISION-003",
                "type": "missing_revision",
                "sample_id": "S-FAIL-REV",
                "revision": 99,
            },
            {
                "id": "INJ-NONFINITE-004",
                "type": "nonfinite_quality",
                "sample_id": "S-FAIL-NONFINITE",
            },
        ],
    }
    commands = {
        "schema_version": 1,
        "commands": [
            {
                "id": "CMD-001",
                "op": "add_samples",
                "batch_id": "batch-003",
                "samples": [
                    _sample("S0100", "tpl_summary", 2, "shard-d", [
                        ("topic", "rolling batch arrival"),
                        ("style", "concise"),
                        ("audience", "ingest team"),
                    ]),
                    _sample("S0101", "tpl_note", 1, "shard-d", [
                        ("topic", "new source shard"),
                        ("body", "quota applies per source shard"),
                    ]),
                ],
            },
            {
                "id": "CMD-002",
                "op": "replace_sample",
                "sample_id": "S0003",
                "sample": _sample("S0003", "tpl_summary", 2, "shard-a", [
                    ("topic", "incremental cache verification updated"),
                    ("style", "detailed"),
                    ("audience", "release engineering"),
                ]),
            },
            {
                "id": "CMD-003",
                "op": "delete_sample",
                "sample_id": "S0005",
            },
            {
                "id": "CMD-004",
                "op": "upsert_template_revision",
                "revision_data": {
                    "template_id": "tpl_summary",
                    "revision": 4,
                    "parent_revision": 3,
                    "instruction_template": "Give a {style} summary of {topic} for {audience} as {format}.",
                    "answer_template": "For {audience}, {topic} in {style} {format} form.",
                    "slots": [
                        {"name": "format", "allowed_values": ["bullet", "prose"]},
                    ],
                },
            },
            {
                "id": "CMD-005",
                "op": "add_samples",
                "batch_id": "batch-004",
                "samples": [
                    _sample("S0102", "tpl_summary", 4, "shard-d", [
                        ("topic", "template descendant invalidation"),
                        ("style", "concise"),
                        ("audience", "cache team"),
                        ("format", "bullet"),
                    ]),
                ],
            },
            {
                "id": "CMD-006",
                "op": "set_slot_constraint",
                "template_id": "tpl_summary",
                "revision": 2,
                "slot": "style",
                "fields": {"max_length": 30},
            },
            {
                "id": "CMD-007",
                "op": "set_dedupe_threshold",
                "value": 800,
            },
            {
                "id": "CMD-008",
                "op": "set_quantization",
                "quantization": "Q10000",
                "decimal_places": 6,
            },
            {
                "id": "CMD-009",
                "op": "rollback_template",
                "template_id": "tpl_task",
                "to_revision": 1,
            },
            {
                "id": "CMD-010",
                "op": "set_scheduler",
                "max_workers": 3,
                "render_point_budget": 10_000,
                "source_quota_points": 10_000,
            },
        ],
    }
    write_yaml(root / "data" / "templates.yaml", templates)
    write_yaml(sample_dir / "batch-001.yaml", batch_001)
    write_yaml(sample_dir / "batch-002.yaml", batch_002)
    write_yaml(root / "data" / "injections.yaml", injections)
    write_yaml(root / "config" / "pipeline.yaml", config)
    write_yaml(root / "commands" / "edit-sequence.yaml", commands)
    return {
        "template_revisions": len(templates["templates"]),
        "samples": len(batch_001["samples"]) + len(batch_002["samples"]),
        "injections": len(injections["injections"]),
        "commands": len(commands["commands"]),
    }
