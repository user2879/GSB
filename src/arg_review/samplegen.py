from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from .canonical import fingerprint


SEED = 20261127


def deterministic_seed_trace(seed: int = SEED) -> dict[str, Any]:
    python_rng = random.Random(seed)
    numpy_rng = np.random.default_rng(seed)
    return {
        "seed": seed,
        "python_random_permille": [python_rng.randint(0, 1000) for _ in range(8)],
        "numpy_random_permille": [
            int(value) for value in numpy_rng.integers(0, 1001, size=8)
        ],
    }


def token_spans(tokens: list[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    for token in tokens:
        start = cursor
        end = start + len(token)
        spans.append((start, end))
        cursor = end + 1
    return spans


def span_for_tokens(tokens: list[str], start_index: int, end_index: int) -> list[int]:
    # The task's offsets are half-open indices into the controlled token sequence.
    # Character spans are derived separately for longest-trigger ranking.
    _ = token_spans(tokens)
    return [start_index, end_index]


def make_arg(
    argument_id: str,
    role: str,
    tokens: list[str],
    start_index: int,
    end_index: int,
    confidence_permille: int,
) -> dict[str, Any]:
    start, end = span_for_tokens(tokens, start_index, end_index)
    return {
        "arg_id": argument_id,
        "role": role,
        "start": start,
        "end": end,
        "confidence_permille": confidence_permille,
    }


def make_event(
    event_id: str,
    event_type: str,
    tokens: list[str],
    trigger_token_span: tuple[int, int],
    arguments: list[dict[str, Any]],
    relations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    trigger_start, trigger_end = span_for_tokens(
        tokens, trigger_token_span[0], trigger_token_span[1]
    )
    return {
        "event_id": event_id,
        "event_type": event_type,
        "tokens": tokens,
        "trigger_span": [trigger_start, trigger_end],
        "arguments": arguments,
        "relations": relations or [],
    }


def relation(
    relation_id: str,
    kind: str,
    source_event: str,
    source_arg: str,
    target_event: str,
    target_arg: str,
) -> dict[str, Any]:
    return {
        "relation_id": relation_id,
        "kind": kind,
        "source": {"event_id": source_event, "arg_id": source_arg},
        "target": {"event_id": target_event, "arg_id": target_arg},
    }


def controlled_vocabulary() -> dict[str, Any]:
    terms = sorted(
        {
            "alpha", "corp", "beta", "labs", "northwind", "group", "meridian",
            "health", "civic", "fund", "university", "ai", "lab", "maya", "chen",
            "zhou", "lin", "dr", "policy", "institute", "and", "at", "of", "for",
            "in", "on", "about", "with", "to", "the", "monday", "shanghai",
            "beijing", "london", "singapore", "acquires", "completes", "acquisition",
            "leads", "800m", "investment", "announces", "420m", "funding", "hosts",
            "meets", "honors", "prize", "chip", "research", "q3", "2026",
            "consortium", "museum",
        }
    )
    return {"seed": SEED, "terms": terms, "fingerprint": fingerprint(terms, 16)}


def schema_revisions() -> dict[str, Any]:
    acquisition = {
        "trigger_terms": ["acquires", "completes acquisition", "acquisition"],
        "roles": ["buyer", "target", "price", "time", "place", "advisor"],
        "required_roles": ["buyer", "target"],
        "mutex_groups": [["buyer", "advisor"]],
    }
    funding = {
        "trigger_terms": ["investment", "funding", "leads investment"],
        "roles": [
            "investor", "lead_investor", "recipient", "amount", "time", "place",
            "topic",
        ],
        "required_roles": ["investor", "recipient"],
        "mutex_groups": [["investor", "lead_investor"]],
    }
    meeting_v1 = {
        "trigger_terms": ["hosts", "meets", "meeting"],
        "roles": ["host", "guest", "place", "time", "topic"],
        "required_roles": ["host"],
        "mutex_groups": [],
    }
    award = {
        "trigger_terms": ["honors", "prize ceremony", "award"],
        "roles": ["winner", "nominee", "presenter", "prize", "place", "time"],
        "required_roles": ["winner"],
        "mutex_groups": [["winner", "nominee"]],
    }
    meeting_v3 = dict(meeting_v1)
    meeting_v3["mutex_groups"] = [["host", "guest"]]
    meeting_v3["required_roles"] = ["host"]
    return {
        "active_revision_id": "rev-3",
        "revisions": [
            {
                "revision_id": "rev-1",
                "parent_revision": None,
                "version": 1,
                "note": "initial deterministic synthesis schema",
                "event_types": {
                    "acquisition": acquisition,
                    "funding": funding,
                    "meeting": meeting_v1,
                    "award": award,
                },
            },
            {
                "revision_id": "rev-2",
                "parent_revision": "rev-1",
                "version": 2,
                "note": "add shanghai meeting trigger",
                "event_types": {
                    "acquisition": acquisition,
                    "funding": funding,
                    "meeting": {
                        **meeting_v1,
                        "trigger_terms": meeting_v1["trigger_terms"]
                        + ["shanghai meeting"],
                    },
                    "award": award,
                },
            },
            {
                "revision_id": "rev-3",
                "parent_revision": "rev-2",
                "version": 3,
                "note": "make host and guest mutually exclusive",
                "event_types": {
                    "acquisition": acquisition,
                    "funding": funding,
                    "meeting": meeting_v3,
                    "award": award,
                },
            },
        ],
    }


def sample_events() -> list[dict[str, Any]]:
    e1_tokens = "alpha corp acquires beta labs in shanghai on monday".split()
    e1 = make_event(
        "E001", "acquisition", e1_tokens, (2, 3),
        [
            make_arg("E001-A1", "buyer", e1_tokens, 0, 2, 920),
            make_arg("E001-A2", "target", e1_tokens, 3, 5, 880),
            make_arg("E001-A3", "place", e1_tokens, 6, 7, 760),
            make_arg("E001-A4", "time", e1_tokens, 8, 9, 700),
        ],
    )

    e2_tokens = (
        "northwind group completes acquisition of meridian health in london".split()
    )
    e2 = make_event(
        "E002", "acquisition", e2_tokens, (2, 4),
        [
            make_arg("E002-A1", "buyer", e2_tokens, 0, 2, 910),
            make_arg("E002-A2", "target", e2_tokens, 5, 7, 870),
            make_arg("E002-A3", "place", e2_tokens, 8, 9, 640),
        ],
    )

    e3_tokens = "alpha corp leads 800m investment in beta labs chip research".split()
    e3 = make_event(
        "E003", "funding", e3_tokens, (4, 5),
        [
            make_arg("E003-A1", "investor", e3_tokens, 0, 2, 930),
            make_arg("E003-A2", "amount", e3_tokens, 3, 4, 900),
            make_arg("E003-A3", "recipient", e3_tokens, 6, 8, 895),
            make_arg("E003-A4", "topic", e3_tokens, 8, 10, 820),
        ],
    )

    e4_tokens = "civic fund announces 420m funding for university ai lab".split()
    e4 = make_event(
        "E004", "funding", e4_tokens, (4, 5),
        [
            make_arg("E004-A1", "investor", e4_tokens, 0, 2, 860),
            make_arg("E004-A2", "amount", e4_tokens, 3, 4, 840),
            make_arg("E004-A3", "recipient", e4_tokens, 6, 9, 900),
        ],
    )

    e5_tokens = (
        "alpha corp hosts maya chen and zhou lin in beijing about ai policy".split()
    )
    e5 = make_event(
        "E005", "meeting", e5_tokens, (2, 3),
        [
            make_arg("E005-A1", "host", e5_tokens, 0, 2, 925),
            make_arg("E005-A2", "guest", e5_tokens, 3, 5, 880),
            make_arg("E005-A3", "guest", e5_tokens, 6, 8, 870),
            make_arg("E005-A4", "place", e5_tokens, 9, 10, 780),
            make_arg("E005-A5", "topic", e5_tokens, 11, 13, 810),
        ],
    )

    e6_tokens = "maya chen meets zhou lin at meridian university in london".split()
    e6 = make_event(
        "E006", "meeting", e6_tokens, (2, 3),
        [
            make_arg("E006-A1", "guest", e6_tokens, 0, 2, 650),
            make_arg("E006-A2", "guest", e6_tokens, 3, 5, 640),
            make_arg("E006-A3", "place", e6_tokens, 9, 10, 620),
        ],
    )

    e7_tokens = "meridian labs honors dr lin with chip prize in singapore".split()
    e7 = make_event(
        "E007", "award", e7_tokens, (2, 3),
        [
            make_arg("E007-A1", "presenter", e7_tokens, 0, 2, 720),
            make_arg("E007-A2", "winner", e7_tokens, 3, 5, 610),
            make_arg("E007-A3", "prize", e7_tokens, 6, 8, 800),
            make_arg("E007-A4", "place", e7_tokens, 9, 10, 770),
        ],
    )

    e1["relations"] = [
        relation("R001", "same_entity", "E001", "E001-A1", "E003", "E003-A1"),
        relation("R002", "same_entity", "E001", "E001-A2", "E003", "E003-A3"),
        relation("R003", "cross_event_ref", "E001", "E001-A1", "E005", "E005-A1"),
    ]
    e5["relations"] = [
        relation("R004", "same_entity", "E005", "E005-A2", "E006", "E006-A1"),
        relation("R005", "same_entity", "E005", "E005-A3", "E006", "E006-A2"),
    ]
    e7["relations"] = [
        relation("R006", "cross_event_ref", "E002", "E002-A2", "E007", "E007-A1"),
    ]
    return [e1, e2, e3, e4, e5, e6, e7]


def invalid_sample_lines() -> list[str]:
    good_tokens = "alpha corp acquires beta labs in shanghai".split()
    bad_type = make_event(
        "X001", "not_a_type", good_tokens, (2, 3),
        [make_arg("X001-A1", "buyer", good_tokens, 0, 2, 900)],
    )

    offset_tokens = "alpha corp acquires beta".split()
    bad_offset = make_event(
        "X002", "acquisition", offset_tokens, (2, 3),
        [make_arg("X002-A1", "buyer", offset_tokens, 0, 2, 900)],
    )
    bad_offset["arguments"][0]["end"] = 999

    conf_tokens = "alpha corp acquires beta".split()
    bad_conf = make_event(
        "X003", "acquisition", conf_tokens, (2, 3),
        [make_arg("X003-A1", "buyer", conf_tokens, 0, 2, 1001)],
    )

    rel_tokens = "alpha corp acquires beta".split()
    bad_rel = make_event(
        "X004", "acquisition", rel_tokens, (2, 3),
        [make_arg("X004-A1", "buyer", rel_tokens, 0, 2, 900)],
        [relation("XR1", "same_entity", "X004", "X004-A1", "E001", "NOPE")],
    )

    role_tokens = "alpha corp hosts maya chen".split()
    bad_role = make_event(
        "X005", "meeting", role_tokens, (2, 3),
        [make_arg("X005-A1", "ghost_role", role_tokens, 0, 2, 900)],
    )

    lines = [
        json.dumps(bad_type, ensure_ascii=False),
        json.dumps(bad_offset, ensure_ascii=False),
        json.dumps(bad_conf, ensure_ascii=False),
        json.dumps(bad_rel, ensure_ascii=False),
        json.dumps(bad_role, ensure_ascii=False),
        '{"event_id":"X006","broken json":',
    ]
    return lines


def default_config() -> dict[str, Any]:
    return {
        "active_revision_id": "rev-3",
        "random_seed": SEED,
        "quantization": {
            "frac_digits": 3,
            "level": "standard",
            "rounding": "half_up",
            "level_multipliers": {"fine": 1, "standard": 5, "coarse": 25},
        },
        "extraction": {
            "window_tokens": 6,
            "fragment_min_tokens": 1,
            "fragment_max_tokens": 4,
            "ideal_fragment_tokens": 2,
            "raw_weight_permille": 500,
            "role_weight_permille": 300,
            "distance_weight_permille": 100,
            "length_weight_permille": 100,
        },
        "role_weights_permille": {
            "buyer": 930,
            "target": 900,
            "investor": 920,
            "lead_investor": 910,
            "recipient": 890,
            "host": 940,
            "guest": 820,
            "winner": 880,
            "nominee": 800,
            "presenter": 760,
            "place": 700,
            "time": 700,
            "amount": 860,
            "topic": 720,
            "price": 850,
            "advisor": 680,
        },
        "decision": {"accept_threshold_permille": 600},
        "merge": {"dedup_threshold_permille": 650},
        "budget": {
            "max_workers": 4,
            "global_arg_score_points": 200,
            "per_event_arg_quota": 30,
            "max_trigger_matches_per_batch": 20,
        },
        "snapshot_dir": "artifacts/snapshots",
        "ledger_path": "artifacts/ledger.json",
    }


def dump_simple_yaml(value: Any, indent: int = 0) -> str:
    pad = " " * indent
    lines: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}{key}:")
                lines.append(dump_simple_yaml(item, indent + 2).rstrip("\n"))
            else:
                lines.append(f"{pad}{key}: {yaml_scalar(item)}")
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                first = True
                for key, child in item.items():
                    prefix = f"{pad}- " if first else f"{pad}  "
                    if isinstance(child, (dict, list)):
                        lines.append(f"{prefix}{key}:")
                        lines.append(dump_simple_yaml(child, indent + 4).rstrip("\n"))
                    else:
                        lines.append(f"{prefix}{key}: {yaml_scalar(child)}")
                    first = False
            else:
                lines.append(f"{pad}- {yaml_scalar(item)}")
    return "\n".join(lines) + "\n"


def yaml_scalar(value: Any) -> str:
    if isinstance(value, str):
        return value if value.replace("_", "").isalnum() else json.dumps(value)
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def edit_commands() -> list[dict[str, Any]]:
    replacement = sample_events()[5]
    replacement["arguments"][2]["confidence_permille"] = 760
    new_event_tokens = "consortium meets at museum in beijing q3".split()
    new_event = make_event(
        "E008", "meeting", new_event_tokens, (1, 2),
        [
            make_arg("E008-A1", "host", new_event_tokens, 0, 1, 830),
            make_arg("E008-A2", "place", new_event_tokens, 3, 4, 770),
            make_arg("E008-A3", "time", new_event_tokens, 5, 6, 720),
        ],
    )
    draft_revision = json.loads(json.dumps(schema_revisions()["revisions"][2]))
    draft_revision["revision_id"] = "rev-draft-delete"
    draft_revision["parent_revision"] = "rev-3"
    draft_revision["version"] = 4
    draft_revision["note"] = "temporary draft used to verify revision deletion"
    return [
        {"op": "set_quantization", "frac_digits": 4, "level": "fine"},
        {"op": "update_weights", "role_weights_permille": {"host": 970}},
        {"op": "set_dedup_threshold", "value": 600},
        {"op": "replace_event", "event": replacement},
        {"op": "add_events", "events": [new_event]},
        {"op": "add_revision", "revision": draft_revision, "activate": False},
        {"op": "delete_revision", "revision_id": "rev-draft-delete"},
        {"op": "update_role_rules", "revision_id": "rev-4", "parent_revision_id": "rev-3",
         "version": 4, "event_type": "meeting",
         "required_roles": [], "mutex_groups": []},
        {"op": "rollback_revision", "revision_id": "rev-2"},
        {"op": "rollback_revision", "revision_id": "rev-4"},
        {"op": "delete_event", "event_id": "E004"},
        {"op": "set_quantization", "frac_digits": 3, "level": "coarse"},
        {"op": "set_budget", "max_workers": 3},
    ]


def injection_config() -> dict[str, Any]:
    return {
        "injections": [
            {
                "injection_id": "INJ-OFFSET-001",
                "event_id": "E001",
                "stage": "validation",
                "fault_type": "offset_out_of_bounds",
                "target_arg_id": "E001-A3",
                "business_code": "E_SCHEMA_INVALID",
            },
            {
                "injection_id": "INJ-CONF-002",
                "event_id": "E003",
                "stage": "argument",
                "fault_type": "bad_confidence",
                "target_arg_id": "E003-A2",
                "business_code": "E_SCHEMA_INVALID",
            },
            {
                "injection_id": "INJ-REL-003",
                "event_id": "E005",
                "stage": "reference",
                "fault_type": "missing_relation_arg",
                "target_relation_id": "R004",
                "business_code": "E_SCHEMA_INVALID",
            },
            {
                "injection_id": "INJ-ROLE-004",
                "event_id": "E007",
                "stage": "adjudication",
                "fault_type": "unknown_role",
                "target_arg_id": "E007-A1",
                "role": "injected_ghost_role",
                "business_code": "E_SCHEMA_INVALID",
            },
            {
                "injection_id": "INJ-CHAIN-005",
                "revision_id": "rev-3",
                "stage": "schema",
                "fault_type": "broken_revision_chain",
                "missing_parent": "rev-missing-injected",
                "business_code": "E_SCHEMA_INVALID",
            },
        ]
    }


def generate_sample_tree(root: str | Path) -> dict[str, Path]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    events_path = root / "events.jsonl"
    invalid_path = root / "invalid_events.jsonl"
    revisions_path = root / "schema_revisions.json"
    config_path = root / "extraction_config.yaml"
    vocab_path = root / "vocabulary.json"
    commands_path = root / "edit_commands.jsonl"
    injections_path = root / "injections.json"
    seed_trace_path = root / "generation_meta.json"

    events = sample_events()
    events_path.write_text(
        "\n".join(json.dumps(event, ensure_ascii=False) for event in events) + "\n",
        encoding="utf-8",
    )
    invalid_path.write_text("\n".join(invalid_sample_lines()) + "\n", encoding="utf-8")
    revisions = schema_revisions()
    revisions_path.write_text(
        json.dumps(revisions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    config = default_config()
    config_path.write_text(
        "# Deterministic extraction configuration; indentation YAML subset.\n"
        + dump_simple_yaml(config),
        encoding="utf-8",
    )
    vocab_path.write_text(
        json.dumps(controlled_vocabulary(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    commands = edit_commands()
    commands_path.write_text(
        "\n".join(json.dumps(cmd, ensure_ascii=False) for cmd in commands) + "\n",
        encoding="utf-8",
    )
    injections_path.write_text(
        json.dumps(injection_config(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    seed_trace = deterministic_seed_trace()
    seed_trace_path.write_text(
        json.dumps(seed_trace, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "events": events_path,
        "invalid_events": invalid_path,
        "revisions": revisions_path,
        "config": config_path,
        "vocabulary": vocab_path,
        "commands": commands_path,
        "injections": injections_path,
        "generation_meta": seed_trace_path,
    }
