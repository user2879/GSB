from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .canonical import fingerprint
from .yaml_lite import load as load_yaml


class WorkspaceState:
    def __init__(
        self,
        events: list[dict[str, Any]],
        revisions_doc: dict[str, Any],
        config: dict[str, Any],
        vocabulary: dict[str, Any] | list[str],
        injections: list[dict[str, Any]] | None = None,
        global_version: int = 0,
        parse_errors: list[dict[str, Any]] | None = None,
    ) -> None:
        self.events = events
        self.revisions_doc = revisions_doc
        self.config = config
        self.vocabulary = vocabulary
        self.injections = injections or []
        self.global_version = global_version
        self.parse_errors = parse_errors or []

    @property
    def active_revision_id(self) -> str:
        return str(
            self.config.get("active_revision_id")
            or self.revisions_doc.get("active_revision_id")
        )

    @property
    def vocab_terms(self) -> set[str]:
        if isinstance(self.vocabulary, dict):
            return set(self.vocabulary.get("terms", []))
        return set(self.vocabulary)

    def revision_map(self) -> dict[str, dict[str, Any]]:
        return {item["revision_id"]: item for item in self.revisions_doc["revisions"]}

    def config_hash(self) -> str:
        return fingerprint(self.config, 16)

    def state_fingerprint(self) -> str:
        return fingerprint(
            {
                "events": sorted(self.events, key=lambda e: e.get("event_id", "")),
                "revisions": self.revisions_doc,
                "config": self.config,
                "vocabulary": sorted(self.vocab_terms),
                "active_revision_id": self.active_revision_id,
            },
            16,
        )

    def clone(self) -> "WorkspaceState":
        return WorkspaceState(
            events=copy.deepcopy(self.events),
            revisions_doc=copy.deepcopy(self.revisions_doc),
            config=copy.deepcopy(self.config),
            vocabulary=copy.deepcopy(self.vocabulary),
            injections=copy.deepcopy(self.injections),
            global_version=self.global_version,
            parse_errors=copy.deepcopy(self.parse_errors),
        )


def _read_jsonl(path: Path) -> tuple[list[Any], list[dict[str, Any]]]:
    records: list[Any] = []
    parse_errors: list[dict[str, Any]] = []
    if not path.exists():
        return records, parse_errors
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            parse_errors.append(
                {
                    "line_no": line_no,
                    "raw": line,
                    "errors": [f"json_decode:{exc.msg}"],
                    "business_code": "E_SCHEMA_INVALID",
                }
            )
    return records, parse_errors


def load_workspace(
    sample_dir: str | Path,
    injection_path: str | Path | None = None,
    include_invalid: bool = False,
) -> WorkspaceState:
    root = Path(sample_dir)
    events, parse_errors = _read_jsonl(root / "events.jsonl")
    if include_invalid:
        invalid_events, invalid_parse_errors = _read_jsonl(root / "invalid_events.jsonl")
        events.extend(invalid_events)
        parse_errors.extend(invalid_parse_errors)
    revisions_doc = json.loads(
        (root / "schema_revisions.json").read_text(encoding="utf-8")
    )
    config_path = root / "extraction_config.yaml"
    config = load_yaml(config_path) if config_path.exists() else {}
    vocab_path = root / "vocabulary.json"
    vocabulary: Any = (
        json.loads(vocab_path.read_text(encoding="utf-8"))
        if vocab_path.exists()
        else {"terms": sorted({token for event in events for token in event.get("tokens", [])})}
    )
    injections: list[dict[str, Any]] = []
    inj_path = Path(injection_path) if injection_path else root / "injections.json"
    if inj_path.exists():
        injections = json.loads(inj_path.read_text(encoding="utf-8")).get(
            "injections", []
        )
    state = WorkspaceState(
        events=[e for e in events if isinstance(e, dict)],
        revisions_doc=revisions_doc,
        config=config,
        vocabulary=vocabulary,
        injections=injections,
        global_version=0,
        parse_errors=parse_errors,
    )
    return state


def load_commands(path: str | Path) -> list[dict[str, Any]]:
    commands = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            commands.append(json.loads(line))
    return commands


def apply_command(state: WorkspaceState, command: dict[str, Any]) -> WorkspaceState:
    """Apply one edit command and return the same mutated state."""
    op = command["op"]
    state.global_version += 1

    if op == "add_events":
        existing = {event["event_id"] for event in state.events}
        for event in command["events"]:
            if event["event_id"] not in existing:
                state.events.append(copy.deepcopy(event))
                existing.add(event["event_id"])

    elif op == "delete_event":
        state.events = [
            event for event in state.events if event.get("event_id") != command["event_id"]
        ]

    elif op == "replace_event":
        replacement = copy.deepcopy(command["event"])
        state.events = [
            replacement if event.get("event_id") == replacement["event_id"] else event
            for event in state.events
        ]

    elif op == "add_revision":
        revision = copy.deepcopy(command["revision"])
        ids = state.revision_map()
        if revision["revision_id"] in ids:
            raise ValueError(f"revision already exists: {revision['revision_id']}")
        if not revision.get("parent_revision"):
            revision["parent_revision"] = state.active_revision_id
        parent = revision["parent_revision"]
        if parent not in ids:
            raise ValueError(f"parent revision does not exist: {parent}")
        revision.setdefault("version", ids[parent]["version"] + 1)
        state.revisions_doc["revisions"].append(revision)
        if command.get("activate", True):
            state.config["active_revision_id"] = revision["revision_id"]
            state.revisions_doc["active_revision_id"] = revision["revision_id"]

    elif op == "delete_revision":
        revision_id = command["revision_id"]
        revisions = state.revisions_doc["revisions"]
        target = next((r for r in revisions if r["revision_id"] == revision_id), None)
        if target is None:
            return state
        parent = target.get("parent_revision")
        for revision in revisions:
            if revision.get("parent_revision") == revision_id:
                revision["parent_revision"] = parent
        state.revisions_doc["revisions"] = [
            r for r in revisions if r["revision_id"] != revision_id
        ]
        if state.active_revision_id == revision_id:
            state.config["active_revision_id"] = parent
            state.revisions_doc["active_revision_id"] = parent

    elif op == "rollback_revision":
        revision_id = command["revision_id"]
        if revision_id not in state.revision_map():
            raise ValueError(f"unknown rollback revision: {revision_id}")
        state.config["active_revision_id"] = revision_id
        state.revisions_doc["active_revision_id"] = revision_id

    elif op == "update_role_rules":
        event_type = command["event_type"]
        revision_id = command["revision_id"]
        parent_id = command.get("parent_revision_id", state.active_revision_id)
        parent = state.revision_map()[parent_id]
        event_types = copy.deepcopy(parent["event_types"])
        if event_type not in event_types:
            raise ValueError(f"unknown event type for rule update: {event_type}")
        if "required_roles" in command:
            event_types[event_type]["required_roles"] = list(command["required_roles"])
        if "mutex_groups" in command:
            event_types[event_type]["mutex_groups"] = copy.deepcopy(
                command["mutex_groups"]
            )
        if "roles" in command:
            event_types[event_type]["roles"] = list(command["roles"])
        if "trigger_terms" in command:
            event_types[event_type]["trigger_terms"] = list(command["trigger_terms"])
        new_revision = {
            "revision_id": revision_id,
            "parent_revision": parent_id,
            "version": command.get("version", parent["version"] + 1),
            "note": command.get("note", "commanded role rule update"),
            "event_types": event_types,
        }
        state.revisions_doc["revisions"].append(new_revision)
        state.config["active_revision_id"] = revision_id
        state.revisions_doc["active_revision_id"] = revision_id

    elif op == "update_weights":
        state.config.setdefault("role_weights_permille", {}).update(
            command.get("role_weights_permille", {})
        )
        state.config.setdefault("extraction", {}).update(
            command.get("extraction", {})
        )

    elif op == "set_dedup_threshold":
        state.config.setdefault("merge", {})["dedup_threshold_permille"] = int(
            command["value"]
        )

    elif op == "set_accept_threshold":
        state.config.setdefault("decision", {})["accept_threshold_permille"] = int(
            command["value"]
        )

    elif op == "set_quantization":
        quant = state.config.setdefault("quantization", {})
        if "frac_digits" in command:
            quant["frac_digits"] = int(command["frac_digits"])
        if "level" in command:
            quant["level"] = command["level"]
        if "rounding" in command:
            quant["rounding"] = command["rounding"]

    elif op == "set_budget":
        budget = state.config.setdefault("budget", {})
        for key in (
            "max_workers",
            "global_arg_score_points",
            "per_event_arg_quota",
            "max_trigger_matches_per_batch",
        ):
            if key in command:
                budget[key] = int(command[key])

    elif op == "update_vocabulary":
        if isinstance(state.vocabulary, dict):
            terms = state.vocabulary.setdefault("terms", [])
        else:
            terms = state.vocabulary
        for term in command.get("add_terms", []):
            if term not in terms:
                terms.append(term)
        for term in command.get("remove_terms", []):
            if term in terms:
                terms.remove(term)

    else:
        raise ValueError(f"unknown edit command op: {op}")

    return state
