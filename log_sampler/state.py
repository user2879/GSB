"""Persistent edit session and command replay."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from .canonical import atomic_write_json, fingerprint, read_json

SESSION_SCHEMA = "lds-session-v1"


class CommandError(ValueError):
    pass


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _bucket_key(device_id: Any, severity: Any) -> str:
    if not isinstance(device_id, str) or not device_id:
        raise CommandError("device_id must be a non-empty string")
    if not isinstance(severity, str) or not severity:
        raise CommandError("severity must be a non-empty string")
    return f"{device_id}|{severity}"


def apply_command(
    config: dict[str, Any],
    batches: list[dict[str, Any]],
    command: dict[str, Any],
) -> None:
    kind = command.get("type")
    if kind in {"append_batch", "replace_batch"}:
        batch = command.get("batch")
        if batch is None:
            batch = {
                "arrival_seq": command.get("arrival_seq"),
                "records": command.get("records"),
            }
        if not isinstance(batch, dict) or not _positive_int(batch.get("arrival_seq")):
            raise CommandError(f"{kind} requires a positive arrival_seq")
        if not isinstance(batch.get("records"), list):
            raise CommandError(f"{kind} requires a records list")
        if kind == "append_batch":
            batches.append(copy.deepcopy(batch))
        else:
            for index, current in enumerate(batches):
                if current.get("arrival_seq") == batch["arrival_seq"]:
                    batches[index] = copy.deepcopy(batch)
                    break
            else:
                batches.append(copy.deepcopy(batch))
        return

    if kind == "delete_log":
        log_id = command.get("log_id")
        if not isinstance(log_id, str) or not log_id:
            raise CommandError("delete_log requires log_id")
        removed = 0
        for batch in batches:
            records = batch.get("records", [])
            if not isinstance(records, list):
                continue
            kept = [record for record in records if record.get("log_id") != log_id]
            removed += len(records) - len(kept)
            batch["records"] = kept
        if removed == 0:
            raise CommandError(f"delete_log target not found: {log_id}")
        return

    if kind in {"set_bucket_capacity", "set_bucket_priority"}:
        bucket = _bucket_key(command.get("device_id"), command.get("severity"))
        overrides = config.setdefault("bucket_overrides", {})
        override = overrides.setdefault(bucket, {})
        if kind == "set_bucket_capacity":
            capacity = command.get("capacity")
            if not _positive_int(capacity):
                raise CommandError("bucket capacity must be a positive integer")
            override["capacity"] = capacity
        else:
            priority = command.get("priority")
            if not _non_negative_int(priority):
                raise CommandError("bucket priority must be a non-negative integer")
            override["priority"] = priority
        return

    if kind in {"set_profile", "set_sampling_profile"}:
        profile = command.get("profile")
        if not isinstance(profile, str) or profile not in config.get("profiles", {}):
            raise CommandError(f"unknown sampling profile: {profile}")
        config["active_profile"] = profile
        return

    if kind == "set_global_budget":
        budget = command.get("global_budget")
        if not _positive_int(budget):
            raise CommandError("global_budget must be a positive integer")
        profile_name = config.get("active_profile", "balanced")
        config["profiles"][profile_name]["global_budget"] = budget
        return

    if kind == "set_quantization":
        decimals = command.get("decimals")
        tier = command.get("tier")
        allowed = config.get("allowed_quantization", {})
        if decimals not in allowed.get("decimals", []):
            raise CommandError(f"decimals {decimals} is outside allowed enum")
        if tier not in allowed.get("tiers", []):
            raise CommandError(f"tier {tier} is outside allowed enum")
        profile_name = command.get("profile", config.get("active_profile", "balanced"))
        profiles = config.get("profiles", {})
        if profile_name not in profiles:
            raise CommandError(f"unknown profile: {profile_name}")
        profiles[profile_name]["quantization"] = {
            "decimals": decimals,
            "tier": tier,
        }
        return

    if kind in {"set_evidence_format", "set_exchange_version"}:
        version = command.get("format", command.get("version", command.get("evidence_format")))
        if version not in {"p1", "p2", "p3"}:
            raise CommandError("evidence format must be p1, p2 or p3")
        config["evidence_format"] = version
        return

    raise CommandError(f"unknown command type: {kind}")


class SessionStore:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.path = self.root / "session.json"
        self.root.mkdir(parents=True, exist_ok=True)

    def initialize(
        self,
        config: dict[str, Any],
        batches: list[dict[str, Any]],
        injections: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if self.path.exists():
            return self.load()
        session = {
            "schema": SESSION_SCHEMA,
            "base_config": copy.deepcopy(config),
            "effective_config": copy.deepcopy(config),
            "batches": copy.deepcopy(batches),
            "injections": copy.deepcopy(injections or []),
            "applied_commands": [],
        }
        atomic_write_json(self.path, session)
        return session

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            raise FileNotFoundError(f"session not initialized: {self.path}")
        value = read_json(self.path)
        if not isinstance(value, dict) or value.get("schema") != SESSION_SCHEMA:
            raise ValueError("session file schema mismatch")
        return value

    def save(self, session: dict[str, Any]) -> None:
        atomic_write_json(self.path, session)

    def apply_commands(self, commands: list[dict[str, Any]]) -> dict[str, Any]:
        session = self.load()
        config = session["effective_config"]
        batches = session["batches"]
        applied = session.setdefault("applied_commands", [])
        applied_by_id = {
            item["id"]: item for item in applied if isinstance(item, dict) and "id" in item
        }
        for index, command in enumerate(commands):
            if not isinstance(command, dict):
                raise CommandError(f"command[{index}] must be an object")
            command_id = str(command.get("id", f"cmd-{index:03d}"))
            command_fp = fingerprint(command)
            previous = applied_by_id.get(command_id)
            if previous is not None:
                if previous.get("fingerprint") != command_fp:
                    raise CommandError(f"command id redefined: {command_id}")
                continue
            try:
                apply_command(config, batches, command)
            except CommandError as exc:
                raise CommandError(f"{command_id}: {exc}") from exc
            record = {
                "id": command_id,
                "index": index,
                "fingerprint": command_fp,
            }
            applied.append(record)
            applied_by_id[command_id] = record
            self.save(session)
        return session

    def effective_input(self) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
        session = self.load()
        return (
            copy.deepcopy(session["effective_config"]),
            copy.deepcopy(session["batches"]),
            copy.deepcopy(session.get("injections", [])),
        )

