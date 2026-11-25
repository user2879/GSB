"""Deterministic edit-command application and revision rollback."""

from __future__ import annotations

import copy
from typing import Any

from .errors import E_REVISION_CHAIN, E_VERSION_DECREASED
from .hashing import fingerprint
from .validation import ensure_entry_version_monotonic, validate_command_stream


def _recompute_fixed_point(config: dict[str, Any]) -> None:
    fixed = config["fixed_point"]
    effective_bits = int(fixed["fraction_bits"]) + int(fixed["tier_shifts"][fixed["quantization_tier"]])
    if effective_bits < 0:
        raise ValueError("effective fixed-point width must be non-negative")
    config["effective_fraction_bits"] = effective_bits
    config["fixed_point_scale"] = 1 << effective_bits
    config["high_precision_bits"] = effective_bits + 1
    config["high_precision_scale"] = 1 << (effective_bits + 1)
    config["config_hash"] = fingerprint(
        {
            "concurrency": config["concurrency"],
            "fixed_point": config["fixed_point"],
            "match_budget_points": config["match_budget_points"],
            "doc_quota_points": config["doc_quota_points"],
            "jv1_widths": config["jv1_widths"],
        },
        20,
    )


def apply_command_stream(base_state: dict[str, Any], command_document: dict[str, Any]) -> dict[str, Any]:
    schema_issues = validate_command_stream(command_document)
    if schema_issues:
        from .errors import PipelineValidationError

        raise PipelineValidationError(schema_issues)

    state = copy.deepcopy(base_state)
    lexicon = state["lexicon"]
    base_revision = int(lexicon.get("revision", 0))
    current_revision = base_revision
    snapshots: dict[int, dict[str, Any]] = {
        current_revision: {
            "lexicon": copy.deepcopy(lexicon),
            "weights": copy.deepcopy(state["weights"]),
            "rules": copy.deepcopy(state["rules"]),
        }
    }
    applied: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    def snapshot_next() -> int:
        nonlocal current_revision
        current_revision += 1
        lexicon["revision"] = current_revision
        snapshots[current_revision] = {
            "lexicon": copy.deepcopy(lexicon),
            "weights": copy.deepcopy(state["weights"]),
            "rules": copy.deepcopy(state["rules"]),
        }
        return current_revision

    entries_by_id = {entry["id"]: entry for entry in lexicon["entries"]}

    for command in sorted(command_document["commands"], key=lambda item: int(item["seq"])):
        seq = int(command["seq"])
        op = command["op"]
        record = {"seq": seq, "op": op, "status": "applied"}
        try:
            if op == "upsert_entry":
                incoming = command["entry"]
                existing = entries_by_id.get(incoming["id"])
                issues = ensure_entry_version_monotonic(existing, incoming)
                if issues:
                    failures.append({"seq": seq, "op": op, "code": issues[0].code, "message": issues[0].message})
                    record["status"] = "failed"
                    record["code"] = issues[0].code
                else:
                    if existing is None:
                        lexicon["entries"].append(copy.deepcopy(incoming))
                    else:
                        existing.clear()
                        existing.update(copy.deepcopy(incoming))
                    entries_by_id[incoming["id"]] = lexicon["entries"][-1] if existing is None else existing
                    record["revision"] = snapshot_next()

            elif op == "set_frequency":
                entry = entries_by_id[command["id"]]
                requested_version = command.get("entry_version", int(entry["version"]) + 1)
                issues = ensure_entry_version_monotonic(entry, {"version": requested_version, "id": entry["id"]})
                if issues:
                    failures.append({"seq": seq, "op": op, "code": E_VERSION_DECREASED, "message": issues[0].message})
                    record["status"] = "failed"
                    record["code"] = E_VERSION_DECREASED
                else:
                    entry["frequency"] = int(command["frequency"])
                    entry["version"] = int(requested_version)
                    record["revision"] = snapshot_next()

            elif op == "delete_entry":
                entry = entries_by_id.pop(command["id"])
                lexicon["entries"].remove(entry)
                record["revision"] = snapshot_next()

            elif op == "update_weights":
                if "transition_weights" in command:
                    state["weights"].setdefault("transition_weights", {}).update(copy.deepcopy(command["transition_weights"]))
                else:
                    state["weights"].update(copy.deepcopy(command.get("weights", {})))
                record["weights_fingerprint"] = "updated"

            elif op == "replace_weights":
                state["weights"] = copy.deepcopy(command["weights"])

            elif op == "upsert_rule":
                rules = state["rules"].setdefault("rules", [])
                rule = command["rule"]
                for index, existing_rule in enumerate(rules):
                    if existing_rule["rule_id"] == rule["rule_id"]:
                        rules[index] = copy.deepcopy(rule)
                        break
                else:
                    rules.append(copy.deepcopy(rule))

            elif op == "replace_rules":
                state["rules"] = copy.deepcopy(command["rules"])

            elif op == "rollback_revision":
                target = int(command["revision"])
                if target not in snapshots:
                    failures.append(
                        {
                            "seq": seq,
                            "op": op,
                            "code": E_REVISION_CHAIN,
                            "message": f"rollback target revision {target} has no local snapshot",
                        }
                    )
                    record["status"] = "failed"
                    record["code"] = E_REVISION_CHAIN
                else:
                    restored = copy.deepcopy(snapshots[target])
                    state["lexicon"] = restored["lexicon"]
                    state["weights"] = restored["weights"]
                    state["rules"] = restored["rules"]
                    lexicon = state["lexicon"]
                    entries_by_id = {entry["id"]: entry for entry in lexicon["entries"]}
                    current_revision = target
                    lexicon["revision"] = target
                    record["revision"] = target
                    record["status"] = "rolled_back"

            elif op == "set_exchange_version":
                value = command["version"]
                if value not in {"jv1", "jv2"}:
                    raise ValueError("exchange version must be jv1 or jv2")
                state["exchange_version"] = value
                record["exchange_version"] = value

            elif op == "set_quant_tier":
                tier = command["tier"]
                config = state["config"]
                if tier not in config["fixed_point"]["tier_shifts"]:
                    raise ValueError(f"unknown quantization tier {tier}")
                config["fixed_point"]["quantization_tier"] = tier
                _recompute_fixed_point(config)
                record["effective_fraction_bits"] = config["effective_fraction_bits"]

            elif op == "set_fraction_bits":
                bits = int(command["fraction_bits"])
                if bits < 0:
                    raise ValueError("fraction_bits must be non-negative")
                state["config"]["fixed_point"]["fraction_bits"] = bits
                _recompute_fixed_point(state["config"])
                record["effective_fraction_bits"] = state["config"]["effective_fraction_bits"]

            else:
                raise ValueError(f"unknown command op {op}")

            applied.append(record)
        except Exception as exc:  # deterministic command failure list
            failure = {"seq": seq, "op": op, "code": "E_COMMAND_INVALID", "message": str(exc)}
            failures.append(failure)
            record["status"] = "failed"
            record["code"] = failure["code"]
            applied.append(record)

    failures.sort(key=lambda item: item["seq"])
    return {
        "state": state,
        "applied_commands": applied,
        "command_failures": failures,
        "revision": current_revision,
        "snapshots": snapshots,
    }
