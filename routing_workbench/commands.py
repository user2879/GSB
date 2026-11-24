"""Editable command stream application with deterministic invalidation hints."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from .errors import ErrorCode, Failure
from .io_layer import parse_utterance
from .util import deep_merge


@dataclass
class CommandResult:
    command_id: str
    command_index: int
    command_type: str
    status: str
    dirty_ids: set[str] = field(default_factory=set)
    changed: dict[str, Any] = field(default_factory=dict)
    failures: list[Failure] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "command_id": self.command_id,
            "command_index": self.command_index,
            "command_type": self.command_type,
            "status": self.status,
            "dirty_ids": sorted(self.dirty_ids),
            "changed": self.changed,
            "failure_count": len(self.failures),
        }


def _get(mapping: dict[str, Any], names: tuple[str, ...], default: Any = None) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return default


def _normalize_trigger(raw: Any) -> dict[str, Any]:
    if isinstance(raw, str):
        return {"term": raw, "weight": 1}
    if isinstance(raw, (list, tuple)):
        tokens = [str(token) for token in raw]
        return {"tokens": tokens, "term": "".join(tokens), "weight": 1}
    if isinstance(raw, dict):
        item = copy.deepcopy(raw)
        term = _get(item, ("term", "trigger", "text", "phrase", "pattern", "value"))
        tokens = _get(item, ("tokens", "words", "items", "sequence"))
        if tokens is not None:
            item["tokens"] = [str(token) for token in tokens]
            item.setdefault("term", "".join(item["tokens"]))
        elif term is not None:
            item.setdefault("term", str(term))
        item.setdefault("weight", _get(item, ("w", "score"), 1))
        return item
    return {"term": str(raw), "weight": 1}


def _normalize_trigger_table(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        return [_normalize_trigger({"term": key, "weight": value}) for key, value in raw.items()]
    if isinstance(raw, (list, tuple)):
        return [_normalize_trigger(item) for item in raw]
    return [_normalize_trigger(raw)]


def _normalize_pattern(raw: Any, name: str | None = None) -> dict[str, Any]:
    if isinstance(raw, dict):
        item = copy.deepcopy(raw)
    else:
        item = {"tokens": raw}
    if name is not None:
        item.setdefault("name", name)
    return item


def ensure_config_shape(raw: dict[str, Any]) -> dict[str, Any]:
    """Return a canonical mutable copy while preserving unknown top-level extensions."""
    config = copy.deepcopy(raw)
    vocabulary = _get(config, ("vocabulary", "vocab", "controlled_vocabulary", "terms"))
    if vocabulary is not None:
        config["vocabulary"] = [str(item) for item in vocabulary] if not isinstance(vocabulary, str) else [vocabulary]

    slots = _get(config, ("slots", "slot_table", "slot_declarations"), {})
    if isinstance(slots, list):
        slot_map: dict[str, Any] = {}
        for item in slots:
            if isinstance(item, str):
                slot_map[item] = {}
            elif isinstance(item, dict):
                name = _get(item, ("name", "slot", "id"))
                if name:
                    slot_map[str(name)] = copy.deepcopy(item)
        config["slots"] = slot_map
    elif isinstance(slots, dict):
        config["slots"] = copy.deepcopy(slots)
    else:
        config["slots"] = {}

    patterns_raw = _get(config, ("slot_patterns", "patterns"), {})
    patterns: dict[str, dict[str, Any]] = {}
    if isinstance(patterns_raw, dict):
        for name, item in patterns_raw.items():
            patterns[str(name)] = _normalize_pattern(item, str(name))
    elif isinstance(patterns_raw, list):
        for index, item in enumerate(patterns_raw):
            normalized = _normalize_pattern(item)
            name = normalized.get("name", f"pattern_{index + 1}")
            normalized["name"] = str(name)
            patterns[str(name)] = normalized
    config["slot_patterns"] = patterns

    required = _get(config, ("required_slots", "required_slot_table", "required"), {})
    config["required_slots"] = copy.deepcopy(required) if isinstance(required, dict) else {}

    domains_raw = _get(config, ("domains", "domain_set", "routes"), {})
    domains: dict[str, dict[str, Any]] = {}
    if isinstance(domains_raw, dict):
        domain_items = list(domains_raw.items())
    elif isinstance(domains_raw, list):
        domain_items = []
        for index, item in enumerate(domains_raw):
            if isinstance(item, dict):
                name = _get(item, ("name", "domain", "id"), f"domain_{index + 1}")
                domain_items.append((str(name), item))
    else:
        domain_items = []
    for name, item in domain_items:
        if not isinstance(item, dict):
            domains[str(name)] = {"triggers": [], "intents": {}}
            continue
        domain = copy.deepcopy(item)
        domain["triggers"] = _normalize_trigger_table(_get(domain, ("triggers", "trigger_words", "keywords")))
        intents_raw = _get(domain, ("intents", "intent_set", "routes"), {})
        intents: dict[str, dict[str, Any]] = {}
        if isinstance(intents_raw, dict):
            intent_items = list(intents_raw.items())
        elif isinstance(intents_raw, list):
            intent_items = []
            for index, intent_item in enumerate(intents_raw):
                if isinstance(intent_item, dict):
                    intent_name = _get(intent_item, ("name", "intent", "id"), f"intent_{index + 1}")
                    intent_items.append((str(intent_name), intent_item))
        else:
            intent_items = []
        for intent_name, intent_item in intent_items:
            intent = copy.deepcopy(intent_item) if isinstance(intent_item, dict) else {}
            intent["triggers"] = _normalize_trigger_table(_get(intent, ("triggers", "trigger_words", "keywords")))
            refs = _get(intent, ("slot_patterns", "patterns", "slots"), [])
            if isinstance(refs, str):
                refs = [refs]
            elif isinstance(refs, dict):
                if any(key in refs for key in ("tokens", "pattern", "slot", "slot_name")):
                    refs = [refs]
                else:
                    refs = list(refs.values())
            elif refs is None:
                refs = []
            normalized_refs: list[Any] = []
            for ref_index, ref in enumerate(refs):
                if isinstance(ref, str):
                    normalized_refs.append(ref)
                else:
                    normalized_refs.append(_normalize_pattern(ref, None))
            intent["slot_patterns"] = normalized_refs
            required_for_intent = _get(intent, ("required_slots", "required", "must_have_slots"), None)
            if required_for_intent is None and str(intent_name) in config["required_slots"]:
                required_for_intent = config["required_slots"][str(intent_name)]
            intent["required_slots"] = list(required_for_intent or [])
            intents[str(intent_name)] = intent
        domain["intents"] = intents
        domains[str(name)] = domain
    config["domains"] = domains

    thresholds = _get(config, ("thresholds", "threshold"), {})
    if not isinstance(thresholds, dict):
        thresholds = {}
    if "domain" not in thresholds:
        thresholds["domain"] = _get(config, ("domain_threshold", "level1_threshold"), "0.5")
    if "intent" not in thresholds:
        thresholds["intent"] = _get(config, ("intent_threshold", "level2_threshold"), "0.5")
    config["thresholds"] = thresholds

    negation = _get(config, ("negation_rules", "negations", "negation", "negation_words"), [])
    if isinstance(negation, dict):
        negation = [negation]
    elif isinstance(negation, str):
        negation = [negation]
    config["negation_rules"] = copy.deepcopy(negation)

    recal = _get(config, ("recalibration", "recalibration_weights", "confidence_recalibration"), {})
    config["recalibration"] = copy.deepcopy(recal) if isinstance(recal, dict) else {}

    supplies = {
        "fixed_decimals": ("fixed_decimals", "decimal_places", "scale_decimals", "decimals", 5),
        "quantization_levels": ("quantization_levels", "quant_levels", 10 ** 5),
        "quantization_scale": ("quantization_scale", "score_scale", None),
        "thread_pool_size": ("thread_pool_size", "max_workers", "workers", "thread_count", "threads", 1),
        "max_thread_pool_size": ("max_thread_pool_size", "worker_limit", "max_threads", 32),
        "batch_point_budget": ("batch_point_budget", "route_point_budget", "point_budget", "budget", 1000000),
        "snapshot_interval_commands": ("snapshot_interval_commands", "snapshot_interval", "checkpoint_interval", 1),
    }
    for canonical, keys in supplies.items():
        if canonical in config:
            continue
        *aliases, default = keys
        value = _get(config, tuple(aliases), None)
        if value is not None:
            config[canonical] = value
        elif canonical == "quantization_scale":
            continue
        else:
            config[canonical] = default
    if "quantization_scale" not in config:
        config["quantization_scale"] = config.get("quantization_levels", 100000)
    return config


def _command_payload(command: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in command.items():
        if key in {"command_index", "command_id", "type", "op", "action", "payload", "data", "changes", "patch"}:
            continue
        payload[key] = value
    for key in ("payload", "data", "changes", "patch"):
        value = command.get(key)
        if isinstance(value, dict):
            payload.update(value)
        elif value is not None and not payload:
            payload["value"] = value
    return payload


def _normalize_command_type(command: dict[str, Any]) -> str:
    raw = _get(command, ("type", "op", "action", "command"), "")
    return str(raw).strip().casefold().replace("-", "_").replace(" ", "_")


def _domain_and_intent_names(payload: dict[str, Any]) -> tuple[str | None, str | None]:
    scope = str(_get(payload, ("scope", "level", "target_level", "kind"), "")).casefold()
    domain = _get(payload, ("domain", "domain_name", "domain_id"))
    intent = _get(payload, ("intent", "intent_name", "intent_id"))
    if scope == "domain" and intent is None:
        intent = None
    return (str(domain) if domain is not None else None, str(intent) if intent is not None else None)


def _find_intent(config: dict[str, Any], domain: str | None, intent: str | None) -> tuple[str, str] | None:
    domains = config.get("domains", {})
    if domain is not None:
        if domain in domains and intent in domains[domain].get("intents", {}):
            return domain, str(intent)
        return None
    if intent is None:
        return None
    found: list[tuple[str, str]] = []
    for domain_name, domain_value in domains.items():
        if intent in domain_value.get("intents", {}):
            found.append((domain_name, intent))
    return found[0] if len(found) == 1 else None


def _find_pattern(config: dict[str, Any], name: str) -> dict[str, Any] | None:
    pattern = config.get("slot_patterns", {}).get(name)
    if isinstance(pattern, dict):
        return pattern
    for domain in config.get("domains", {}).values():
        for intent in domain.get("intents", {}).values():
            for ref in intent.get("slot_patterns", []):
                if isinstance(ref, dict) and ref.get("name") == name:
                    return ref
    return None


def _attach_pattern_reference(config: dict[str, Any], domain: str, intent: str, pattern_name: str) -> None:
    refs = config["domains"][domain]["intents"][intent].setdefault("slot_patterns", [])
    if not any((ref == pattern_name) or (isinstance(ref, dict) and ref.get("name") == pattern_name) for ref in refs):
        refs.append(pattern_name)


def _trigger_matches(entry: dict[str, Any], selector: Any, by_id: bool = False) -> bool:
    if isinstance(selector, dict):
        selector = _get(selector, ("entry_id", "id", "term", "trigger", "tokens", "new_term"))
    if by_id:
        return str(entry.get("entry_id", entry.get("id", ""))) == str(selector)
    if isinstance(selector, (list, tuple)):
        entry_tokens = entry.get("tokens")
        return list(entry_tokens or []) == [str(item) for item in selector]
    text = str(selector)
    return str(entry.get("term", "")) == text or "".join(entry.get("tokens", [])) == text


def _failure(command: dict[str, Any], message: str, field: str | None = None) -> Failure:
    return Failure(
        ErrorCode.ROUTE_INVALID.value,
        str(command.get("command_id", "__commands__")),
        "commands",
        message,
        field=field,
        command_index=int(command.get("command_index", -1)),
    )


def apply_command(
    command: dict[str, Any],
    config: dict[str, Any],
    utterances: dict[str, dict[str, Any]],
    all_ids: set[str],
) -> CommandResult:
    command_index = int(command.get("command_index", 0))
    command_id = str(command.get("command_id", f"C{command_index + 1:04d}"))
    command_type = _normalize_command_type(command)
    payload = _command_payload(command)
    result = CommandResult(command_id, command_index, command_type, "applied")
    all_dirty = set(utterances)

    try:
        if command_type in {"replace_utterance", "update_utterance", "edit_utterance", "upsert_utterance", "add_utterance"}:
            raw_utterance = payload.get("utterance")
            if isinstance(raw_utterance, dict):
                raw_patch = copy.deepcopy(raw_utterance)
            else:
                raw_patch = copy.deepcopy(payload)
            target_id = _get(raw_patch, ("utterance_id", "utteranceId", "id", "uid"))
            if target_id is None:
                target_id = _get(payload, ("target", "target_id", "utterance_id"))
            if target_id is None:
                raise ValueError("utterance command requires utterance_id")
            target_id = str(target_id)
            existing = utterances.get(target_id)
            merged = copy.deepcopy(existing) if existing is not None else {}
            merged.update(raw_patch)
            merged.setdefault("utterance_id", target_id)
            parsed = parse_utterance(merged, 0)
            if parsed.utterance_id != target_id:
                raise ValueError("utterance_id in payload does not match target")
            utterances[target_id] = merged
            result.dirty_ids = {target_id}
            result.changed = {"utterance_id": target_id, "operation": "upsert"}

        elif command_type in {"delete_utterance", "remove_utterance"}:
            target_id = _get(payload, ("utterance_id", "utteranceId", "id", "uid", "target", "target_id"))
            if target_id is None:
                raise ValueError("delete_utterance requires utterance_id")
            target_id = str(target_id)
            if target_id not in utterances:
                raise ValueError(f"unknown utterance_id {target_id!r}")
            del utterances[target_id]
            result.dirty_ids = {target_id}
            result.changed = {"utterance_id": target_id, "operation": "delete"}

        elif command_type in {"add_trigger", "update_trigger", "delete_trigger", "remove_trigger", "rename_trigger"}:
            domain_name, intent_name = _domain_and_intent_names(payload)
            selector = _get(payload, ("old", "old_term", "existing", "entry_id", "old_entry_id", "term", "trigger"))
            new_raw = _get(payload, ("new", "new_trigger", "trigger", "term", "tokens", "value"))
            weight = _get(payload, ("weight", "new_weight", "w"))
            if command_type == "add_trigger":
                if new_raw is None:
                    raise ValueError("add_trigger requires trigger/term")
                entry = _normalize_trigger(new_raw)
                if weight is not None:
                    entry["weight"] = weight
            else:
                if new_raw is None and weight is None and selector is None:
                    raise ValueError(f"{command_type} requires a trigger selector")
                entry = _normalize_trigger(new_raw) if new_raw is not None else {}
                if weight is not None:
                    entry["weight"] = weight

            if intent_name is not None:
                target = _find_intent(config, domain_name, intent_name)
                if target is None:
                    raise ValueError(f"unknown intent {domain_name or '*'} / {intent_name}")
                domain_name, intent_name = target
                table = config["domains"][domain_name]["intents"][intent_name].setdefault("triggers", [])
            else:
                if domain_name is None or domain_name not in config.get("domains", {}):
                    raise ValueError("domain trigger command requires an existing domain")
                table = config["domains"][domain_name].setdefault("triggers", [])
            if not isinstance(table, list):
                table = _normalize_trigger_table(table)
                if intent_name is not None:
                    config["domains"][domain_name]["intents"][intent_name]["triggers"] = table
                else:
                    config["domains"][domain_name]["triggers"] = table
            if command_type == "add_trigger":
                table.append(entry)
            else:
                matches = [index for index, item in enumerate(table) if _trigger_matches(item, selector, False)]
                if not matches:
                    matches = [index for index, item in enumerate(table) if _trigger_matches(item, selector, True)]
                if len(matches) != 1:
                    raise ValueError("trigger selector must match exactly one trigger")
                index = matches[0]
                if command_type in {"delete_trigger", "remove_trigger"}:
                    table.pop(index)
                else:
                    updated = {**table[index], **entry}
                    table[index] = updated
            result.dirty_ids = set(all_dirty)
            result.changed = {"domain": domain_name, "intent": intent_name, "operation": command_type}

        elif command_type in {"switch_slot_pattern", "update_slot_pattern", "replace_slot_pattern", "add_slot_pattern"}:
            domain_name, intent_name = _domain_and_intent_names(payload)
            pattern_raw = _get(payload, ("pattern", "new_pattern", "definition", "value"))
            pattern_name = _get(payload, ("pattern_name", "name", "new_name", "target"))
            if isinstance(pattern_raw, dict):
                pattern_name = pattern_name or pattern_raw.get("name")
            elif pattern_name is None and isinstance(pattern_raw, str):
                pattern_name = pattern_raw
            if not pattern_name:
                raise ValueError("slot pattern command requires pattern_name")
            pattern_name = str(pattern_name)
            existing = _find_pattern(config, pattern_name)
            if isinstance(pattern_raw, dict):
                patch = copy.deepcopy(pattern_raw)
            else:
                patch = copy.deepcopy(payload)
                for key in ("command_id", "command_index", "scope", "level", "domain", "intent", "operation"):
                    patch.pop(key, None)
            patch.pop("new_pattern", None)
            patch.pop("pattern", None)
            if existing is None:
                if command_type in {"switch_slot_pattern", "update_slot_pattern", "replace_slot_pattern"}:
                    raise ValueError(f"unknown slot pattern {pattern_name!r}")
                existing = {"name": pattern_name}
                config.setdefault("slot_patterns", {})[pattern_name] = existing
            existing.update(patch)
            existing["name"] = pattern_name
            if intent_name is None:
                intent_name = existing.get("intent")
                domain_name = existing.get("domain", domain_name)
            target = _find_intent(config, domain_name, intent_name) if intent_name else None
            if target is None and intent_name:
                raise ValueError(f"unknown intent for slot pattern: {intent_name}")
            if target:
                existing.setdefault("intent", target[1])
                _attach_pattern_reference(config, target[0], target[1], pattern_name)
            result.dirty_ids = set(all_dirty)
            result.changed = {"pattern": pattern_name, "intent": target[1] if target else None, "operation": command_type}

        elif command_type in {"set_required_slots", "update_required_slots"}:
            domain_name, intent_name = _domain_and_intent_names(payload)
            target = _find_intent(config, domain_name, intent_name)
            if target is None:
                raise ValueError("required-slot command requires an existing intent")
            slots = _get(payload, ("required_slots", "slots", "value", "values"), [])
            if isinstance(slots, str):
                slots = [slots]
            if not isinstance(slots, (list, tuple)):
                raise ValueError("required_slots must be a list")
            config["domains"][target[0]]["intents"][target[1]]["required_slots"] = [str(slot) for slot in slots]
            result.dirty_ids = set(all_dirty)
            result.changed = {"domain": target[0], "intent": target[1], "required_slots": list(slots)}

        elif command_type in {"update_config", "set_config", "patch_config", "set_threshold", "update_thresholds", "set_recalibration", "set_quantization", "set_concurrency", "set_budget"}:
            patch_raw = _get(payload, ("config", "patch", "changes", "value"))
            if isinstance(patch_raw, dict):
                patch = copy.deepcopy(patch_raw)
            else:
                patch = copy.deepcopy(payload)
            if command_type in {"set_threshold", "update_thresholds"}:
                thresholds = config.setdefault("thresholds", {})
                for key in ("domain", "intent", "domain_threshold", "intent_threshold"):
                    if key in patch:
                        normalized_key = "domain" if key.startswith("domain") else "intent"
                        thresholds[normalized_key] = patch[key]
                patch = {}
            elif command_type == "set_recalibration":
                recal = config.setdefault("recalibration", {})
                recal.update(copy.deepcopy(patch))
                patch = {}
            elif command_type == "set_quantization":
                old_scale = config.get("quantization_scale", config.get("quantization_levels"))
                explicit_scale = "quantization_scale" in patch or "score_scale" in patch
                quant_changed = "quantization_levels" in patch or "quant_levels" in patch
                decimals_changed = "fixed_decimals" in patch or "decimal_places" in patch
                for key in ("fixed_decimals", "decimal_places", "quantization_levels", "quant_levels", "quantization_scale", "score_scale"):
                    if key in patch:
                        canonical = "fixed_decimals" if key in {"fixed_decimals", "decimal_places"} else ("quantization_levels" if key in {"quantization_levels", "quant_levels"} else "quantization_scale")
                        config[canonical] = patch[key]
                if not explicit_scale:
                    if quant_changed:
                        config["quantization_scale"] = config["quantization_levels"]
                    elif decimals_changed:
                        decimals = int(config.get("fixed_decimals", 5))
                        config["quantization_levels"] = 10**decimals
                        config["quantization_scale"] = 10**decimals
                    else:
                        config["quantization_scale"] = old_scale
                patch = {}
            elif command_type == "set_concurrency":
                for key in ("thread_pool_size", "max_workers", "workers", "thread_count", "threads"):
                    if key in patch:
                        config["thread_pool_size"] = patch[key]
                if "max_thread_pool_size" in patch:
                    config["max_thread_pool_size"] = patch["max_thread_pool_size"]
                patch = {}
            elif command_type == "set_budget":
                for key in ("batch_point_budget", "route_point_budget", "point_budget", "budget"):
                    if key in patch:
                        config["batch_point_budget"] = patch[key]
                patch = {}
            if patch:
                merged = deep_merge(config, patch)
                # Re-canonicalize in case a nested domain/intent is replaced by a list form.
                merged = ensure_config_shape(merged)
                config.clear()
                config.update(merged)
            result.dirty_ids = set(all_dirty)
            result.changed = {"operation": command_type}

        elif command_type in {"noop", "touch", "checkpoint"}:
            result.dirty_ids = set()
            result.status = "noop"

        else:
            raise ValueError(f"unsupported command type {command_type!r}")

    except (ValueError, TypeError, KeyError, IndexError) as exc:
        result.status = "failed"
        result.dirty_ids = set()
        result.failures.append(_failure(command, str(exc), field=command_type or "command"))
    return result
