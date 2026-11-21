"""Apply edit-plan batches to the mutable config and sample store."""
from __future__ import annotations

from .errors import E_INJECTION_MIGRATION_CONFLICT
from .loader import validate_sample_row
from .models import HierarchyConfig


class EditConflict(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class EditApplier:
    def __init__(self, raw_config: dict, vocab):
        self.raw = raw_config
        self.vocab = vocab

    def current_config(self) -> HierarchyConfig:
        return HierarchyConfig.from_dict(self.raw, self.vocab)

    # ------------------------------------------------------------------ samples
    def apply_samples(self, batch: dict, samples: dict, seen_ids: set,
                      failures: list) -> set:
        affected = set()
        for row in batch.get("add_samples", []):
            sample, row_failures = validate_sample_row(
                row, row.get("_line", -1), seen_ids, self.vocab, self.current_config())
            if row_failures:
                failures.extend(row_failures)
                continue
            seen_ids.add(sample.utterance_id)
            samples[sample.utterance_id] = sample
            affected.add(sample.utterance_id)
        for utterance_id in batch.get("remove_samples", []):
            if utterance_id in samples:
                del samples[utterance_id]
            seen_ids.discard(utterance_id)
            affected.add(utterance_id)
        return affected

    # --------------------------------------------------------------- migrations
    def apply_migrations(self, batch: dict) -> set:
        affected_intents = set()
        for move in batch.get("intent_migrations", []):
            intent = move["intent"]
            target_domain = move["target_domain"]
            source_domain = move.get("source_domain")
            if target_domain not in self.raw["domains"]:
                raise EditConflict(
                    E_INJECTION_MIGRATION_CONFLICT,
                    f"migration target domain unknown: {target_domain}")
            located = None
            for domain, names in self.raw["domain_intents"].items():
                if intent in names:
                    located = domain
                    break
            if located is None:
                raise EditConflict(
                    E_INJECTION_MIGRATION_CONFLICT,
                    f"migration intent not present in hierarchy: {intent}")
            if source_domain is not None and located != source_domain:
                raise EditConflict(
                    E_INJECTION_MIGRATION_CONFLICT,
                    f"migration source conflict: intent {intent} lives in "
                    f"{located}, plan says {source_domain}")
            if target_domain in self.raw["domain_intents"] and \
                    intent in self.raw["domain_intents"][target_domain]:
                raise EditConflict(
                    E_INJECTION_MIGRATION_CONFLICT,
                    f"intent {intent} already mapped to target {target_domain}")
            self.raw["domain_intents"][located].remove(intent)
            self.raw["domain_intents"].setdefault(target_domain, []).append(intent)
            affected_intents.add(intent)
        return affected_intents

    # ------------------------------------------------------------- slot schemas
    def apply_slot_edits(self, batch: dict) -> set:
        affected_intents = set()
        for edit in batch.get("slot_pattern_edits", []):
            intent = edit["intent"]
            if intent not in self._all_intents():
                raise EditConflict(
                    E_INJECTION_MIGRATION_CONFLICT,
                    f"slot edit on unknown intent: {intent}")
            new_slots = edit.get("required_slots")
            if new_slots is not None:
                for slot in new_slots:
                    if slot not in self.raw["slot_definitions"]:
                        from .errors import E_SLOT_SCHEMA_INVALID
                        raise EditConflict(
                            E_SLOT_SCHEMA_INVALID,
                            f"slot pattern references undeclared slot: {slot}")
                self.raw["intent_required_slots"][intent] = list(new_slots)
            value_adds = edit.get("add_slot_values", [])
            for add in value_adds:
                slot = add["slot"]
                if slot not in self.raw["slot_definitions"]:
                    from .errors import E_SLOT_SCHEMA_INVALID
                    raise EditConflict(
                        E_SLOT_SCHEMA_INVALID,
                        f"slot value add references undeclared slot: {slot}")
                existing = self.raw["slot_definitions"][slot]["values"]
                for value in add["values"]:
                    if value not in existing:
                        existing.append(list(value))
            affected_intents.add(intent)
            # Slot-definition edits can affect every intent requiring that slot.
            for add in value_adds:
                for other_intent, slots in self.raw["intent_required_slots"].items():
                    if add["slot"] in slots:
                        affected_intents.add(other_intent)
        return affected_intents

    # --------------------------------------------------------------- thresholds
    def apply_threshold_edits(self, batch: dict) -> dict:
        info = {"changed": False, "level": None}
        edits = batch.get("threshold_edits", {})
        for level in ("domain", "intent"):
            if level in edits:
                value = edits[level]
                from .fixedpoint import is_finite_number
                from decimal import Decimal
                if not is_finite_number(value):
                    from .errors import E_THRESHOLD_NOT_FINITE
                    raise EditConflict(
                        E_THRESHOLD_NOT_FINITE,
                        f"non-finite {level} threshold: {value!r}")
                if not (Decimal("0") <= Decimal(str(value)) <= Decimal("1")):
                    from .errors import E_CONFIG_INVALID
                    raise EditConflict(
                        E_CONFIG_INVALID,
                        f"{level} threshold outside [0,1]: {value}")
                self.raw["thresholds"][level] = str(value)
                info = {"changed": True, "level": level}
        if "active_quantization_level" in edits:
            level = edits["active_quantization_level"]
            if level not in self.raw["quantization_levels"]:
                from .errors import E_CONFIG_INVALID
                raise EditConflict(
                    E_CONFIG_INVALID, f"unknown quantization level: {level}")
            self.raw["active_quantization_level"] = level
            info = {"changed": True, "level": "quantization"}
        for knob in ("completeness_weight", "suppression_weight", "floor"):
            if knob in edits:
                self.raw["recalibration"][knob] = str(edits[knob])
                info = {"changed": True, "level": "recalibration"}
        return info

    def _all_intents(self) -> set:
        result = set()
        for names in self.raw["domain_intents"].values():
            result.update(names)
        return result
