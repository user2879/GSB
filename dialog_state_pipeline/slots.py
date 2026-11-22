"""Slot value parsing and deterministic strategy-based merging.

Confidence is a fixed-point integer in ``[0, confidence_max]``.  Weighted
accumulation keeps two integer accumulators (``weighted_sum`` stores
``value * scale * confidence`` so it stays integral) and always processes
operators in event order; the division happens once when the effective value
is read.  The fixed operation order guarantees identical results for the
incremental and full replay paths.
"""

from .config import FSMDefinition
from .errors import PipelineError
from .models import SlotRecord


def parse_value(type_ref, raw, fsm: FSMDefinition):
    """Validate *raw* against the referenced slot type.

    Returns the normalised value.  Raises ``E_SLOT_INVALID`` on any type or
    constraint violation; callers must leave the slot untouched.
    """
    kind, constraints = fsm.resolve_kind(type_ref)
    if kind == "string":
        if not isinstance(raw, str):
            raise PipelineError("E_SLOT_INVALID",
                                f"expected string, got {type(raw).__name__}", "slot")
        value = raw
        max_len = constraints.get("max_length")
        if max_len is not None and len(value) > int(max_len):
            raise PipelineError("E_SLOT_INVALID",
                                f"string length {len(value)} exceeds max_length {max_len}",
                                "slot")
        if "enum" in constraints and value not in constraints["enum"]:
            raise PipelineError("E_SLOT_INVALID",
                                f"{value!r} not in enum {constraints['enum']}", "slot")
        return value
    if kind == "bool":
        if not isinstance(raw, bool):
            raise PipelineError("E_SLOT_INVALID",
                                f"expected bool, got {type(raw).__name__}", "slot")
        return raw
    if kind == "int":
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise PipelineError("E_SLOT_INVALID",
                                f"expected int, got {type(raw).__name__}", "slot")
        value = raw
    else:  # float - ints are accepted as exact floats
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise PipelineError("E_SLOT_INVALID",
                                f"expected number, got {type(raw).__name__}", "slot")
        value = float(raw)
    if "min" in constraints and value < constraints["min"]:
        raise PipelineError("E_SLOT_INVALID",
                            f"value {value} below min {constraints['min']}", "slot")
    if "max" in constraints and value > constraints["max"]:
        raise PipelineError("E_SLOT_INVALID",
                            f"value {value} above max {constraints['max']}", "slot")
    if "enum" in constraints and value not in constraints["enum"]:
        raise PipelineError("E_SLOT_INVALID",
                            f"{value!r} not in enum {constraints['enum']}", "slot")
    return value


def validate_confidence(raw, confidence_max):
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise PipelineError("E_SLOT_INVALID",
                            f"confidence must be an int, got {type(raw).__name__}", "slot")
    if not 0 <= raw <= confidence_max:
        raise PipelineError("E_SLOT_INVALID",
                            f"confidence {raw} outside [0, {confidence_max}]", "slot")
    return raw


def make_records(fsm: FSMDefinition):
    return {
        name: SlotRecord(name=name, type=spec.type_ref, strategy=spec.strategy)
        for name, spec in fsm.slots.items()
    }


def apply_updates(records, updates, turn, batch_id, fsm, config,
                  injection_id=None):
    """Apply one turn's slot updates.

    Every operator is parsed *before* any record is mutated, so a turn with a
    bad operator leaves every slot untouched.  Returns the failure dicts for
    invalid operators (the slot stays unchanged in those cases).
    """
    parsed = []
    failures = []
    for op in updates:
        name = op.get("slot")
        if name not in fsm.slots:
            failures.append(_slot_failure(name, turn, batch_id, injection_id,
                                          f"unknown slot {name!r}"))
            continue
        spec = fsm.slots[name]
        try:
            value = parse_value(spec.type_ref, op.get("value"), fsm)
            confidence = validate_confidence(op.get("confidence", 0),
                                             config.confidence_max)
        except PipelineError as exc:
            failures.append(_slot_failure(name, turn, batch_id,
                                          op.get("injection_id") or injection_id,
                                          str(exc)))
            continue
        parsed.append((name, value, confidence,
                       op.get("injection_id") or injection_id))

    scale = 10 ** config.fixed_point_decimals
    for name, value, confidence, inj_id in parsed:
        record = records[name]
        strategy = record.strategy
        if strategy == "overwrite":
            record.value = value
            record.confidence = confidence
            record.weighted_sum = 0
            record.weight_total = 0
        elif strategy == "larger":
            if record.value is None or value > record.value:
                record.value = value
                record.confidence = confidence
                record.history.append({
                    "turn": turn, "strategy": strategy, "action": "accept",
                    "value": value, "confidence": confidence,
                    "source_batch": batch_id, "injection_id": inj_id,
                })
            else:
                record.history.append({
                    "turn": turn, "strategy": strategy, "action": "skip_not_larger",
                    "value": value, "confidence": confidence,
                    "source_batch": batch_id, "injection_id": inj_id,
                })
            continue
        elif strategy == "weighted":
            scaled = int(round(value * scale))
            # Fixed order: extend accumulators first, effective value read
            # afterwards via effective_weighted_value().
            record.weighted_sum += scaled * confidence
            record.weight_total += confidence
            record.value = effective_weighted_value(record, scale)
            record.confidence = min(config.confidence_max, record.weight_total)
        else:  # reject: first write wins, every later attempt is refused
            if record.value is None:
                record.value = value
                record.confidence = confidence
                record.history.append({
                    "turn": turn, "strategy": strategy, "action": "accept_first",
                    "value": value, "confidence": confidence,
                    "source_batch": batch_id, "injection_id": inj_id,
                })
            else:
                record.history.append({
                    "turn": turn, "strategy": strategy, "action": "reject_overwrite",
                    "value": value, "confidence": confidence,
                    "source_batch": batch_id, "injection_id": inj_id,
                })
            continue
        record.history.append({
            "turn": turn, "strategy": strategy, "action": "merge",
            "value": value, "confidence": confidence,
            "source_batch": batch_id, "injection_id": inj_id,
        })
    return failures


def effective_weighted_value(record, scale):
    if record.weight_total == 0:
        return None
    # Half-up rounding on the integral fixed-point accumulator.
    numerator = record.weighted_sum
    denominator = record.weight_total
    q, r = divmod(abs(numerator), denominator)
    rounded = q + (1 if 2 * r >= denominator else 0)
    if numerator < 0:
        rounded = -rounded
    return rounded / scale


def _slot_failure(slot_name, turn, batch_id, injection_id, message):
    from .models import FailureRecord
    return FailureRecord(
        session_id="", turn=turn, stage="slot", code="E_SLOT_INVALID",
        message=f"slot {slot_name!r}: {message}", batch_id=batch_id,
        injection_id=injection_id,
    ).to_dict()
