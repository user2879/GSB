"""Independent fault-injection configuration.

The injection file is deliberately separate from the normal pipeline config so
it can be swapped without touching production parameters.  Each item carries a
stable ``injection_id`` that is stamped onto every event/failure it creates and
surfaced in the diagnostic report, proving the hit.

Supported fault kinds (real input faults, never mock exceptions):

* ``bad_slot_value``   -> triggers E_SLOT_INVALID during merging
* ``turn_jump``        -> triggers E_STREAM_INVALID during materialisation
* ``unknown_transition`` -> triggers E_TRANSITION_REJECTED at advance time
* ``quota_override``   -> forces a tiny per-session quota -> E_QUOTA_EXCEEDED
"""

import copy
import json

from .errors import PipelineError

FAULT_KINDS = {"bad_slot_value", "turn_jump", "unknown_transition", "quota_override"}


def load_injection(path):
    if path is None:
        return {"injections": []}
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict) or not isinstance(raw.get("injections", []), list):
        raise PipelineError("E_CONFIG_INVALID",
                            "injection file must contain an 'injections' list", "load")
    seen = set()
    for item in raw["injections"]:
        inj_id = item.get("injection_id")
        if not inj_id or inj_id in seen:
            raise PipelineError("E_CONFIG_INVALID",
                                "every injection needs a unique injection_id", "load")
        seen.add(inj_id)
        if item.get("kind") not in FAULT_KINDS:
            raise PipelineError("E_CONFIG_INVALID",
                                f"injection {inj_id} has unknown kind {item.get('kind')!r}",
                                "load")
    return raw


def apply_injections(materialised, injection_config):
    """Mutate the materialised turn stream according to injection items.

    Returns ``(materialised, applied_ids)``.  Items that cannot find their
    target session/batch are skipped and reported by the caller via the
    manifest of requested ids.
    """
    materialised = copy.deepcopy(materialised)
    applied = []
    for item in injection_config.get("injections", []):
        kind = item["kind"]
        sid = item["session_id"]
        batch_id = item.get("batch_id", "")
        inj_id = item["injection_id"]
        if sid not in materialised:
            continue
        turns = materialised[sid]
        if kind == "quota_override":
            applied.append(inj_id)
            continue
        target = _find_turn(turns, batch_id, item.get("turn"))
        if target is None:
            continue
        if kind == "bad_slot_value":
            target["slot_updates"].append({
                "slot": item["slot"],
                "value": item["bad_value"],
                "confidence": item.get("confidence", 10000),
                "injection_id": inj_id,
            })
        elif kind == "turn_jump":
            # Create a real hole by renumbering to a non-adjacent turn.
            target["turn"] = target["turn"] + int(item.get("gap", 5))
        elif kind == "unknown_transition":
            target["action"] = item.get("action", "__injected_unknown_action__")
        target["injection_id"] = inj_id
        applied.append(inj_id)
    # turn_jump injections break density; rebuild the raw view by re-validating
    # through the same rule the loader uses (the pipeline runner does this).
    return materialised, applied


def quota_overrides(injection_config):
    return {
        item["session_id"]: int(item["quota"])
        for item in injection_config.get("injections", [])
        if item["kind"] == "quota_override"
    }


def quota_injection_ids(injection_config):
    return {
        item["session_id"]: item["injection_id"]
        for item in injection_config.get("injections", [])
        if item["kind"] == "quota_override"
    }


def _find_turn(turns, batch_id, turn_no):
    if turn_no is not None:
        for turn in turns:
            if turn["turn"] == turn_no:
                return turn
    for turn in turns:
        if turn["batch_id"] == batch_id:
            return turn
    return None
