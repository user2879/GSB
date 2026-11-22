"""State-machine definition and pipeline configuration loading/validation."""

import json
from dataclasses import dataclass, field

import yaml

from .errors import PipelineError
from .hashing import digest
from .models import TransitionRule

BUILTIN_TYPES = {"string", "int", "float", "bool"}
VALID_STRATEGIES = {"overwrite", "larger", "weighted", "reject"}
NUMERIC_KINDS = {"int", "float"}


@dataclass(frozen=True)
class SlotType:
    name: str
    kind: str
    constraints: dict


@dataclass(frozen=True)
class SlotSpec:
    name: str
    type_ref: str
    strategy: str


@dataclass
class FSMDefinition:
    version: int
    states: tuple
    initial_state: str
    slot_types: dict            # name -> SlotType
    slots: dict                 # name -> SlotSpec
    transitions: tuple          # tuple[TransitionRule]

    def fingerprint_payload(self):
        """Stable material used for the FSM fingerprint in cache keys."""
        return {
            "version": self.version,
            "states": list(self.states),
            "initial_state": self.initial_state,
            "slot_types": {
                name: {"kind": st.kind, "constraints": st.constraints}
                for name, st in sorted(self.slot_types.items())
            },
            "slots": {
                name: {"type": spec.type_ref, "strategy": spec.strategy}
                for name, spec in sorted(self.slots.items())
            },
            "transitions": [rule.to_dict() for rule in self.transitions],
        }

    @property
    def fingerprint(self):
        return digest(self.fingerprint_payload())

    def resolve_kind(self, type_ref):
        if type_ref in BUILTIN_TYPES:
            return type_ref, {}
        if type_ref in self.slot_types:
            st = self.slot_types[type_ref]
            return st.kind, st.constraints
        raise PipelineError("E_FSM_INVALID",
                            f"slot type {type_ref!r} cannot be resolved", "load")


def load_fsm(path):
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return parse_fsm(raw)


def parse_fsm(raw):
    if not isinstance(raw, dict):
        raise PipelineError("E_FSM_INVALID", "FSM definition must be a mapping", "load")

    states = raw.get("states")
    if not isinstance(states, list) or not states:
        raise PipelineError("E_FSM_INVALID", "states must be a non-empty list", "load")
    if any(not isinstance(s, str) or not s.strip() for s in states):
        raise PipelineError("E_FSM_INVALID", "every state must be a non-empty string", "load")
    if len(set(states)) != len(states):
        raise PipelineError("E_FSM_INVALID", "states must be unique", "load")
    state_set = set(states)

    initial = raw.get("initial_state")
    if not isinstance(initial, str) or initial not in state_set:
        raise PipelineError("E_FSM_INVALID",
                            f"initial_state {initial!r} must be one of the states", "load")

    slot_types = {}
    for name, spec in (raw.get("slot_types") or {}).items():
        if not isinstance(spec, dict) or "type" not in spec:
            raise PipelineError("E_FSM_INVALID",
                                f"slot type {name!r} must declare a base type", "load")
        kind = spec["type"]
        if kind not in BUILTIN_TYPES:
            raise PipelineError("E_FSM_INVALID",
                                f"slot type {name!r} base kind {kind!r} is unknown", "load")
        constraints = {k: v for k, v in spec.items() if k != "type"}
        slot_types[name] = SlotType(name=name, kind=kind, constraints=constraints)

    slots = {}
    for name, spec in (raw.get("slots") or {}).items():
        if not isinstance(spec, dict):
            raise PipelineError("E_FSM_INVALID", f"slot {name!r} must be a mapping", "load")
        type_ref = spec.get("type")
        strategy = spec.get("strategy")
        if strategy not in VALID_STRATEGIES:
            raise PipelineError("E_FSM_INVALID",
                                f"slot {name!r} strategy {strategy!r} is invalid", "load")
        # Resolve now so an unresolvable reference fails at load time.
        kind, _ = _resolve(type_ref, slot_types)
        if strategy == "weighted" and kind not in NUMERIC_KINDS:
            raise PipelineError("E_FSM_INVALID",
                                f"weighted strategy needs a numeric slot; {name!r} is {kind}",
                                "load")
        if strategy == "larger" and kind not in NUMERIC_KINDS:
            raise PipelineError("E_FSM_INVALID",
                                f"larger strategy needs a numeric slot; {name!r} is {kind}",
                                "load")
        slots[name] = SlotSpec(name=name, type_ref=type_ref, strategy=strategy)

    rules = []
    seen_rule_ids = set()
    for item in raw.get("transitions") or []:
        if not isinstance(item, dict):
            raise PipelineError("E_FSM_INVALID", "every transition must be a mapping", "load")
        rule_id = item.get("id")
        frm, action, to = item.get("from"), item.get("action"), item.get("to")
        if not rule_id or rule_id in seen_rule_ids:
            raise PipelineError("E_FSM_INVALID",
                                f"transition id {rule_id!r} missing or duplicated", "load")
        seen_rule_ids.add(rule_id)
        if frm not in state_set or to not in state_set:
            raise PipelineError("E_FSM_INVALID",
                                f"transition {rule_id!r} references illegal state "
                                f"({frm!r} -> {to!r})", "load")
        if not isinstance(action, str) or not action.strip():
            raise PipelineError("E_FSM_INVALID",
                                f"transition {rule_id!r} needs a non-empty action", "load")
        rules.append(TransitionRule(rule_id=rule_id, from_state=frm,
                                    action=action, to_state=to))

    return FSMDefinition(
        version=int(raw.get("version", 1)),
        states=tuple(states),
        initial_state=initial,
        slot_types=slot_types,
        slots=slots,
        transitions=tuple(rules),
    )


def _resolve(type_ref, slot_types):
    if type_ref in BUILTIN_TYPES:
        return type_ref, {}
    if type_ref in slot_types:
        st = slot_types[type_ref]
        return st.kind, st.constraints
    raise PipelineError("E_FSM_INVALID",
                        f"slot type {type_ref!r} cannot be resolved", "load")


@dataclass
class PipelineConfig:
    version: int
    max_workers: int
    global_budget: int
    default_quota: int
    session_quotas: dict
    confidence_max: int
    fixed_point_decimals: int
    snapshot_dir: str
    name: str = "pipeline"

    def quota_for(self, session_id):
        return int(self.session_quotas.get(session_id, self.default_quota))

    def fingerprint_payload(self):
        return {
            "version": self.version,
            "max_workers": self.max_workers,
            "global_budget": self.global_budget,
            "default_quota": self.default_quota,
            "session_quotas": self.session_quotas,
            "confidence_max": self.confidence_max,
            "fixed_point_decimals": self.fixed_point_decimals,
            "name": self.name,
        }

    @property
    def fingerprint(self):
        return digest(self.fingerprint_payload())


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return parse_config(raw)


def parse_config(raw):
    if not isinstance(raw, dict):
        raise PipelineError("E_CONFIG_INVALID", "pipeline config must be a JSON object", "load")
    try:
        max_workers = int(raw["max_workers"])
        global_budget = int(raw["global_budget"])
        default_quota = int(raw["default_quota"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PipelineError("E_CONFIG_INVALID",
                            f"missing or non-integer numeric setting: {exc}", "load")
    if max_workers < 1:
        raise PipelineError("E_CONFIG_INVALID", "max_workers must be >= 1", "load")
    if global_budget < 0 or default_quota < 1:
        raise PipelineError("E_CONFIG_INVALID",
                            "global_budget must be >= 0 and default_quota >= 1", "load")
    session_quotas = raw.get("session_quotas", {})
    if not isinstance(session_quotas, dict) or any(
            not isinstance(v, int) or v < 1 for v in session_quotas.values()):
        raise PipelineError("E_CONFIG_INVALID",
                            "session_quotas must map session id to positive int", "load")
    confidence_max = int(raw.get("confidence_max", 10000))
    fixed_point_decimals = int(raw.get("fixed_point_decimals", 6))
    if confidence_max < 1 or fixed_point_decimals < 0:
        raise PipelineError("E_CONFIG_INVALID", "bad confidence / fixed-point settings", "load")
    return PipelineConfig(
        version=int(raw.get("version", 1)),
        max_workers=max_workers,
        global_budget=global_budget,
        default_quota=default_quota,
        session_quotas=dict(sorted(session_quotas.items())),
        confidence_max=confidence_max,
        fixed_point_decimals=fixed_point_decimals,
        snapshot_dir=str(raw.get("snapshot_dir", ".snapshot")),
        name=str(raw.get("name", "pipeline")),
    )
