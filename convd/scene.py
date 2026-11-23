"""Scenario and executor-set loading with strong validation (E_SCENE)."""

import hashlib
import json

from .errors import SceneError

MAX_PORT = 65535


def _require(cond, field, message):
    if not cond:
        raise SceneError(field, message)


def _require_int(value, field, minimum=None, maximum=None):
    _require(isinstance(value, int) and not isinstance(value, bool),
             field, "field %s must be an integer" % field)
    if minimum is not None:
        _require(value >= minimum, field,
                 "field %s must be >= %d" % (field, minimum))
    if maximum is not None:
        _require(value <= maximum, field,
                 "field %s must be <= %d" % (field, maximum))
    return value


def _require_str(value, field):
    _require(isinstance(value, str) and value, field,
             "field %s must be a non-empty string" % field)
    return value


def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True)


def content_digest(obj):
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


class Scenario:
    FIELDS = ("scenario_id", "max_rounds", "window_ms", "pulse_interval_ms",
              "lease_ms", "receipt_timeout_ms", "budget_version",
              "max_in_flight", "max_in_flight_per_executor",
              "max_retries", "executor_set_version")

    def __init__(self, data):
        _require(isinstance(data, dict), "scenario", "scenario must be an object")
        self.scenario_id = _require_str(data.get("scenario_id"), "scenario_id")
        self.max_rounds = _require_int(data.get("max_rounds"), "max_rounds", 1)
        self.window_ms = _require_int(data.get("window_ms"), "window_ms", 1)
        self.pulse_interval_ms = _require_int(
            data.get("pulse_interval_ms"), "pulse_interval_ms", 1)
        self.lease_ms = _require_int(data.get("lease_ms"), "lease_ms", 1)
        self.receipt_timeout_ms = _require_int(
            data.get("receipt_timeout_ms"), "receipt_timeout_ms", 1)
        self.budget_version = _require_int(
            data.get("budget_version"), "budget_version", 0)
        self.max_in_flight = _require_int(
            data.get("max_in_flight", 4), "max_in_flight", 1)
        self.max_in_flight_per_executor = _require_int(
            data.get("max_in_flight_per_executor", 1),
            "max_in_flight_per_executor", 1)
        self.max_retries = _require_int(
            data.get("max_retries", 1), "max_retries", 0)
        self.executor_set_version = _require_int(
            data.get("executor_set_version", 1), "executor_set_version", 0)
        _require(self.window_ms >= self.lease_ms, "window_ms",
                 "window_ms must be >= lease_ms")

    def to_dict(self):
        return {name: getattr(self, name) for name in self.FIELDS}

    @property
    def digest(self):
        return content_digest(self.to_dict())


class ExecutorSpec:
    FIELDS = ("executor_id", "port", "target_digest", "seed")

    def __init__(self, data, index):
        _require(isinstance(data, dict), "executors[%d]" % index,
                 "executor entry must be an object")
        self.executor_id = _require_str(
            data.get("executor_id"), "executors[%d].executor_id" % index)
        self.port = _require_int(
            data.get("port"), "executors[%d].port" % index, 1, MAX_PORT)
        self.target_digest = _require_str(
            data.get("target_digest"), "executors[%d].target_digest" % index)
        self.seed = _require_int(
            data.get("seed"), "executors[%d].seed" % index, 0)

    def to_dict(self):
        return {name: getattr(self, name) for name in self.FIELDS}


class ExecutorSet:
    def __init__(self, data):
        _require(isinstance(data, dict), "executors",
                 "executor set must be an object")
        _require(isinstance(data.get("executors"), list), "executors",
                 "executor set must contain a list field 'executors'")
        raw = data["executors"]
        _require(len(raw) > 0, "executors", "executor set must not be empty")
        specs = [ExecutorSpec(item, idx) for idx, item in enumerate(raw)]
        seen_ids = set()
        seen_ports = set()
        for idx, spec in enumerate(specs):
            _require(spec.executor_id not in seen_ids,
                     "executors[%d].executor_id" % idx,
                     "duplicate executor_id %s" % spec.executor_id)
            _require(spec.port not in seen_ports,
                     "executors[%d].port" % idx,
                     "duplicate port %d" % spec.port)
            seen_ids.add(spec.executor_id)
            seen_ports.add(spec.port)
        # Canonical ordering by executor id makes every downstream
        # trajectory independent of declaration / startup order.
        self.specs = tuple(sorted(specs, key=lambda s: s.executor_id))

    def to_dict(self):
        return {"executors": [spec.to_dict() for spec in self.specs]}

    @property
    def digest(self):
        return content_digest(self.to_dict())


def load_scenario(path):
    with open(path, "r", encoding="utf-8") as handle:
        return Scenario(json.load(handle))


def load_executor_set(path):
    with open(path, "r", encoding="utf-8") as handle:
        return ExecutorSet(json.load(handle))
