"""Experiment declaration loading and strict validation.

Any violation raises ChaosError(E_SPEC) carrying the *first* offending field
path; nothing is partially loaded (the Spec object is only returned when the
whole document validates).
"""

import json

from .errors import ChaosError, E_SPEC

SUPPORTED_INJECTIONS = ("latency", "error_rate", "suspend", "kill")
LOOPBACK = "127.0.0.1"


def _fail(path, msg):
    raise ChaosError(E_SPEC, msg, field=path)


def _require(cond, path, msg):
    if not cond:
        _fail(path, msg)


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _pos_int(d, key, path):
    v = d.get(key)
    _require(_is_int(v) and v > 0, f"{path}.{key}", f"{key} must be a positive integer")
    return v


def _nonneg_int(d, key, path):
    v = d.get(key)
    _require(_is_int(v) and v >= 0, f"{path}.{key}", f"{key} must be a non-negative integer")
    return v


class Service:
    def __init__(self, d, path):
        self.id = d.get("id")
        _require(isinstance(self.id, str) and self.id, f"{path}.id", "service id must be a non-empty string")
        self.host = d.get("host")
        _require(self.host == LOOPBACK, f"{path}.host", "host must be 127.0.0.1 (loopback only)")
        self.port = d.get("port")
        _require(_is_int(self.port) and 0 <= self.port <= 65535, f"{path}.port",
                 "port must be an integer in 0..65535 (0 = allocate at runtime)")
        self.control_port = d.get("control_port")
        _require(_is_int(self.control_port) and 0 <= self.control_port <= 65535,
                 f"{path}.control_port", "control_port must be an integer in 0..65535")
        self.fixed_latency_ms = _nonneg_int(d, "fixed_latency_ms", path)
        self.error_rate_ppm = d.get("error_rate_ppm")
        _require(_is_int(self.error_rate_ppm) and 0 <= self.error_rate_ppm <= 10000,
                 f"{path}.error_rate_ppm", "error_rate_ppm must be an integer in 0..10000")
        self.rebuildable = bool(d.get("rebuildable", True))

    def to_dict(self):
        return {
            "id": self.id, "host": self.host, "port": self.port,
            "control_port": self.control_port,
            "fixed_latency_ms": self.fixed_latency_ms,
            "error_rate_ppm": self.error_rate_ppm,
            "rebuildable": self.rebuildable,
        }


class Injection:
    def __init__(self, d, path, service_ids):
        self.type = d.get("type")
        _require(self.type in SUPPORTED_INJECTIONS, f"{path}.type",
                 f"type must be one of {SUPPORTED_INJECTIONS}")
        self.target = d.get("target")
        _require(self.target in service_ids, f"{path}.target",
                 "target must reference an existing service id")
        self.duration_ms = _pos_int(d, "duration_ms", path)
        self.observe_ms = _pos_int(d, "observe_ms", path)
        params = d.get("params", {})
        _require(isinstance(params, dict), f"{path}.params", "params must be an object")
        if self.type == "latency":
            _pos_int(params, "delay_ms", f"{path}.params")
        elif self.type == "error_rate":
            ppm = params.get("error_rate_ppm")
            _require(_is_int(ppm) and 0 <= ppm <= 10000, f"{path}.params.error_rate_ppm",
                     "error_rate_ppm must be an integer in 0..10000")
        elif self.type == "kill":
            restart = params.get("restart", True)
            _require(isinstance(restart, bool), f"{path}.params.restart", "restart must be a boolean")
        self.params = params

    def to_dict(self):
        return {
            "type": self.type, "target": self.target,
            "duration_ms": self.duration_ms, "observe_ms": self.observe_ms,
            "params": self.params,
        }


class Budget:
    def __init__(self, d, path):
        _require(isinstance(d, dict), path, "budget must be an object")
        self.max_total_injections = _pos_int(d, "max_total_injections", path)
        self.max_parallel = _pos_int(d, "max_parallel", path)
        self.max_observe_ms = _pos_int(d, "max_observe_ms", path)

    def to_dict(self):
        return {
            "max_total_injections": self.max_total_injections,
            "max_parallel": self.max_parallel,
            "max_observe_ms": self.max_observe_ms,
        }


class Sampling:
    def __init__(self, d, path):
        _require(isinstance(d, dict), path, "sampling must be an object")
        self.interval_ms = _pos_int(d, "interval_ms", path)
        self.min_samples = _pos_int(d, "min_samples", path)
        self.scale = _pos_int(d, "scale", path)

    def to_dict(self):
        return {
            "interval_ms": self.interval_ms,
            "min_samples": self.min_samples,
            "scale": self.scale,
        }


class Spec:
    def __init__(self, raw):
        _require(isinstance(raw, dict), "$", "top-level declaration must be an object")
        self.name = raw.get("name", "experiment")
        _require(isinstance(self.name, str), "$.name", "name must be a string")
        seed = raw.get("seed")
        _require(_is_int(seed), "$.seed", "seed must be an integer")
        self.seed = seed
        self.budget = Budget(raw.get("budget"), "$.budget")
        self.sampling = Sampling(raw.get("sampling"), "$.sampling")

        services = raw.get("services")
        _require(isinstance(services, list) and services, "$.services",
                 "services must be a non-empty array")
        self.services = []
        seen_ids, seen_ports = set(), set()
        for i, s in enumerate(services):
            path = f"$.services[{i}]"
            _require(isinstance(s, dict), path, "service must be an object")
            svc = Service(s, path)
            _require(svc.id not in seen_ids, f"{path}.id", f"duplicate service id {svc.id!r}")
            seen_ids.add(svc.id)
            for p, key in ((svc.port, "port"), (svc.control_port, "control_port")):
                if p != 0:
                    _require(p not in seen_ports, f"{path}.{key}", f"duplicate port {p}")
                    seen_ports.add(p)
            self.services.append(svc)

        injections = raw.get("injections")
        _require(isinstance(injections, list) and injections, "$.injections",
                 "injections must be a non-empty array")
        self.injections = []
        for i, inj in enumerate(injections):
            path = f"$.injections[{i}]"
            _require(isinstance(inj, dict), path, "injection must be an object")
            self.injections.append(Injection(inj, path, seen_ids))

    def service_by_id(self, sid):
        for s in self.services:
            if s.id == sid:
                return s
        raise KeyError(sid)

    def to_dict(self):
        return {
            "name": self.name,
            "seed": self.seed,
            "budget": self.budget.to_dict(),
            "sampling": self.sampling.to_dict(),
            "services": [s.to_dict() for s in self.services],
            "injections": [i.to_dict() for i in self.injections],
        }


def load_spec(path):
    """Load and strictly validate a declaration file. All-or-nothing."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ChaosError(E_SPEC, f"declaration is not valid JSON: {exc}", field="$")
    except OSError as exc:
        raise ChaosError(E_SPEC, f"cannot read declaration: {exc}", field="$")
    return Spec(raw)
