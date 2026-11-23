from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .canonical import canonical_json, is_int, sha256_value
from .errors import LabError


QUANT_PROFILES: dict[str, dict[str, Any]] = {
    "fp32-v1": {"bits": None, "version": 1, "description": "IEEE-754 float32 values"},
    "int8-sym-v1": {"bits": 8, "version": 1, "description": "symmetric int8 weight quantization"},
    "int4-sym-v1": {"bits": 4, "version": 1, "description": "symmetric int4 weight quantization"},
}

BACKEND_KEYS = {
    "backend_id",
    "host",
    "port",
    "model_dim",
    "hidden_dim",
    "seed",
    "quant_profile",
}
RULE_KEYS = {
    "rule_id",
    "request_prefix",
    "stable_backend",
    "canary_backend",
    "canary_weight_bp",
    "batch_size",
}
SAMPLE_KEYS = {
    "sample_id",
    "sample_seq",
    "client_key",
    "rule_id",
    "request_path",
    "input_seed",
}


@dataclass(frozen=True)
class BackendSpec:
    backend_id: str
    host: str
    port: int
    model_dim: int
    hidden_dim: int
    seed: int
    quant_profile: str

    def declaration(self) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "host": self.host,
            "port": self.port,
            "model_dim": self.model_dim,
            "hidden_dim": self.hidden_dim,
            "seed": self.seed,
            "quant_profile": self.quant_profile,
        }

    def evolve(self, **changes: Any) -> "BackendSpec":
        values = self.declaration()
        values.update(changes)
        return BackendSpec(**values)


@dataclass(frozen=True)
class RuleSpec:
    rule_id: str
    request_prefix: str
    stable_backend: str
    canary_backend: str
    canary_weight_bp: int
    batch_size: int

    def declaration(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "request_prefix": self.request_prefix,
            "stable_backend": self.stable_backend,
            "canary_backend": self.canary_backend,
            "canary_weight_bp": self.canary_weight_bp,
            "batch_size": self.batch_size,
        }


@dataclass(frozen=True)
class SampleSpec:
    sample_id: str
    sample_seq: int
    client_key: str
    rule_id: str
    request_path: str
    input_seed: int

    def declaration(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "sample_seq": self.sample_seq,
            "client_key": self.client_key,
            "rule_id": self.rule_id,
            "request_path": self.request_path,
            "input_seed": self.input_seed,
        }


@dataclass(frozen=True)
class LoadedSpec:
    backends: tuple[BackendSpec, ...]
    rules: tuple[RuleSpec, ...]
    samples: tuple[SampleSpec, ...]

    @property
    def backends_by_id(self) -> dict[str, BackendSpec]:
        return {item.backend_id: item for item in self.backends}

    @property
    def rules_by_id(self) -> dict[str, RuleSpec]:
        return {item.rule_id: item for item in self.rules}

    @property
    def samples_by_id(self) -> dict[str, SampleSpec]:
        return {item.sample_id: item for item in self.samples}

    def backend_version(self) -> str:
        return sha256_value([item.declaration() for item in sorted(self.backends, key=lambda x: x.backend_id)])

    def rules_version(self) -> str:
        return sha256_value([item.declaration() for item in sorted(self.rules, key=lambda x: x.rule_id)])

    def samples_version(self) -> str:
        return sha256_value([item.declaration() for item in sorted(self.samples, key=lambda x: x.sample_seq)])


def _spec_error(message: str, field: str) -> LabError:
    return LabError("E_SPEC", message, field=field)


def _require_object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _spec_error("must be a JSON object", field)
    return value


def _require_list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise _spec_error("must be a JSON array", field)
    return value


def _require_key(obj: dict[str, Any], key: str, field: str) -> Any:
    if key not in obj:
        raise _spec_error("required field is missing", f"{field}.{key}")
    return obj[key]


def _require_nonempty_string(obj: dict[str, Any], key: str, field: str) -> str:
    value = _require_key(obj, key, field)
    if not isinstance(value, str) or not value:
        raise _spec_error("must be a non-empty string", f"{field}.{key}")
    if "\x00" in value:
        raise _spec_error("must not contain NUL", f"{field}.{key}")
    return value


def _require_positive_int(obj: dict[str, Any], key: str, field: str, maximum: int | None = None) -> int:
    value = _require_key(obj, key, field)
    if not is_int(value) or value <= 0:
        raise _spec_error("must be a positive integer", f"{field}.{key}")
    if maximum is not None and value > maximum:
        raise _spec_error(f"must be <= {maximum}", f"{field}.{key}")
    return value


def _reject_unknown_keys(obj: dict[str, Any], allowed: set[str], field: str) -> None:
    for key in sorted(obj):
        if key not in allowed:
            raise _spec_error("unknown field", f"{field}.{key}")


def validate_backend(raw: Any, index: int, seen_ids: set[str], seen_ports: set[int]) -> BackendSpec:
    field = f"backends[{index}]"
    obj = _require_object(raw, field)
    _reject_unknown_keys(obj, BACKEND_KEYS, field)
    backend_id = _require_nonempty_string(obj, "backend_id", field)
    if backend_id in seen_ids:
        raise _spec_error("duplicate backend_id", f"{field}.backend_id")
    seen_ids.add(backend_id)
    host = _require_nonempty_string(obj, "host", field)
    if host != "127.0.0.1":
        raise _spec_error("host must be the literal 127.0.0.1", f"{field}.host")
    port = _require_key(obj, "port", field)
    if not is_int(port) or port < 1024 or port > 65535:
        raise _spec_error("must be an integer high port in 1024..65535", f"{field}.port")
    if port in seen_ports:
        raise _spec_error("port duplicates another backend port", f"{field}.port")
    seen_ports.add(port)
    model_dim = _require_positive_int(obj, "model_dim", field, maximum=4096)
    hidden_dim = _require_positive_int(obj, "hidden_dim", field, maximum=8192)
    seed = _require_key(obj, "seed", field)
    if not is_int(seed) or seed < 0 or seed > 0xFFFFFFFFFFFFFFFF:
        raise _spec_error("must be an unsigned 64-bit integer", f"{field}.seed")
    quant_profile = _require_nonempty_string(obj, "quant_profile", field)
    if quant_profile not in QUANT_PROFILES:
        raise _spec_error(
            "unsupported quantization profile; expected one of " + ", ".join(sorted(QUANT_PROFILES)),
            f"{field}.quant_profile",
        )
    return BackendSpec(backend_id, host, port, model_dim, hidden_dim, seed, quant_profile)


def validate_rule(
    raw: Any,
    index: int,
    seen_ids: set[str],
    backends: dict[str, BackendSpec],
) -> RuleSpec:
    field = f"rules[{index}]"
    obj = _require_object(raw, field)
    _reject_unknown_keys(obj, RULE_KEYS, field)
    rule_id = _require_nonempty_string(obj, "rule_id", field)
    if rule_id in seen_ids:
        raise _spec_error("duplicate rule_id", f"{field}.rule_id")
    seen_ids.add(rule_id)
    prefix = _require_nonempty_string(obj, "request_prefix", field)
    if not prefix.startswith("/"):
        raise _spec_error("must start with '/'", f"{field}.request_prefix")
    stable_backend = _require_nonempty_string(obj, "stable_backend", field)
    if stable_backend not in backends:
        raise _spec_error("unknown stable_backend", f"{field}.stable_backend")
    canary_backend = _require_nonempty_string(obj, "canary_backend", field)
    if canary_backend not in backends:
        raise _spec_error("unknown canary_backend", f"{field}.canary_backend")
    if stable_backend == canary_backend:
        raise _spec_error("stable and canary backend must differ", f"{field}.canary_backend")
    weight = _require_key(obj, "canary_weight_bp", field)
    if not is_int(weight) or weight < 0 or weight > 10000:
        raise _spec_error("must be an integer in 0..10000", f"{field}.canary_weight_bp")
    batch_size = _require_positive_int(obj, "batch_size", field, maximum=4096)
    return RuleSpec(rule_id, prefix, stable_backend, canary_backend, weight, batch_size)


def validate_sample(
    raw: Any,
    index: int,
    seen_ids: set[str],
    seen_seqs: set[int],
    rules: dict[str, RuleSpec],
) -> SampleSpec:
    field = f"samples[{index}]"
    obj = _require_object(raw, field)
    _reject_unknown_keys(obj, SAMPLE_KEYS, field)
    sample_id = _require_nonempty_string(obj, "sample_id", field)
    if sample_id in seen_ids:
        raise _spec_error("duplicate sample_id", f"{field}.sample_id")
    seen_ids.add(sample_id)
    sample_seq = _require_positive_int(obj, "sample_seq", field)
    if sample_seq in seen_seqs:
        raise _spec_error("duplicate sample_seq", f"{field}.sample_seq")
    seen_seqs.add(sample_seq)
    client_key = _require_nonempty_string(obj, "client_key", field)
    if len(client_key) > 256:
        raise _spec_error("must be at most 256 characters", f"{field}.client_key")
    rule_id = _require_nonempty_string(obj, "rule_id", field)
    if rule_id not in rules:
        raise _spec_error("unknown rule_id", f"{field}.rule_id")
    request_path = _require_nonempty_string(obj, "request_path", field)
    if not request_path.startswith(rules[rule_id].request_prefix):
        raise _spec_error(
            f"must start with rule prefix {rules[rule_id].request_prefix!r}",
            f"{field}.request_path",
        )
    input_seed = _require_key(obj, "input_seed", field)
    if not is_int(input_seed) or input_seed < 0 or input_seed > 0xFFFFFFFFFFFFFFFF:
        raise _spec_error("must be an unsigned 64-bit integer", f"{field}.input_seed")
    return SampleSpec(sample_id, sample_seq, client_key, rule_id, request_path, input_seed)


def validate_backends(raw_backends: Any) -> tuple[BackendSpec, ...]:
    items = _require_list(raw_backends, "backends")
    if not items:
        raise _spec_error("must contain at least one backend", "backends")
    seen_ids: set[str] = set()
    seen_ports: set[int] = set()
    return tuple(validate_backend(item, i, seen_ids, seen_ports) for i, item in enumerate(items))


def validate_rules(raw_rules: Any, backends: Iterable[BackendSpec]) -> tuple[RuleSpec, ...]:
    items = _require_list(raw_rules, "rules")
    if not items:
        raise _spec_error("must contain at least one rule", "rules")
    backend_map = {item.backend_id: item for item in backends}
    seen_ids: set[str] = set()
    return tuple(validate_rule(item, i, seen_ids, backend_map) for i, item in enumerate(items))


def validate_samples(raw_samples: Iterable[Any], rules: Iterable[RuleSpec]) -> tuple[SampleSpec, ...]:
    rule_map = {item.rule_id: item for item in rules}
    seen_ids: set[str] = set()
    seen_seqs: set[int] = set()
    result: list[SampleSpec] = []
    for index, raw in enumerate(raw_samples):
        result.append(validate_sample(raw, index, seen_ids, seen_seqs, rule_map))
    if not result:
        raise _spec_error("must contain at least one sample", "samples")
    return tuple(result)


def _read_json_file(path: str | Path, field: str) -> Any:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except OSError as exc:
        raise _spec_error(f"cannot read JSON file: {exc}", field) from exc
    except json.JSONDecodeError as exc:
        raise _spec_error(
            f"invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}", field
        ) from exc


def _read_ndjson_samples(path: str | Path) -> list[Any]:
    result: list[Any] = []
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    result.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise _spec_error(
                        f"invalid NDJSON at line {line_number}, column {exc.colno}: {exc.msg}",
                        f"samples[{len(result)}]",
                    ) from exc
    except OSError as exc:
        raise _spec_error(f"cannot read sample NDJSON: {exc}", "samples") from exc
    return result


def load_spec(
    backends_path: str | Path,
    rules_path: str | Path,
    samples_path: str | Path,
) -> LoadedSpec:
    raw_backends = _read_json_file(backends_path, "backends")
    raw_rules = _read_json_file(rules_path, "rules")
    if isinstance(raw_backends, dict):
        _reject_unknown_keys(raw_backends, {"format_version", "backends"}, "backends_file")
        raw_backends = _require_key(raw_backends, "backends", "backends_file")
    if isinstance(raw_rules, dict):
        _reject_unknown_keys(raw_rules, {"format_version", "rules"}, "rules_file")
        raw_rules = _require_key(raw_rules, "rules", "rules_file")
    backends = validate_backends(raw_backends)
    rules = validate_rules(raw_rules, backends)
    raw_samples = _read_ndjson_samples(samples_path)
    samples = validate_samples(raw_samples, rules)
    return LoadedSpec(backends, rules, samples)


def load_combined_spec(spec_path: str | Path, samples_path: str | Path) -> LoadedSpec:
    raw = _read_json_file(spec_path, "$spec")
    obj = _require_object(raw, "$spec")
    _reject_unknown_keys(obj, {"format_version", "backends", "rules"}, "$spec")
    backends = validate_backends(_require_key(obj, "backends", "$spec"))
    rules = validate_rules(_require_key(obj, "rules", "$spec"), backends)
    raw_samples = _read_ndjson_samples(samples_path)
    samples = validate_samples(raw_samples, rules)
    return LoadedSpec(backends, rules, samples)


def spec_to_json(spec: LoadedSpec) -> dict[str, Any]:
    return {
        "backends": [item.declaration() for item in spec.backends],
        "rules": [item.declaration() for item in spec.rules],
        "samples": [item.declaration() for item in spec.samples],
        "versions": {
            "backends": spec.backend_version(),
            "rules": spec.rules_version(),
            "samples": spec.samples_version(),
        },
    }


def sample_canonical_line(sample: SampleSpec) -> str:
    return canonical_json(sample.declaration())

