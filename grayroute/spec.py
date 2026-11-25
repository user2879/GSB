"""Loading and strong validation of local JSON declarations and NDJSON streams.

Files consumed:
  * ``backends.json`` - deterministic inference backends
  * ``rules.json``    - routing rules with monotonically increasing revisions
  * ``merge_rules.json`` - cross-version receipt merge policy
  * ``requests.ndjson`` - ordered request stream

Any validation failure raises :class:`SpecError` (``E_SPEC``) pointing at the
first illegal object id and field.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from .canonical import canonical_bytes, content_hash
from .constants import (
    BUCKETS,
    LOOPBACK_HOST,
    PORT_MAX,
    PORT_MIN,
    SUPPORTED_PARAM_FORMATS,
    SUPPORTED_QUANTIZATIONS,
    WEIGHT_MAX,
    WEIGHT_MIN,
)
from .errors import SpecError


# --------------------------------------------------------------------------- #
# data models
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Backend:
    backend_id: str
    host: str
    port: int
    model_dim: int
    hidden_dim: int
    quantization: str
    param_format: str
    param_summary: str
    role: str  # "stable" | "gray"

    def key(self) -> tuple[str, str]:
        return (self.role, self.backend_id)


@dataclass(frozen=True)
class RuleRevision:
    rule_id: str
    revision: int
    weight_permyriad: int
    batch_size: int
    param_version_seq: int
    param_summary: str
    backend_ids: tuple[str, ...]


@dataclass(frozen=True)
class Rule:
    rule_id: str
    revisions: tuple[RuleRevision, ...]

    def revision_at(self, revision: int) -> RuleRevision:
        for rev in self.revisions:
            if rev.revision == revision:
                return rev
        raise SpecError(
            f"rule {self.rule_id} has no revision {revision}",
            object_id=self.rule_id, field="revision",
        )


@dataclass(frozen=True)
class Request:
    request_id: str
    arrival_seq: int
    client_key: str
    rule_id: str
    rule_revision: int
    vector: tuple[float, ...]


@dataclass
class MergeRule:
    # ordering keys, each "desc" or "asc"; fixed by policy but declared in file
    order: tuple[str, ...] = ("param_version_seq", "format_version", "merge_seq")
    direction: tuple[str, ...] = ("desc", "desc", "asc")


@dataclass
class Spec:
    backends: dict[str, Backend]
    rules: dict[str, Rule]
    merge_rule: MergeRule
    requests: tuple[Request, ...]
    model_dim: int

    def revision_for(self, request: Request) -> RuleRevision:
        return self.rules[request.rule_id].revision_at(request.rule_revision)

    def spec_summary(self) -> str:
        payload = {
            "backends": [b.__dict__ for b in sorted(self.backends.values(),
                                                    key=lambda b: b.backend_id)],
            "rules": [
                {"rule_id": r.rule_id,
                 "revisions": [rv.__dict__ for rv in r.revisions]}
                for r in sorted(self.rules.values(), key=lambda r: r.rule_id)
            ],
            "requests": [q.request_id for q in self.requests],
        }
        return content_hash(payload)


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _require(obj: dict, name: str, object_id: str, types: tuple):
    if name not in obj:
        raise SpecError(f"missing field '{name}'", object_id=object_id, field=name)
    if not isinstance(obj[name], types) or isinstance(obj[name], bool):
        raise SpecError(
            f"field '{name}' expects {types}, got {type(obj[name]).__name__}",
            object_id=object_id, field=name,
        )
    return obj[name]


def _positive_int(value: Any, object_id: str, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise SpecError(
            f"field '{field_name}' must be a positive integer",
            object_id=object_id, field=field_name,
        )
    return value


def _read_json(path: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        raise SpecError(f"file not found: {path}", object_id=path, field="file")
    except json.JSONDecodeError as exc:
        raise SpecError(
            f"invalid JSON: {exc.msg} at line {exc.lineno}",
            object_id=path, field="file",
        )


def _read_ndjson(path: str) -> list[Any]:
    try:
        handle = open(path, "r", encoding="utf-8")
    except FileNotFoundError:
        raise SpecError(f"file not found: {path}", object_id=path, field="file")
    rows: list[Any] = []
    with handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SpecError(
                    f"invalid NDJSON line {lineno}: {exc.msg}",
                    object_id=f"{path}:{lineno}", field="line",
                )
    return rows


# --------------------------------------------------------------------------- #
# backends
# --------------------------------------------------------------------------- #
def load_backends(path: str) -> dict[str, Backend]:
    raw = _read_json(path)
    object_id = os.path.basename(path)
    if not isinstance(raw, list):
        raise SpecError("backends.json must be a JSON array",
                        object_id=object_id, field="root")
    backends: dict[str, Backend] = {}
    seen_ports: set[int] = set()
    model_dims: set[int] = set()
    for item in raw:
        bid = _require(item, "backend_id", object_id, (str,))
        if bid in backends:
            raise SpecError(f"duplicate backend_id '{bid}'",
                            object_id=bid, field="backend_id")
        host = _require(item, "host", bid, (str,))
        if host != LOOPBACK_HOST:
            raise SpecError(
                f"host must be the literal {LOOPBACK_HOST}",
                object_id=bid, field="host",
            )
        port = _require(item, "port", bid, (int,))
        if isinstance(port, bool) or not (PORT_MIN <= port <= PORT_MAX):
            raise SpecError(
                f"port must be an integer in [{PORT_MIN},{PORT_MAX}]",
                object_id=bid, field="port",
            )
        if port in seen_ports:
            raise SpecError(f"port {port} is used by more than one backend",
                            object_id=bid, field="port")
        seen_ports.add(port)
        model_dim = _positive_int(
            _require(item, "model_dim", bid, (int,)), bid, "model_dim")
        hidden_dim = _positive_int(
            _require(item, "hidden_dim", bid, (int,)), bid, "hidden_dim")
        quant = _require(item, "quantization", bid, (str,))
        if quant not in SUPPORTED_QUANTIZATIONS:
            raise SpecError(
                f"unsupported quantization '{quant}'",
                object_id=bid, field="quantization",
            )
        param_format = _require(item, "param_format", bid, (str,))
        if param_format not in SUPPORTED_PARAM_FORMATS:
            raise SpecError(
                f"unsupported param_format '{param_format}'",
                object_id=bid, field="param_format",
            )
        summary = _require(item, "param_summary", bid, (str,))
        role = _require(item, "role", bid, (str,))
        if role not in ("stable", "gray"):
            raise SpecError("role must be 'stable' or 'gray'",
                            object_id=bid, field="role")
        model_dims.add(model_dim)
        backends[bid] = Backend(
            backend_id=bid, host=host, port=port, model_dim=model_dim,
            hidden_dim=hidden_dim, quantization=quant,
            param_format=param_format, param_summary=summary, role=role,
        )
    if not backends:
        raise SpecError("at least one backend required",
                        object_id=object_id, field="root")
    if len(model_dims) != 1:
        raise SpecError("all backends must share one model_dim",
                        object_id=object_id, field="model_dim")
    return backends


# --------------------------------------------------------------------------- #
# routing rules
# --------------------------------------------------------------------------- #
def load_rules(path: str, backends: dict[str, Backend]) -> dict[str, Rule]:
    raw = _read_json(path)
    object_id = os.path.basename(path)
    if not isinstance(raw, list):
        raise SpecError("rules.json must be a JSON array",
                        object_id=object_id, field="root")
    rules: dict[str, Rule] = {}
    for item in raw:
        rid = _require(item, "rule_id", object_id, (str,))
        if rid in rules:
            raise SpecError(f"duplicate rule_id '{rid}'",
                            object_id=rid, field="rule_id")
        revisions_raw = _require(item, "revisions", rid, (list,))
        if not revisions_raw:
            raise SpecError("rule needs at least one revision",
                            object_id=rid, field="revisions")
        revisions: list[RuleRevision] = []
        prev_revision = None
        for rev_item in revisions_raw:
            revision = _positive_int(
                _require(rev_item, "revision", rid, (int,)), rid, "revision")
            if prev_revision is not None and revision <= prev_revision:
                raise SpecError(
                    "rule revision numbers must be strictly increasing",
                    object_id=rid, field="revision",
                )
            prev_revision = revision
            weight = _require(rev_item, "weight_permyriad", rid, (int,))
            if isinstance(weight, bool) or not (WEIGHT_MIN <= weight <= WEIGHT_MAX):
                raise SpecError(
                    f"weight_permyriad must be in [{WEIGHT_MIN},{WEIGHT_MAX}]",
                    object_id=rid, field="weight_permyriad",
                )
            batch_size = _positive_int(
                _require(rev_item, "batch_size", rid, (int,)), rid, "batch_size")
            pseq = _positive_int(
                _require(rev_item, "param_version_seq", rid, (int,)),
                rid, "param_version_seq")
            summary = _require(rev_item, "param_summary", rid, (str,))
            backend_ids = _require(rev_item, "backend_ids", rid, (list,))
            if not backend_ids:
                raise SpecError("revision must reference backends",
                                object_id=rid, field="backend_ids")
            resolved: list[str] = []
            for ref in backend_ids:
                if not isinstance(ref, str) or ref not in backends:
                    raise SpecError(
                        f"unknown backend reference '{ref}'",
                        object_id=rid, field="backend_ids",
                    )
                resolved.append(ref)
            roles = {backends[b].role for b in resolved}
            if roles != {"stable", "gray"}:
                raise SpecError(
                    "revision backends must cover both stable and gray roles",
                    object_id=rid, field="backend_ids",
                )
            revisions.append(RuleRevision(
                rule_id=rid, revision=revision, weight_permyriad=weight,
                batch_size=batch_size, param_version_seq=pseq,
                param_summary=summary, backend_ids=tuple(resolved),
            ))
        rules[rid] = Rule(rule_id=rid, revisions=tuple(revisions))
    if not rules:
        raise SpecError("at least one rule required",
                        object_id=object_id, field="root")
    return rules


def load_merge_rule(path: str) -> MergeRule:
    raw = _read_json(path)
    object_id = os.path.basename(path)
    allowed = {"param_version_seq", "format_version", "merge_seq"}
    order = tuple(raw.get("order", MergeRule.order))
    direction = tuple(raw.get("direction", MergeRule.direction))
    if set(order) != allowed or len(order) != 3:
        raise SpecError("merge order must enumerate the three keys",
                        object_id=object_id, field="order")
    if any(d not in ("asc", "desc") for d in direction):
        raise SpecError("merge direction must be asc or desc",
                        object_id=object_id, field="direction")
    return MergeRule(order=order, direction=direction)


# --------------------------------------------------------------------------- #
# request stream
# --------------------------------------------------------------------------- #
def load_requests(path: str, rules: dict[str, Rule],
                  model_dim: int) -> tuple[Request, ...]:
    rows = _read_ndjson(path)
    object_id = os.path.basename(path)
    requests: list[Request] = []
    seen_ids: set[str] = set()
    seen_seqs: set[int] = set()
    for row in rows:
        rid = _require(row, "request_id", object_id, (str,))
        if rid in seen_ids:
            raise SpecError(f"duplicate request_id '{rid}'",
                            object_id=rid, field="request_id")
        seen_ids.add(rid)
        seq = _require(row, "arrival_seq", rid, (int,))
        if isinstance(seq, bool) or seq < 0:
            raise SpecError("arrival_seq must be a non-negative integer",
                            object_id=rid, field="arrival_seq")
        if seq in seen_seqs:
            raise SpecError(f"duplicate arrival_seq {seq}",
                            object_id=rid, field="arrival_seq")
        seen_seqs.add(seq)
        client_key = _require(row, "client_key", rid, (str,))
        rule_id = _require(row, "rule_id", rid, (str,))
        if rule_id not in rules:
            raise SpecError(f"unknown rule_id '{rule_id}'",
                            object_id=rid, field="rule_id")
        revision = _positive_int(
            _require(row, "rule_revision", rid, (int,)), rid, "rule_revision")
        if revision not in {rv.revision for rv in rules[rule_id].revisions}:
            raise SpecError(
                f"rule {rule_id} has no revision {revision}",
                object_id=rid, field="rule_revision",
            )
        vector = _require(row, "vector", rid, (list,))
        if len(vector) != model_dim:
            raise SpecError(
                f"vector length {len(vector)} != model_dim {model_dim}",
                object_id=rid, field="vector",
            )
        for value in vector:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise SpecError("vector entries must be numbers",
                                object_id=rid, field="vector")
        requests.append(Request(
            request_id=rid, arrival_seq=seq, client_key=client_key,
            rule_id=rule_id, rule_revision=revision,
            vector=tuple(float(v) for v in vector),
        ))
    if not requests:
        raise SpecError("request stream is empty",
                        object_id=object_id, field="root")
    return tuple(requests)


def load_spec(config_dir: str, request_path: str | None = None) -> Spec:
    backends = load_backends(os.path.join(config_dir, "backends.json"))
    rules = load_rules(os.path.join(config_dir, "rules.json"), backends)
    merge_rule = load_merge_rule(os.path.join(config_dir, "merge_rules.json"))
    model_dim = next(iter(backends.values())).model_dim
    if request_path is None:
        request_path = os.path.join(os.path.dirname(config_dir.rstrip("/")),
                                    "data", "requests.ndjson")
    requests = load_requests(request_path, rules, model_dim)
    return Spec(backends=backends, rules=rules, merge_rule=merge_rule,
                requests=requests, model_dim=model_dim)


def spec_canonical(spec: Spec) -> bytes:
    """Stable canonical bytes covering the full validated specification."""
    return canonical_bytes({
        "backends": [b.__dict__ for b in
                     sorted(spec.backends.values(), key=lambda b: b.backend_id)],
        "rules": [
            {"rule_id": r.rule_id, "revisions": [rv.__dict__ for rv in r.revisions]}
            for r in sorted(spec.rules.values(), key=lambda r: r.rule_id)
        ],
        "requests": [
            {"request_id": q.request_id, "arrival_seq": q.arrival_seq,
             "client_key": q.client_key, "rule_id": q.rule_id,
             "rule_revision": q.rule_revision, "vector": list(q.vector)}
            for q in sorted(spec.requests, key=lambda q: q.arrival_seq)
        ],
    })
