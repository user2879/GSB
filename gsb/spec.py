"""后端声明、灰度路由规则与 NDJSON 请求流的加载与强校验。

任一不合法以 E_SPEC 拒绝，并指明首个违规字段与所在后端或规则标识。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .errors import GSBError, e_spec

LOOPBACK_LITERAL = "127.0.0.1"
SUPPORTED_QUANT_PROFILES = ("int8", "int4", "fp16")
SUPPORTED_FORMAT_VERSIONS = ("v1", "v2", "v3")
MAX_WEIGHT = 10000


@dataclass(frozen=True)
class BackendDecl:
    backend_id: str
    host: str
    port: int
    model_dim: int
    hidden_dim: int
    seed: int
    quant_profile: str
    format_version: str


@dataclass(frozen=True)
class RouteRule:
    rule_id: str
    stable_backend: str
    gray_backend: str
    weight: int
    rule_version: int


@dataclass(frozen=True)
class Request:
    request_id: str
    batch_id: str
    client_key: str
    rule_id: str


@dataclass
class Batch:
    batch_id: str
    requests: list[Request] = field(default_factory=list)


@dataclass
class Spec:
    backends: dict[str, BackendDecl]
    rules: dict[str, RouteRule]
    requests: list[Request]
    batches: list[Batch]


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_str(obj: dict, key: str, owner: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value:
        raise e_spec(f"字段 {key} 必须为非空字符串", field=key, owner=owner)
    return value


def validate_backend(obj: dict, index: int) -> BackendDecl:
    raw_id = obj.get("backend_id")
    owner = raw_id if isinstance(raw_id, str) and raw_id else f"<backends[{index}]>"
    backend_id = _require_str(obj, "backend_id", owner)
    host = _require_str(obj, "host", backend_id)
    if host != LOOPBACK_LITERAL:
        raise e_spec(
            f"host 必须为回环字面量 {LOOPBACK_LITERAL}", field="host", owner=backend_id
        )
    port = obj.get("port")
    if not _is_int(port) or not (1 <= port <= 65535):
        raise e_spec("port 必须为 1..65535 的整数", field="port", owner=backend_id)
    model_dim = obj.get("model_dim")
    if not _is_int(model_dim) or model_dim <= 0:
        raise e_spec("model_dim 必须为正整数", field="model_dim", owner=backend_id)
    hidden_dim = obj.get("hidden_dim")
    if not _is_int(hidden_dim) or hidden_dim <= 0:
        raise e_spec("hidden_dim 必须为正整数", field="hidden_dim", owner=backend_id)
    seed = obj.get("seed")
    if not _is_int(seed):
        raise e_spec("seed 必须为整数", field="seed", owner=backend_id)
    quant = obj.get("quant_profile")
    if quant not in SUPPORTED_QUANT_PROFILES:
        raise e_spec(
            f"quant_profile 必须属于 {SUPPORTED_QUANT_PROFILES}",
            field="quant_profile",
            owner=backend_id,
        )
    fmt = obj.get("format_version")
    if fmt not in SUPPORTED_FORMAT_VERSIONS:
        raise e_spec(
            f"format_version 必须属于 {SUPPORTED_FORMAT_VERSIONS}",
            field="format_version",
            owner=backend_id,
        )
    return BackendDecl(
        backend_id=backend_id,
        host=host,
        port=port,
        model_dim=model_dim,
        hidden_dim=hidden_dim,
        seed=seed,
        quant_profile=quant,
        format_version=fmt,
    )


def validate_rule(obj: dict, index: int, backend_ids: set[str]) -> RouteRule:
    raw_id = obj.get("rule_id")
    owner = raw_id if isinstance(raw_id, str) and raw_id else f"<rules[{index}]>"
    rule_id = _require_str(obj, "rule_id", owner)
    stable = _require_str(obj, "stable_backend", rule_id)
    if stable not in backend_ids:
        raise e_spec(
            "stable_backend 未引用已声明后端", field="stable_backend", owner=rule_id
        )
    gray = _require_str(obj, "gray_backend", rule_id)
    if gray not in backend_ids:
        raise e_spec(
            "gray_backend 未引用已声明后端", field="gray_backend", owner=rule_id
        )
    weight = obj.get("weight")
    if not _is_int(weight) or not (0 <= weight <= MAX_WEIGHT):
        raise e_spec(
            f"weight 必须为 0..{MAX_WEIGHT} 的整数", field="weight", owner=rule_id
        )
    rule_version = obj.get("rule_version")
    if not _is_int(rule_version) or rule_version <= 0:
        raise e_spec(
            "rule_version 必须为正整数", field="rule_version", owner=rule_id
        )
    return RouteRule(
        rule_id=rule_id,
        stable_backend=stable,
        gray_backend=gray,
        weight=weight,
        rule_version=rule_version,
    )


def load_backends(path: Path) -> dict[str, BackendDecl]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    items = raw.get("backends") if isinstance(raw, dict) else None
    if not isinstance(items, list):
        raise e_spec("backends 必须为数组", field="backends", owner="<backends>")
    decls = [validate_backend(obj, i) for i, obj in enumerate(items)]
    result: dict[str, BackendDecl] = {}
    seen_ports: dict[int, str] = {}
    for decl in decls:
        if decl.backend_id in result:
            raise e_spec(
                "backend_id 重复", field="backend_id", owner=decl.backend_id
            )
        if decl.port in seen_ports:
            raise e_spec(
                f"port 与后端 {seen_ports[decl.port]} 重复",
                field="port",
                owner=decl.backend_id,
            )
        result[decl.backend_id] = decl
        seen_ports[decl.port] = decl.backend_id
    return result


def load_rules(path: Path, backend_ids: set[str]) -> dict[str, RouteRule]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    items = raw.get("rules") if isinstance(raw, dict) else None
    if not isinstance(items, list):
        raise e_spec("rules 必须为数组", field="rules", owner="<rules>")
    rules = [validate_rule(obj, i, backend_ids) for i, obj in enumerate(items)]
    result: dict[str, RouteRule] = {}
    for rule in rules:
        if rule.rule_id in result:
            raise e_spec("rule_id 重复", field="rule_id", owner=rule.rule_id)
        result[rule.rule_id] = rule
    return result


def load_requests(path: Path, rule_ids: set[str]) -> tuple[list[Request], list[Batch]]:
    requests: list[Request] = []
    seen_ids: set[str] = set()
    for lineno, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = line.strip()
        if not line:
            continue
        owner = f"<requests:{lineno}>"
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise e_spec(f"NDJSON 第 {lineno} 行不是合法 JSON", field="json", owner=owner) from exc
        if not isinstance(obj, dict):
            raise e_spec("请求行必须为 JSON 对象", field="json", owner=owner)
        request_id = _require_str(obj, "request_id", owner)
        batch_id = _require_str(obj, "batch_id", request_id)
        client_key = _require_str(obj, "client_key", request_id)
        rule_id = _require_str(obj, "rule_id", request_id)
        if rule_id not in rule_ids:
            raise e_spec("rule_id 未引用已声明规则", field="rule_id", owner=request_id)
        if request_id in seen_ids:
            raise e_spec("request_id 重复", field="request_id", owner=request_id)
        seen_ids.add(request_id)
        requests.append(
            Request(
                request_id=request_id,
                batch_id=batch_id,
                client_key=client_key,
                rule_id=rule_id,
            )
        )
    batches: list[Batch] = []
    by_id: dict[str, Batch] = {}
    for req in requests:
        batch = by_id.get(req.batch_id)
        if batch is None:
            batch = Batch(batch_id=req.batch_id)
            by_id[req.batch_id] = batch
            batches.append(batch)
        batch.requests.append(req)
    return requests, batches


def load_spec(sample_dir) -> Spec:
    sample_dir = Path(sample_dir)
    backends = load_backends(sample_dir / "backends.json")
    rules = load_rules(sample_dir / "rules.json", set(backends))
    requests, batches = load_requests(sample_dir / "requests.ndjson", set(rules))
    return Spec(backends=backends, rules=rules, requests=requests, batches=batches)


def input_content_digest(sample_dir) -> str:
    """对规格输入文件计算稳定内容摘要（文件名排序拼接，不依赖字典序）。"""
    import hashlib

    sample_dir = Path(sample_dir)
    names = sorted(p.name for p in sample_dir.iterdir() if p.is_file())
    hasher = hashlib.sha256()
    for name in names:
        hasher.update(name.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(hashlib.sha256((sample_dir / name).read_bytes()).digest())
    return hasher.hexdigest()


__all__ = [
    "Spec",
    "BackendDecl",
    "RouteRule",
    "Request",
    "Batch",
    "load_spec",
    "input_content_digest",
    "GSBError",
]
