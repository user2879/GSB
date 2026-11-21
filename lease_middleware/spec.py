"""Loading and strong validation of config item / subscriber declarations."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import E_SPEC, MiddlewareError

SUPPORTED_FORMATS = ("r1", "r2", "r3")
CURRENT_FORMAT = "r3"


def _is_positive_int(obj: Any) -> bool:
    # bool is a subclass of int; reject it explicitly.
    return isinstance(obj, int) and not isinstance(obj, bool) and obj > 0


@dataclass(frozen=True)
class ConfigItem:
    key: str
    ttl_seconds: int
    values: tuple[dict[str, Any], ...]          # ordered {version, value}
    _digests: dict[int, str]

    @property
    def current_version(self) -> int:
        return self.values[-1]["version"]

    def value_at(self, version: int) -> Any:
        for entry in self.values:
            if entry["version"] == version:
                return entry["value"]
        raise KeyError(version)

    def digest_at(self, version: int) -> str:
        return self._digests[version]


@dataclass(frozen=True)
class Client:
    id: str
    subscribed_keys: frozenset[str]
    lease_duration_ms: int
    format_version: str


@dataclass(frozen=True)
class Spec:
    configs: dict[str, ConfigItem]
    clients: dict[str, Client]
    source_path: Path

    def require_config(self, key: str) -> ConfigItem:
        try:
            return self.configs[key]
        except KeyError:
            raise MiddlewareError(E_SPEC, f"unknown config key {key!r}", f"configs.{key}")


def canonical_json(value: Any) -> str:
    """Deterministic JSON encoding used for value digests everywhere."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def load_spec(path: str | Path) -> Spec:
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise MiddlewareError(E_SPEC, f"spec file not found: {path}", "file")
    except json.JSONDecodeError as exc:
        raise MiddlewareError(
            E_SPEC, f"spec JSON is not parseable: {exc.msg}", "file",
            context={"line": exc.lineno, "column": exc.colno},
        )

    errors: list[str] = []
    if not isinstance(raw, dict):
        raise MiddlewareError(E_SPEC, "spec root must be a JSON object", "$")

    configs = _parse_configs(raw.get("configs"), errors)
    clients = _parse_clients(raw.get("clients"), errors, configs)

    if errors:
        raise MiddlewareError(
            E_SPEC, "spec validation failed; whole load rejected", "$",
            context={"errors": errors},
        )
    return Spec(configs=configs, clients=clients, source_path=path)


def _parse_configs(raw: Any, errors: list[str]) -> dict[str, ConfigItem]:
    result: dict[str, ConfigItem] = {}
    if not isinstance(raw, list) or not raw:
        errors.append("configs: must be a non-empty array")
        return result

    seen: set[str] = set()
    for idx, item in enumerate(raw):
        base = f"configs[{idx}]"
        if not isinstance(item, dict):
            errors.append(f"{base}: must be an object")
            continue
        key = item.get("key")
        if not isinstance(key, str) or not key:
            errors.append(f"{base}.key: must be a non-empty string")
            continue
        if key in seen:
            errors.append(f"{base}.key: duplicate config key {key!r}")
            continue

        ttl = item.get("ttl_seconds")
        if not _is_positive_int(ttl):
            errors.append(f"{base}.ttl_seconds: must be a positive integer")

        values_raw = item.get("values")
        values, digests, ok = _parse_values(values_raw, f"{base}.values", errors)
        if not ok or not _is_positive_int(ttl):
            continue
        seen.add(key)
        result[key] = ConfigItem(key=key, ttl_seconds=ttl, values=tuple(values), _digests=digests)
    return result


def _parse_values(raw: Any, base: str, errors: list[str]):
    from hashlib import sha256

    if not isinstance(raw, list) or not raw:
        errors.append(f"{base}: must be a non-empty array")
        return [], {}, False
    parsed: list[dict[str, Any]] = []
    digests: dict[int, str] = {}
    versions: set[int] = set()
    prev = 0
    ok = True
    for idx, entry in enumerate(raw):
        p = f"{base}[{idx}]"
        if not isinstance(entry, dict) or "version" not in entry or "value" not in entry:
            errors.append(f"{p}: each value entry needs integer 'version' and 'value'")
            ok = False
            continue
        version = entry["version"]
        if not _is_positive_int(version):
            errors.append(f"{p}.version: must be a positive integer")
            ok = False
            continue
        if version in versions:
            errors.append(f"{p}.version: duplicate version {version}")
            ok = False
            continue
        if version <= prev:
            errors.append(f"{p}.version: versions must be strictly increasing")
            ok = False
            continue
        prev = version
        versions.add(version)
        parsed.append({"version": version, "value": entry["value"]})
        digests[version] = sha256(canonical_json(entry["value"]).encode("utf-8")).hexdigest()
    return parsed, digests, ok


def _parse_clients(raw: Any, errors: list[str],
                   configs: dict[str, ConfigItem]) -> dict[str, Client]:
    result: dict[str, Client] = {}
    if not isinstance(raw, list) or not raw:
        errors.append("clients: must be a non-empty array")
        return result

    seen: set[str] = set()
    for idx, item in enumerate(raw):
        base = f"clients[{idx}]"
        if not isinstance(item, dict):
            errors.append(f"{base}: must be an object")
            continue
        cid = item.get("id")
        if not isinstance(cid, str) or not cid:
            errors.append(f"{base}.id: must be a non-empty string")
            continue
        if cid in seen:
            errors.append(f"{base}.id: duplicate client id {cid!r}")
            continue

        keys = item.get("subscribed_keys")
        if not isinstance(keys, list) or not keys:
            errors.append(f"{base}.subscribed_keys: must be a non-empty array")
            continue
        if any(not isinstance(k, str) for k in keys) or len(set(keys)) != len(keys):
            errors.append(f"{base}.subscribed_keys: entries must be unique strings")
            continue
        missing = [k for k in keys if k not in configs]
        if missing:
            errors.append(f"{base}.subscribed_keys: unknown key(s) {missing}")
            continue

        duration = item.get("lease_duration_ms")
        if not _is_positive_int(duration):
            errors.append(f"{base}.lease_duration_ms: must be a positive integer")

        fmt = item.get("format_version")
        if fmt not in SUPPORTED_FORMATS:
            errors.append(f"{base}.format_version: must be one of {list(SUPPORTED_FORMATS)}")

        if _is_positive_int(duration) and fmt in SUPPORTED_FORMATS:
            seen.add(cid)
            result[cid] = Client(
                id=cid,
                subscribed_keys=frozenset(keys),
                lease_duration_ms=duration,
                format_version=fmt,
            )
    return result
