"""Strict, all-or-nothing JSON configuration loading."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .errors import ErrorCode, TunnelError

SUPPORTED_SCHEMA_VERSION = 1
SUPPORTED_CIPHERS = frozenset(
    {"NONE", "AES-128-GCM", "AES-256-GCM", "CHACHA20-POLY1305"}
)
PEER_BEHAVIORS = frozenset({"normal", "delay-script", "unreachable", "digest-error"})
MIN_MTU = 128
MAX_MTU = 9000
MIN_HIGH_PORT = 1024
MAX_PORT = 65535
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,31}$")


@dataclass(frozen=True)
class Endpoint:
    host: str
    port: int

    def to_dict(self) -> dict[str, Any]:
        return {"host": self.host, "port": self.port}


@dataclass(frozen=True)
class PeerDefinition:
    id: str
    endpoint: Endpoint
    behavior: str
    delay_ms: int = 0
    script: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "id": self.id,
            "host": self.endpoint.host,
            "port": self.endpoint.port,
            "behavior": self.behavior,
            "delay_ms": self.delay_ms,
        }
        if self.script:
            result["script"] = self.script
        return result


@dataclass(frozen=True)
class GlobalLimits:
    bandwidth_kbps: int
    max_sessions: int
    fragment_timeout_ms: int
    peer_timeout_ms: int
    connect_timeout_ms: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Route:
    id: str
    listen: Endpoint
    target: Endpoint
    peer_id: str
    mtu: int
    cipher: str
    bandwidth_kbps: int
    max_sessions: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "listen": self.listen.to_dict(),
            "target": self.target.to_dict(),
            "peer_id": self.peer_id,
            "mtu": self.mtu,
            "cipher": self.cipher,
            "bandwidth_kbps": self.bandwidth_kbps,
            "max_sessions": self.max_sessions,
        }


@dataclass(frozen=True)
class TunnelConfig:
    schema_version: int
    route_version: str
    global_limits: GlobalLimits
    peers: tuple[PeerDefinition, ...]
    routes: tuple[Route, ...]

    def route_map(self) -> dict[str, Route]:
        return {route.id: route for route in self.routes}

    def peer_map(self) -> dict[str, PeerDefinition]:
        return {peer.id: peer for peer in self.peers}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "route_version": self.route_version,
            "global": self.global_limits.to_dict(),
            "peers": [peer.to_dict() for peer in self.peers],
            "routes": [route.to_dict() for route in self.routes],
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _fail(field: str, message: str, **details: Any) -> None:
    raise TunnelError(ErrorCode.E_CONFIG, message, field=field, details=details)


def _object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(field, "必须是 JSON 对象")
    return value


def _string(value: Any, field: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value.strip()):
        _fail(field, "必须是非空字符串")
    return value


def _identifier(value: Any, field: str) -> str:
    text = _string(value, field)
    if not _IDENTIFIER_RE.fullmatch(text):
        _fail(field, "必须匹配 [A-Za-z0-9][A-Za-z0-9._-]{0,31}")
    return text


def _positive_int(value: Any, field: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(field, "必须是整数（布尔值不接受）")
    if value < minimum:
        _fail(field, f"必须是 >= {minimum} 的整数")
    return value


def _port(value: Any, field: str) -> int:
    port = _positive_int(value, field)
    if not (MIN_HIGH_PORT <= port <= MAX_PORT):
        _fail(field, f"必须位于高位端口区间 {MIN_HIGH_PORT}..{MAX_PORT}")
    return port


def _endpoint(value: Any, field: str) -> Endpoint:
    raw = _object(value, field)
    host = _string(raw.get("host"), f"{field}.host")
    if host != "127.0.0.1":
        _fail(f"{field}.host", "只允许 IPv4 回环地址 127.0.0.1")
    port = _port(raw.get("port"), f"{field}.port")
    return Endpoint(host=host, port=port)


def _required(raw: dict[str, Any], name: str, field: str) -> Any:
    if name not in raw:
        _fail(f"{field}.{name}" if field else name, "缺少必填字段")
    return raw[name]


def parse_config(raw: Any) -> TunnelConfig:
    """Validate the complete document before returning any route."""

    root = _object(raw, "$")
    schema_version = _positive_int(
        _required(root, "schema_version", "$"), "schema_version"
    )
    if schema_version != SUPPORTED_SCHEMA_VERSION:
        _fail(
            "schema_version",
            f"不支持的 schema_version，期望 {SUPPORTED_SCHEMA_VERSION}",
        )
    route_version = _string(_required(root, "route_version", "$"), "route_version")

    global_raw = _object(_required(root, "global", "$"), "global")
    global_limits = GlobalLimits(
        bandwidth_kbps=_positive_int(
            _required(global_raw, "bandwidth_kbps", "global"),
            "global.bandwidth_kbps",
        ),
        max_sessions=_positive_int(
            _required(global_raw, "max_sessions", "global"), "global.max_sessions"
        ),
        fragment_timeout_ms=_positive_int(
            _required(global_raw, "fragment_timeout_ms", "global"),
            "global.fragment_timeout_ms",
        ),
        peer_timeout_ms=_positive_int(
            _required(global_raw, "peer_timeout_ms", "global"), "global.peer_timeout_ms"
        ),
        connect_timeout_ms=_positive_int(
            _required(global_raw, "connect_timeout_ms", "global"),
            "global.connect_timeout_ms",
        ),
    )

    peers_raw = _required(root, "peers", "$")
    if not isinstance(peers_raw, list) or not peers_raw:
        _fail("peers", "必须是非空数组")
    peers: list[PeerDefinition] = []
    peer_ids: set[str] = set()
    peer_ports: dict[int, str] = {}
    for index, item in enumerate(peers_raw):
        field = f"peers[{index}]"
        obj = _object(item, field)
        peer_id = _identifier(_required(obj, "id", field), f"{field}.id")
        if peer_id in peer_ids:
            _fail(f"{field}.id", "对端标识重复", value=peer_id)
        peer_ids.add(peer_id)
        endpoint = _endpoint(
            {
                "host": _required(obj, "host", field),
                "port": _required(obj, "port", field),
            },
            field,
        )
        if endpoint.port in peer_ports:
            _fail(
                f"{field}.port",
                "端口与已声明对端端点冲突",
                other=peer_ports[endpoint.port],
            )
        peer_ports[endpoint.port] = f"{field}.port"
        behavior = _string(_required(obj, "behavior", field), f"{field}.behavior")
        if behavior not in PEER_BEHAVIORS:
            _fail(
                f"{field}.behavior",
                "不支持的对端行为",
                supported=sorted(PEER_BEHAVIORS),
            )
        delay_ms = _positive_int(obj.get("delay_ms", 0), f"{field}.delay_ms", minimum=0)
        script_value = obj.get("script")
        script = None
        if script_value is not None:
            script = _string(script_value, f"{field}.script")
        peers.append(
            PeerDefinition(
                id=peer_id,
                endpoint=endpoint,
                behavior=behavior,
                delay_ms=delay_ms,
                script=script,
            )
        )

    routes_raw = _required(root, "routes", "$")
    if not isinstance(routes_raw, list) or not routes_raw:
        _fail("routes", "必须是非空数组")
    routes: list[Route] = []
    route_ids: set[str] = set()
    peer_by_id = {peer.id: peer for peer in peers}
    used_route_ports: dict[int, str] = {}
    for index, item in enumerate(routes_raw):
        field = f"routes[{index}]"
        obj = _object(item, field)
        route_id = _identifier(_required(obj, "id", field), f"{field}.id")
        if route_id in route_ids:
            _fail(f"{field}.id", "隧道标识重复", value=route_id)
        route_ids.add(route_id)
        listen = _endpoint(_required(obj, "listen", field), f"{field}.listen")
        target = _endpoint(_required(obj, "target", field), f"{field}.target")
        peer_id = _identifier(_required(obj, "peer_id", field), f"{field}.peer_id")
        peer = peer_by_id.get(peer_id)
        if peer is None:
            _fail(f"{field}.peer_id", "引用了未定义的对端", value=peer_id)
        if target != peer.endpoint:
            _fail(
                f"{field}.target",
                "路由目标端点必须与对端定义一致",
                peer_id=peer_id,
            )
        for endpoint_field, endpoint in (
            (f"{field}.listen", listen),
            (f"{field}.target", target),
        ):
            if endpoint.port in used_route_ports:
                _fail(
                    f"{endpoint_field}.port",
                    "端口与已声明路由端点冲突",
                    other=used_route_ports[endpoint.port],
                )
            used_route_ports[endpoint.port] = f"{endpoint_field}.port"
        if listen.port in peer_ports:
            _fail(
                f"{field}.listen.port",
                "入站监听端口与对端端点冲突",
                other=peer_ports[listen.port],
            )
        mtu = _positive_int(_required(obj, "mtu", field), f"{field}.mtu")
        if not (MIN_MTU <= mtu <= MAX_MTU):
            _fail(f"{field}.mtu", f"MTU 必须位于 {MIN_MTU}..{MAX_MTU}")
        cipher = _string(_required(obj, "cipher", field), f"{field}.cipher")
        if cipher not in SUPPORTED_CIPHERS:
            _fail(
                f"{field}.cipher",
                "不支持的加密套件标识",
                supported=sorted(SUPPORTED_CIPHERS),
            )
        bandwidth_kbps = _positive_int(
            _required(obj, "bandwidth_kbps", field), f"{field}.bandwidth_kbps"
        )
        max_sessions = _positive_int(
            _required(obj, "max_sessions", field), f"{field}.max_sessions"
        )
        routes.append(
            Route(
                id=route_id,
                listen=listen,
                target=target,
                peer_id=peer_id,
                mtu=mtu,
                cipher=cipher,
                bandwidth_kbps=bandwidth_kbps,
                max_sessions=max_sessions,
            )
        )

    return TunnelConfig(
        schema_version=schema_version,
        route_version=route_version,
        global_limits=global_limits,
        peers=tuple(peers),
        routes=tuple(routes),
    )


def load_config(path: str | Path) -> TunnelConfig:
    file_path = Path(path)
    try:
        with file_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except OSError as exc:
        raise TunnelError(
            ErrorCode.E_CONFIG,
            f"无法读取配置文件: {exc}",
            field="$.file",
        ) from exc
    except json.JSONDecodeError as exc:
        raise TunnelError(
            ErrorCode.E_CONFIG,
            f"JSON 解析失败: {exc.msg}",
            field=f"$.line[{exc.lineno}].column[{exc.colno}]",
            details={"line": exc.lineno, "column": exc.colno},
        ) from exc

    config = parse_config(raw)
    # Ensure the normalized document can be snapshotted immediately.
    json.dumps(config.to_dict(), ensure_ascii=False)
    return config
