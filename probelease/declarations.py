"""Loading and strict validation of local deterministic declarations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from .errors import DependencyError, GrantError, SessionValidationError

SUPPORTED_FORMAT_VERSION = 1
RATIO_SCALE = 10_000


@dataclass(frozen=True)
class ProbeDefinition:
    probe_id: str
    seed: int
    behavior: str
    delay_ms: int
    sequence_offset: int = 0

    def canonical(self) -> dict[str, Any]:
        return {
            "behavior": self.behavior,
            "delay_ms": self.delay_ms,
            "probe_id": self.probe_id,
            "seed": self.seed,
            "sequence_offset": self.sequence_offset,
        }


@dataclass(frozen=True)
class SessionDefinition:
    session_id: str
    port: int
    probe_ids: tuple[str, ...]
    poll_period_ms: int
    datagram_timeout_ms: int
    ack_floor_bp: int
    lease_miss_limit: int
    depends_on: tuple[str, ...]

    def canonical(self) -> dict[str, Any]:
        return {
            "ack_floor_bp": self.ack_floor_bp,
            "datagram_timeout_ms": self.datagram_timeout_ms,
            "depends_on": list(self.depends_on),
            "lease_miss_limit": self.lease_miss_limit,
            "poll_period_ms": self.poll_period_ms,
            "port": self.port,
            "probe_ids": list(self.probe_ids),
            "session_id": self.session_id,
        }


@dataclass(frozen=True)
class DeclarationSet:
    format_version: int
    session_set_version: int
    probe_parameter_version: int
    ack_policy_version: int
    sessions: tuple[SessionDefinition, ...]
    probes: Mapping[str, ProbeDefinition]
    session_map: Mapping[str, SessionDefinition]
    content_digest: str

    def canonical_content(self) -> dict[str, Any]:
        return {
            "ack_policy_version": self.ack_policy_version,
            "format_version": self.format_version,
            "probe_parameter_version": self.probe_parameter_version,
            "probes": {
                probe_id: self.probes[probe_id].canonical()
                for probe_id in sorted(self.probes)
            },
            "session_set_version": self.session_set_version,
            "sessions": [session.canonical() for session in sorted(self.sessions, key=lambda s: s.session_id)],
        }

    def canonical_json(self) -> str:
        return _canonical_json(self.canonical_content())

    def session(self, session_id: str) -> SessionDefinition:
        try:
            return self.session_map[session_id]
        except KeyError as exc:
            raise SessionValidationError(
                "unknown session id", "session_id", {"session_id": session_id}
            ) from exc


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _read_json(path: Path, field_path: str) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise SessionValidationError("declaration file does not exist", field_path) from exc
    except json.JSONDecodeError as exc:
        raise SessionValidationError(
            "invalid JSON declaration",
            field_path,
            {"line": exc.lineno, "column": exc.colno, "reason": exc.msg},
        ) from exc
    except OSError as exc:
        raise SessionValidationError(
            "cannot read declaration file", field_path, {"reason": str(exc)}
        ) from exc


def _expect_object(value: Any, field_path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SessionValidationError("expected a JSON object", field_path)
    return value


def _expect_int(value: Any, field_path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SessionValidationError("expected an integer", field_path)
    return value


def _expect_string(value: Any, field_path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SessionValidationError("expected a non-empty string", field_path)
    return value


def _expect_list(value: Any, field_path: str) -> list[Any]:
    if not isinstance(value, list):
        raise SessionValidationError("expected an array", field_path)
    return value


def _expect_positive_int(value: Any, field_path: str) -> int:
    parsed = _expect_int(value, field_path)
    if parsed <= 0:
        raise SessionValidationError("expected a positive integer", field_path)
    return parsed


def _expect_version(value: Any, field_path: str) -> int:
    parsed = _expect_int(value, field_path)
    if parsed < 1:
        raise SessionValidationError("version must be >= 1", field_path)
    return parsed


def _load_probes(path: Path) -> tuple[int, int, dict[str, ProbeDefinition]]:
    raw = _expect_object(_read_json(path, "probes"), "probes")
    format_version = _expect_int(raw.get("format_version"), "probes.format_version")
    if format_version != SUPPORTED_FORMAT_VERSION:
        raise SessionValidationError(
            "unsupported probe declaration format version",
            "probes.format_version",
            {"supported": SUPPORTED_FORMAT_VERSION, "actual": format_version},
        )
    parameter_version = _expect_version(
        raw.get("probe_parameter_version"), "probes.probe_parameter_version"
    )
    definitions = _expect_object(raw.get("probes"), "probes.probes")
    result: dict[str, ProbeDefinition] = {}
    for index, probe_id in enumerate(definitions):
        base = f"probes.probes[{probe_id}]"
        _expect_string(probe_id, f"probes.probes[{index}]")
        item = _expect_object(definitions[probe_id], base)
        seed = _expect_int(item.get("seed"), f"{base}.seed")
        behavior = _expect_string(item.get("behavior"), f"{base}.behavior")
        if behavior not in {"normal", "drop", "delay"}:
            raise SessionValidationError(
                "behavior must be normal, drop, or delay", f"{base}.behavior"
            )
        delay_ms = _expect_int(item.get("delay_ms", 0), f"{base}.delay_ms")
        if delay_ms < 0:
            raise SessionValidationError("delay_ms must be >= 0", f"{base}.delay_ms")
        if behavior == "delay" and delay_ms <= 0:
            raise SessionValidationError(
                "delayed probes require delay_ms > 0", f"{base}.delay_ms"
            )
        sequence_offset = _expect_int(
            item.get("sequence_offset", 0), f"{base}.sequence_offset"
        )
        result[probe_id] = ProbeDefinition(
            probe_id=probe_id,
            seed=seed,
            behavior=behavior,
            delay_ms=delay_ms,
            sequence_offset=sequence_offset,
        )
    if not result:
        raise SessionValidationError("at least one probe is required", "probes.probes")
    return format_version, parameter_version, result


def _load_ack_policy(path: Path) -> tuple[int, int]:
    raw = _expect_object(_read_json(path, "ack_policy"), "ack_policy")
    format_version = _expect_int(
        raw.get("format_version"), "ack_policy.format_version"
    )
    if format_version != SUPPORTED_FORMAT_VERSION:
        raise SessionValidationError(
            "unsupported acknowledgement policy format version",
            "ack_policy.format_version",
            {"supported": SUPPORTED_FORMAT_VERSION, "actual": format_version},
        )
    policy_version = _expect_version(
        raw.get("ack_policy_version"), "ack_policy.ack_policy_version"
    )
    ratio_scale = _expect_int(raw.get("ratio_scale"), "ack_policy.ratio_scale")
    if ratio_scale != RATIO_SCALE:
        raise GrantError(
            "unsupported acknowledgement ratio scale",
            "ack_policy.ratio_scale",
            {"expected": RATIO_SCALE, "actual": ratio_scale},
        )
    rounding = raw.get("rounding", "half_up")
    if rounding != "half_up":
        raise GrantError(
            "unsupported acknowledgement rounding policy",
            "ack_policy.rounding",
            {"expected": "half_up", "actual": rounding},
        )
    return format_version, policy_version


def _load_sessions(
    path: Path,
    probes: Mapping[str, ProbeDefinition],
) -> tuple[int, int, int, list[SessionDefinition]]:
    raw = _expect_object(_read_json(path, "sessions"), "sessions")
    format_version = _expect_int(raw.get("format_version"), "sessions.format_version")
    if format_version != SUPPORTED_FORMAT_VERSION:
        raise SessionValidationError(
            "unsupported session declaration format version",
            "sessions.format_version",
            {"supported": SUPPORTED_FORMAT_VERSION, "actual": format_version},
        )
    session_set_version = _expect_version(
        raw.get("session_set_version"), "sessions.session_set_version"
    )
    declared_policy_version = _expect_version(
        raw.get("ack_policy_version"), "sessions.ack_policy_version"
    )
    session_items = _expect_list(raw.get("sessions"), "sessions.sessions")
    if not session_items:
        raise SessionValidationError("at least one session is required", "sessions.sessions")

    sessions: list[SessionDefinition] = []
    seen_ids: set[str] = set()
    seen_ports: set[int] = set()
    for index, item_value in enumerate(session_items):
        base = f"sessions.sessions[{index}]"
        item = _expect_object(item_value, base)
        session_id = _expect_string(item.get("session_id"), f"{base}.session_id")
        if session_id in seen_ids:
            raise SessionValidationError(
                "duplicate session id", f"{base}.session_id", {"session_id": session_id}
            )
        seen_ids.add(session_id)
        port = _expect_int(item.get("port"), f"{base}.port")
        if port < 1 or port > 65535:
            raise SessionValidationError("port must be in 1..65535", f"{base}.port")
        if port in seen_ports:
            raise SessionValidationError(
                "duplicate session port", f"{base}.port", {"port": port}
            )
        seen_ports.add(port)
        probe_values = _expect_list(item.get("probe_ids"), f"{base}.probe_ids")
        if not probe_values:
            raise SessionValidationError(
                "probe_ids must not be empty", f"{base}.probe_ids"
            )
        probe_ids: list[str] = []
        for probe_index, probe_value in enumerate(probe_values):
            probe_id = _expect_string(probe_value, f"{base}.probe_ids[{probe_index}]")
            if probe_id in probe_ids:
                raise SessionValidationError(
                    "duplicate probe id in session",
                    f"{base}.probe_ids[{probe_index}]",
                )
            if probe_id not in probes:
                raise SessionValidationError(
                    "probe id has no parameter definition",
                    f"{base}.probe_ids[{probe_index}]",
                    {"probe_id": probe_id},
                )
            probe_ids.append(probe_id)
        poll_period_ms = _expect_positive_int(
            item.get("poll_period_ms"), f"{base}.poll_period_ms"
        )
        datagram_timeout_ms = _expect_positive_int(
            item.get("datagram_timeout_ms"), f"{base}.datagram_timeout_ms"
        )
        if datagram_timeout_ms >= poll_period_ms:
            raise SessionValidationError(
                "datagram timeout must be smaller than poll period",
                f"{base}.datagram_timeout_ms",
            )
        ack_floor_bp = _expect_int(
            item.get("ack_confirmation_floor_bp"), f"{base}.ack_confirmation_floor_bp"
        )
        if ack_floor_bp < 0 or ack_floor_bp > RATIO_SCALE:
            raise SessionValidationError(
                "ack confirmation floor must be in 0..10000",
                f"{base}.ack_confirmation_floor_bp",
            )
        lease_miss_limit = _expect_positive_int(
            item.get("lease_miss_limit"), f"{base}.lease_miss_limit"
        )
        dependency_values = item.get("depends_on", [])
        dependency_list = _expect_list(dependency_values, f"{base}.depends_on")
        dependencies: list[str] = []
        for dep_index, dep_value in enumerate(dependency_list):
            dependency = _expect_string(dep_value, f"{base}.depends_on[{dep_index}]")
            if dependency == session_id:
                raise DependencyError(
                    "session cannot depend on itself",
                    f"{base}.depends_on[{dep_index}]",
                )
            if dependency in dependencies:
                raise SessionValidationError(
                    "duplicate dependency", f"{base}.depends_on[{dep_index}]"
                )
            dependencies.append(dependency)
        sessions.append(
            SessionDefinition(
                session_id=session_id,
                port=port,
                probe_ids=tuple(probe_ids),
                poll_period_ms=poll_period_ms,
                datagram_timeout_ms=datagram_timeout_ms,
                ack_floor_bp=ack_floor_bp,
                lease_miss_limit=lease_miss_limit,
                depends_on=tuple(dependencies),
            )
        )

    by_id = {session.session_id: session for session in sessions}
    for index, session in enumerate(sessions):
        for dep_index, dependency in enumerate(session.depends_on):
            if dependency not in by_id:
                raise SessionValidationError(
                    "dependency target does not exist",
                    f"sessions.sessions[{index}].depends_on[{dep_index}]",
                    {"dependency": dependency},
                )
    _detect_cycle(sessions)
    return format_version, session_set_version, declared_policy_version, sessions


def _detect_cycle(sessions: Iterable[SessionDefinition]) -> None:
    session_list = list(sessions)
    by_id = {session.session_id: session for session in session_list}
    state: dict[str, int] = {session.session_id: 0 for session in session_list}
    stack: list[str] = []
    index_by_id = {session.session_id: i for i, session in enumerate(session_list)}

    def visit(session_id: str) -> None:
        state[session_id] = 1
        stack.append(session_id)
        for edge_index, dependency in enumerate(by_id[session_id].depends_on):
            if state[dependency] == 0:
                visit(dependency)
            elif state[dependency] == 1:
                cycle_start = stack.index(dependency)
                cycle = stack[cycle_start:] + [dependency]
                source_index = index_by_id[session_id]
                raise DependencyError(
                    "session dependency graph contains a cycle",
                    f"sessions.sessions[{source_index}].depends_on[{edge_index}]",
                    {"cycle": cycle},
                )
        stack.pop()
        state[session_id] = 2

    for session in session_list:
        if state[session.session_id] == 0:
            visit(session.session_id)


def load_declarations(
    sessions_path: str | Path,
    probes_path: str | Path,
    ack_policy_path: str | Path,
) -> DeclarationSet:
    """Load all three declaration files or reject the whole set."""

    sessions_file = Path(sessions_path)
    probes_file = Path(probes_path)
    policy_file = Path(ack_policy_path)
    probes_format, probe_version, probes = _load_probes(probes_file)
    policy_format, policy_version = _load_ack_policy(policy_file)
    sessions_format, session_version, declared_policy, sessions = _load_sessions(
        sessions_file, probes
    )
    if probes_format != sessions_format or policy_format != sessions_format:
        raise SessionValidationError(
            "declaration format versions disagree",
            "sessions.format_version",
            {
                "sessions": sessions_format,
                "probes": probes_format,
                "ack_policy": policy_format,
            },
        )
    if declared_policy != policy_version:
        raise GrantError(
            "acknowledgement policy version mismatch",
            "sessions.ack_policy_version",
            {"declared": declared_policy, "loaded": policy_version},
        )
    probe_map = MappingProxyType(dict(probes))
    session_map = MappingProxyType({session.session_id: session for session in sessions})
    provisional = DeclarationSet(
        format_version=sessions_format,
        session_set_version=session_version,
        probe_parameter_version=probe_version,
        ack_policy_version=policy_version,
        sessions=tuple(sessions),
        probes=probe_map,
        session_map=session_map,
        content_digest="",
    )
    content_digest = _digest(provisional.canonical_content())
    return DeclarationSet(
        format_version=provisional.format_version,
        session_set_version=provisional.session_set_version,
        probe_parameter_version=provisional.probe_parameter_version,
        ack_policy_version=provisional.ack_policy_version,
        sessions=provisional.sessions,
        probes=probe_map,
        session_map=session_map,
        content_digest=content_digest,
    )


def declaration_from_dicts(
    sessions_doc: Mapping[str, Any],
    probes_doc: Mapping[str, Any],
    ack_policy_doc: Mapping[str, Any],
    temp_dir: str | Path,
) -> DeclarationSet:
    """Convenience helper for tests and integrations that hold documents in memory."""

    root = Path(temp_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, document in (
        ("sessions.json", sessions_doc),
        ("probes.json", probes_doc),
        ("ack_policy.json", ack_policy_doc),
    ):
        path = root / name
        path.write_text(
            json.dumps(document, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )
        paths.append(path)
    return load_declarations(paths[0], paths[1], paths[2])
