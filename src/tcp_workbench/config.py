"""All-or-nothing JSON loading and strong semantic validation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .errors import E_CONFIG, BusinessError, config_error
from .models import (
    BudgetConfig,
    CongestionConfig,
    PeerResponse,
    PeerScript,
    SessionConfig,
    canonical_json,
    digest_of,
)

MAX_SEQ = 2**32 - 1
CONFIG_KEYS = {
    "schema_version",
    "config_version",
    "session_id",
    "initial_seq",
    "window_limit",
    "mss",
    "peer_script",
    "budget",
    "congestion",
}
CONGESTION_KEYS = {
    "initial_cwnd_mss",
    "initial_ssthresh_bytes",
    "rtt_scale",
    "initial_rto_ms",
    "min_rto_ms",
    "max_rto_ms",
}
BUDGET_KEYS = {"max_events", "max_sent_bytes", "max_elapsed_ms"}
SCRIPT_KEYS = {
    "schema_version",
    "script_version",
    "default_policy",
    "default_delay_ms",
    "responses",
}
RESPONSE_KEYS = {"for_seq", "action", "ack_seq", "delay_ms", "reorder"}


class _DuplicateKey(ValueError):
    pass


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def read_json(
    path: str | os.PathLike[str], *, error_code: str = E_CONFIG, field: str = "$"
) -> Any:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise BusinessError(
            error_code, f"cannot read JSON file: {exc}", [field]
        ) from exc
    try:
        return json.loads(
            text,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, _DuplicateKey, ValueError) as exc:
        raise BusinessError(error_code, f"invalid JSON: {exc}", [field]) from exc


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_object(
    value: Any, path: str, required: set[str], allowed: set[str], errors: list[str]
) -> dict[str, Any]:
    if not isinstance(value, dict):
        errors.append(path)
        return {}
    for key in sorted(required - set(value)):
        errors.append(f"{path}.{key}" if path else key)
    for key in sorted(set(value) - allowed):
        errors.append(f"{path}.{key}" if path else key)
    return value


def _integer(
    obj: dict[str, Any],
    key: str,
    path: str,
    errors: list[str],
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int | None:
    field = f"{path}.{key}" if path else key
    value = obj.get(key)
    if not _is_int(value):
        errors.append(field)
        return None
    if minimum is not None and value < minimum:
        errors.append(field)
    if maximum is not None and value > maximum:
        errors.append(field)
    return value


def validate_session_config(
    raw: Any,
    *,
    source_path: str = "",
    script_source_path: str = "",
) -> tuple[SessionConfig, list[str]]:
    errors: list[str] = []
    obj = _check_object(raw, "$", CONFIG_KEYS, CONFIG_KEYS, errors)
    schema_version = _integer(obj, "schema_version", "$", errors, minimum=1)
    config_version = _integer(obj, "config_version", "$", errors, minimum=1)
    session_id = obj.get("session_id")
    if not isinstance(session_id, str) or not session_id.strip():
        errors.append("$.session_id")
    initial_seq = _integer(
        obj, "initial_seq", "$", errors, minimum=0, maximum=MAX_SEQ - 1
    )
    window_limit = _integer(
        obj, "window_limit", "$", errors, minimum=1, maximum=MAX_SEQ
    )
    mss = _integer(obj, "mss", "$", errors, minimum=1, maximum=MAX_SEQ)
    peer_script = obj.get("peer_script")
    if not isinstance(peer_script, str) or not peer_script.strip():
        errors.append("$.peer_script")

    budget_obj = _check_object(
        obj.get("budget"), "$.budget", BUDGET_KEYS, BUDGET_KEYS, errors
    )
    max_events = _integer(budget_obj, "max_events", "$.budget", errors, minimum=1)
    max_sent_bytes = _integer(
        budget_obj, "max_sent_bytes", "$.budget", errors, minimum=1, maximum=MAX_SEQ
    )
    max_elapsed_ms = _integer(
        budget_obj, "max_elapsed_ms", "$.budget", errors, minimum=1
    )

    congestion_obj = _check_object(
        obj.get("congestion"), "$.congestion", CONGESTION_KEYS, CONGESTION_KEYS, errors
    )
    initial_cwnd_mss = _integer(
        congestion_obj, "initial_cwnd_mss", "$.congestion", errors, minimum=1
    )
    initial_ssthresh = _integer(
        congestion_obj,
        "initial_ssthresh_bytes",
        "$.congestion",
        errors,
        minimum=1,
        maximum=MAX_SEQ,
    )
    rtt_scale = _integer(congestion_obj, "rtt_scale", "$.congestion", errors, minimum=1)
    initial_rto = _integer(
        congestion_obj, "initial_rto_ms", "$.congestion", errors, minimum=1
    )
    min_rto = _integer(congestion_obj, "min_rto_ms", "$.congestion", errors, minimum=1)
    max_rto = _integer(congestion_obj, "max_rto_ms", "$.congestion", errors, minimum=1)

    if mss is not None and window_limit is not None and window_limit < mss:
        errors.append("$.window_limit")
    if (
        mss is not None
        and max_sent_bytes is not None
        and (max_sent_bytes < mss or max_sent_bytes % mss != 0)
    ):
        errors.append("$.budget.max_sent_bytes")
    if (
        initial_seq is not None
        and max_sent_bytes is not None
        and initial_seq + 1 + max_sent_bytes > MAX_SEQ
    ):
        errors.append("$.initial_seq")
    if (
        initial_cwnd_mss is not None
        and mss is not None
        and window_limit is not None
        and initial_cwnd_mss * mss > window_limit
    ):
        errors.append("$.congestion.initial_cwnd_mss")
    if initial_ssthresh is not None and mss is not None and initial_ssthresh < 2 * mss:
        errors.append("$.congestion.initial_ssthresh_bytes")
    if (
        initial_ssthresh is not None
        and window_limit is not None
        and initial_ssthresh > window_limit
    ):
        errors.append("$.congestion.initial_ssthresh_bytes")
    if rtt_scale is not None and rtt_scale != 8:
        errors.append("$.congestion.rtt_scale")
    if min_rto is not None and max_rto is not None and min_rto > max_rto:
        errors.append("$.congestion.min_rto_ms")
    if initial_rto is not None and min_rto is not None and initial_rto < min_rto:
        errors.append("$.congestion.initial_rto_ms")
    if initial_rto is not None and max_rto is not None and initial_rto > max_rto:
        errors.append("$.congestion.initial_rto_ms")

    if errors:
        return None, errors  # type: ignore[return-value]

    budget = BudgetConfig(max_events, max_sent_bytes, max_elapsed_ms)  # type: ignore[arg-type]
    congestion = CongestionConfig(  # type: ignore[arg-type]
        initial_cwnd_mss,
        initial_ssthresh,
        rtt_scale,
        initial_rto,
        min_rto,
        max_rto,
    )
    config = SessionConfig(
        schema_version=schema_version,  # type: ignore[arg-type]
        config_version=config_version,  # type: ignore[arg-type]
        session_id=session_id.strip(),  # type: ignore[union-attr]
        initial_seq=initial_seq,  # type: ignore[arg-type]
        window_limit=window_limit,  # type: ignore[arg-type]
        mss=mss,  # type: ignore[arg-type]
        peer_script=peer_script,  # type: ignore[arg-type]
        budget=budget,
        congestion=congestion,
        source_path=source_path or str(Path(source_path)),
        raw=dict(raw),
        digest=digest_of(raw),
    )
    return config, []


def validate_peer_script(
    raw: Any,
    config: SessionConfig,
    *,
    source_path: str = "",
) -> tuple[PeerScript, list[str]]:
    errors: list[str] = []
    obj = _check_object(raw, "peer_script", SCRIPT_KEYS, SCRIPT_KEYS, errors)
    schema_version = _integer(obj, "schema_version", "peer_script", errors, minimum=1)
    script_version = _integer(obj, "script_version", "peer_script", errors, minimum=1)
    default_policy = obj.get("default_policy")
    if default_policy not in {"ack", "drop"}:
        errors.append("peer_script.default_policy")
    default_delay = _integer(obj, "default_delay_ms", "peer_script", errors, minimum=0)
    responses_value = obj.get("responses")
    if not isinstance(responses_value, list):
        errors.append("peer_script.responses")
        responses_value = []

    first = config.first_data_seq
    last_start = first + config.budget.max_sent_bytes - config.mss
    last_ack = first + config.budget.max_sent_bytes
    parsed: list[PeerResponse] = []
    seen: set[int] = set()
    for index, item in enumerate(responses_value):
        path = f"peer_script.responses[{index}]"
        response_obj = _check_object(
            item, path, {"for_seq", "action"}, RESPONSE_KEYS, errors
        )
        for_seq = _integer(
            response_obj, "for_seq", path, errors, minimum=first, maximum=last_start
        )
        action = response_obj.get("action")
        if action not in {"ack", "dupack", "drop"}:
            errors.append(f"{path}.action")
        delay_ms = (
            _integer(response_obj, "delay_ms", path, errors, minimum=0)
            if "delay_ms" in response_obj
            else 0
        )
        reorder = response_obj.get("reorder", False)
        if not isinstance(reorder, bool):
            errors.append(f"{path}.reorder")
        ack_seq: int | None = None
        if action in {"ack", "dupack"}:
            if "ack_seq" not in response_obj:
                errors.append(f"{path}.ack_seq")
            else:
                ack_seq = _integer(
                    response_obj,
                    "ack_seq",
                    path,
                    errors,
                    minimum=first,
                    maximum=last_ack,
                )
        elif response_obj.get("ack_seq") is not None:
            errors.append(f"{path}.ack_seq")
        if action == "drop" and reorder is True:
            errors.append(f"{path}.reorder")
        if for_seq is not None:
            if for_seq not in seen:
                seen.add(for_seq)
            else:
                errors.append(f"{path}.for_seq")
            if (for_seq - first) % config.mss != 0:
                errors.append(f"{path}.for_seq")
        if for_seq is not None and action in {"ack", "dupack"} and delay_ms is not None:
            parsed.append(
                PeerResponse(
                    for_seq=for_seq,
                    action=str(action),
                    ack_seq=ack_seq,
                    delay_ms=delay_ms,
                    reorder=bool(reorder),
                )
            )
        elif for_seq is not None:
            parsed.append(
                PeerResponse(
                    for_seq=for_seq,
                    action=str(action),
                    ack_seq=None,
                    delay_ms=default_delay if delay_ms is None else delay_ms,
                    reorder=False,
                )
            )

    if errors:
        return None, errors  # type: ignore[return-value]
    script = PeerScript(
        schema_version=schema_version,  # type: ignore[arg-type]
        script_version=script_version,  # type: ignore[arg-type]
        default_policy=str(default_policy),
        default_delay_ms=default_delay,  # type: ignore[arg-type]
        responses=tuple(parsed),
        source_path=source_path,
        raw=dict(raw),
        digest=digest_of(raw),
        response_map={response.for_seq: response for response in parsed},
    )
    return script, []


def load_session_config(
    config_path: str | os.PathLike[str],
    *,
    peer_script_override: str | os.PathLike[str] | None = None,
    script_error_code: str = E_CONFIG,
) -> tuple[SessionConfig, PeerScript]:
    """Load both documents and reject the whole input if any field is wrong."""

    raw = read_json(config_path, field="$")
    config, errors = validate_session_config(raw, source_path=str(Path(config_path)))
    if errors:
        raise config_error(errors)

    if peer_script_override is not None:
        script_path = Path(peer_script_override)
        if not script_path.is_absolute():
            script_path = Path.cwd() / script_path
    else:
        script_path = Path(config.peer_script)
        if not script_path.is_absolute():
            script_path = Path(config_path).resolve().parent / script_path
    try:
        script_raw = read_json(
            script_path, error_code=script_error_code, field="peer_script"
        )
    except BusinessError as exc:
        if script_error_code == E_CONFIG:
            raise config_error(exc.fields, exc.message) from exc
        raise
    script, script_errors = validate_peer_script(
        script_raw, config, source_path=str(script_path)
    )
    if script_errors:
        if script_error_code == E_CONFIG:
            raise config_error(script_errors)
        raise BusinessError(script_error_code, "invalid peer script", script_errors)
    return config, script


def config_digest(config: SessionConfig) -> str:
    return digest_of(config.raw)


def script_digest(script: PeerScript) -> str:
    return digest_of(script.raw)


def config_canonical_json(config: SessionConfig) -> str:
    return canonical_json(config.raw)


def script_canonical_json(script: PeerScript) -> str:
    return canonical_json(script.raw)
