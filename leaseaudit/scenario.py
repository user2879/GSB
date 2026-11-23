from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

from .errors import BusinessError, ErrorCode
from .util import canonical_digest, is_int, read_json_strict


SUPPORTED_SCENARIO_FORMAT = 1
WATERMARK_SCALE_DEFAULT = 1_000_000
WATERMARK_TRUNCATION = "FLOOR_TOWARD_ZERO"

_ROOT_REQUIRED = {
    "cluster_capacity",
    "lease_duration_ms",
    "renew_interval_ms",
    "reclaim_threshold",
    "load_duration_ms",
    "sources",
}
_ROOT_ALLOWED = _ROOT_REQUIRED | {
    "format_version",
    "scenario_version",
    "lease_params_version",
    "reclaim_window_rounds",
    "observation_window_ms",
    "watermark_scale",
}
_SOURCE_REQUIRED = {"source_id", "address", "port", "lease_share", "reserve_quota"}
_SOURCE_ALLOWED = _SOURCE_REQUIRED | {
    "mode",
    "processing_ms",
    "request_interval_ms",
    "seed",
    "initial_issue",
    "stop_renew_at_ms",
    "request_timeout_ms",
}


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    address: str
    port: int
    lease_share: int
    reserve_quota: int
    mode: str = "success"
    processing_ms: int = 0
    request_interval_ms: int = 200
    seed: int = 0
    initial_issue: int | None = None
    stop_renew_at_ms: int | None = None
    request_timeout_ms: int = 1000

    @property
    def effective_initial_issue(self) -> int:
        return self.lease_share if self.initial_issue is None else self.initial_issue

    def param_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "lease_share": self.lease_share,
            "reserve_quota": self.reserve_quota,
        }

    def external_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "address": self.address,
            "port": self.port,
            "mode": self.mode,
            "processing_ms": self.processing_ms,
            "request_interval_ms": self.request_interval_ms,
            "seed": self.seed,
            "initial_issue": self.initial_issue,
            "stop_renew_at_ms": self.stop_renew_at_ms,
            "request_timeout_ms": self.request_timeout_ms,
        }


@dataclass(frozen=True)
class Scenario:
    cluster_capacity: int
    lease_duration_ms: int
    renew_interval_ms: int
    reclaim_threshold: int
    load_duration_ms: int
    sources: Tuple[SourceSpec, ...]
    format_version: int = SUPPORTED_SCENARIO_FORMAT
    scenario_version: str = "1"
    lease_params_version: int = 1
    reclaim_window_rounds: int = 3
    observation_window_ms: int = 250
    watermark_scale: int = WATERMARK_SCALE_DEFAULT

    def source_map(self) -> Dict[str, SourceSpec]:
        return {source.source_id: source for source in self.sources}

    def ordered_sources(self) -> Tuple[SourceSpec, ...]:
        return tuple(sorted(self.sources, key=lambda source: source.source_id))

    def params_dict(self) -> dict:
        return {
            "format_version": self.format_version,
            "lease_params_version": self.lease_params_version,
            "cluster_capacity": self.cluster_capacity,
            "lease_duration_ms": self.lease_duration_ms,
            "renew_interval_ms": self.renew_interval_ms,
            "reclaim_threshold": self.reclaim_threshold,
            "reclaim_window_rounds": self.reclaim_window_rounds,
            "observation_window_ms": self.observation_window_ms,
            "watermark_scale": self.watermark_scale,
            "watermark_truncation": WATERMARK_TRUNCATION,
            "load_duration_ms": self.load_duration_ms,
            "sources": [source.param_dict() for source in self.ordered_sources()],
        }

    def external_dict(self) -> dict:
        return {
            "scenario_version": self.scenario_version,
            "sources": [source.external_dict() for source in self.ordered_sources()],
        }

    def canonical_dict(self) -> dict:
        return {
            "format_version": self.format_version,
            "scenario_version": self.scenario_version,
            **{key: value for key, value in self.params_dict().items() if key != "format_version"},
            "sources": [
                {**source.param_dict(), **source.external_dict()}
                for source in self.ordered_sources()
            ],
        }

    @property
    def params_digest(self) -> str:
        return canonical_digest(self.params_dict())

    @property
    def external_digest(self) -> str:
        return canonical_digest(self.external_dict())

    def scenario_digest_document(self) -> dict:
        return self.canonical_dict()

    @property
    def scenario_digest(self) -> str:
        return canonical_digest(self.scenario_digest_document())

    @property
    def initial_retained_total(self) -> int:
        return sum(source.reserve_quota for source in self.sources)

    @property
    def initial_global_allocatable(self) -> int:
        return self.cluster_capacity - self.initial_retained_total

    def to_json_dict(self) -> dict:
        """Return a fully explicit, round-trippable scenario document."""

        return {
            "format_version": self.format_version,
            "scenario_version": self.scenario_version,
            "lease_params_version": self.lease_params_version,
            "cluster_capacity": self.cluster_capacity,
            "lease_duration_ms": self.lease_duration_ms,
            "renew_interval_ms": self.renew_interval_ms,
            "reclaim_threshold": self.reclaim_threshold,
            "reclaim_window_rounds": self.reclaim_window_rounds,
            "observation_window_ms": self.observation_window_ms,
            "watermark_scale": self.watermark_scale,
            "load_duration_ms": self.load_duration_ms,
            "sources": [
                {
                    "source_id": source.source_id,
                    "address": source.address,
                    "port": source.port,
                    "lease_share": source.lease_share,
                    "reserve_quota": source.reserve_quota,
                    "mode": source.mode,
                    "processing_ms": source.processing_ms,
                    "request_interval_ms": source.request_interval_ms,
                    "seed": source.seed,
                    "initial_issue": source.effective_initial_issue,
                    "stop_renew_at_ms": source.stop_renew_at_ms,
                    "request_timeout_ms": source.request_timeout_ms,
                }
                for source in self.ordered_sources()
            ],
        }


def _scene_error(message: str, field_path: str, **details: Any) -> BusinessError:
    return BusinessError(
        ErrorCode.E_SCENE,
        message,
        field_path=field_path,
        details=details,
    )


def _quota_error(message: str, field_path: str, **details: Any) -> BusinessError:
    return BusinessError(
        ErrorCode.E_QUOTA,
        message,
        field_path=field_path,
        details=details,
    )


def _require_int(
    container: Mapping[str, Any],
    key: str,
    path: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
    positive: bool = False,
) -> int:
    if key not in container:
        raise _scene_error("required integer field is missing", f"{path}.{key}" if path else key)
    value = container[key]
    field_path = f"{path}.{key}" if path else key
    if not is_int(value):
        raise _scene_error("value must be an integer (booleans and floats are rejected)", field_path)
    if positive and value <= 0:
        raise _scene_error("value must be greater than zero", field_path, value=value)
    if minimum is not None and value < minimum:
        raise _scene_error("value is below the allowed minimum", field_path, value=value, minimum=minimum)
    if maximum is not None and value > maximum:
        raise _scene_error("value is above the allowed maximum", field_path, value=value, maximum=maximum)
    return value


def _optional_int(
    container: Mapping[str, Any],
    key: str,
    path: str,
    default: int | None,
    *,
    minimum: int | None = None,
    allow_null: bool = False,
) -> int | None:
    if key not in container:
        return default
    value = container[key]
    field_path = f"{path}.{key}" if path else key
    if value is None and allow_null:
        return None
    if not is_int(value):
        raise _scene_error("value must be an integer", field_path)
    if minimum is not None and value < minimum:
        raise _scene_error("value is below the allowed minimum", field_path, value=value, minimum=minimum)
    return value


def _require_string(container: Mapping[str, Any], key: str, path: str) -> str:
    if key not in container:
        raise _scene_error("required string field is missing", f"{path}.{key}" if path else key)
    value = container[key]
    field_path = f"{path}.{key}" if path else key
    if not isinstance(value, str) or not value.strip():
        raise _scene_error("value must be a non-empty string", field_path)
    return value


def validate_scenario(document: Any) -> Scenario:
    if not isinstance(document, dict):
        raise _scene_error("scenario root must be a JSON object", "$")

    missing = sorted(_ROOT_REQUIRED - set(document))
    if missing:
        raise _scene_error("required root field is missing", missing[0])
    unknown = sorted(set(document) - _ROOT_ALLOWED)
    if unknown:
        raise _scene_error("unknown root field", unknown[0])

    format_version = _optional_int(document, "format_version", "", SUPPORTED_SCENARIO_FORMAT, minimum=1)
    if format_version != SUPPORTED_SCENARIO_FORMAT:
        raise _scene_error(
            "unsupported scenario format version",
            "format_version",
            supported=SUPPORTED_SCENARIO_FORMAT,
            actual=format_version,
        )
    scenario_version = document.get("scenario_version", "1")
    if not isinstance(scenario_version, str) or not scenario_version:
        raise _scene_error("value must be a non-empty string", "scenario_version")
    lease_params_version = _optional_int(document, "lease_params_version", "", 1, minimum=1)

    capacity = _require_int(document, "cluster_capacity", "", positive=True)
    lease_duration_ms = _require_int(document, "lease_duration_ms", "", positive=True)
    renew_interval_ms = _require_int(document, "renew_interval_ms", "", minimum=0)
    if renew_interval_ms >= lease_duration_ms:
        raise _scene_error(
            "renew interval must be strictly smaller than lease duration",
            "renew_interval_ms",
            renew_interval_ms=renew_interval_ms,
            lease_duration_ms=lease_duration_ms,
        )
    reclaim_threshold = _require_int(document, "reclaim_threshold", "", minimum=0)
    load_duration_ms = _require_int(document, "load_duration_ms", "", positive=True)
    reclaim_window_rounds = _optional_int(
        document, "reclaim_window_rounds", "", 3, minimum=0
    )
    observation_window_ms = _optional_int(
        document, "observation_window_ms", "", 250, minimum=1
    )
    watermark_scale = _optional_int(
        document, "watermark_scale", "", WATERMARK_SCALE_DEFAULT, minimum=1
    )

    raw_sources = document["sources"]
    if not isinstance(raw_sources, list) or not raw_sources:
        raise _scene_error("sources must be a non-empty JSON array", "sources")

    sources: list[SourceSpec] = []
    seen_ids: set[str] = set()
    seen_ports: set[int] = set()
    reserve_total = 0
    for index, raw in enumerate(raw_sources):
        base = f"sources[{index}]"
        if not isinstance(raw, dict):
            raise _scene_error("source entry must be a JSON object", base)
        missing_source = sorted(_SOURCE_REQUIRED - set(raw))
        if missing_source:
            raise _scene_error("required source field is missing", f"{base}.{missing_source[0]}")
        unknown_source = sorted(set(raw) - _SOURCE_ALLOWED)
        if unknown_source:
            raise _scene_error("unknown source field", f"{base}.{unknown_source[0]}")

        source_id = _require_string(raw, "source_id", base)
        if source_id in seen_ids:
            raise _scene_error("source_id must be globally unique", f"{base}.source_id", value=source_id)
        seen_ids.add(source_id)

        address = _require_string(raw, "address", base)
        if address != "127.0.0.1":
            raise _scene_error(
                "address must be the exact IPv4 loopback literal 127.0.0.1",
                f"{base}.address",
                value=address,
            )

        port = _require_int(raw, "port", base, minimum=1, maximum=65535)
        if port in seen_ports:
            raise _scene_error("port must be globally unique", f"{base}.port", value=port)
        seen_ports.add(port)

        lease_share = _require_int(raw, "lease_share", base, minimum=0)
        reserve_quota = _require_int(raw, "reserve_quota", base, minimum=0)
        if reserve_quota > lease_share:
            raise _scene_error(
                "reserve_quota cannot exceed lease_share",
                f"{base}.reserve_quota",
                reserve_quota=reserve_quota,
                lease_share=lease_share,
            )
        reserve_total += reserve_quota
        if reserve_total > capacity:
            raise _quota_error(
                "sum of source reserve quotas exceeds cluster capacity",
                f"{base}.reserve_quota",
                sum_reserve_quota=reserve_total,
                cluster_capacity=capacity,
                excess=reserve_total - capacity,
            )

        mode = raw.get("mode", "success")
        if mode not in {"success", "fixed_delay", "refuse"}:
            raise _scene_error(
                "mode must be success, fixed_delay, or refuse",
                f"{base}.mode",
                value=mode,
            )
        processing_ms = _optional_int(raw, "processing_ms", base, 0, minimum=0)
        request_interval_ms = _optional_int(raw, "request_interval_ms", base, 200, minimum=1)
        seed = _optional_int(raw, "seed", base, 0, minimum=0)
        initial_issue = _optional_int(raw, "initial_issue", base, lease_share, minimum=0)
        if initial_issue is None or initial_issue > lease_share:
            raise _scene_error(
                "initial_issue must be between zero and lease_share",
                f"{base}.initial_issue",
                initial_issue=initial_issue,
                lease_share=lease_share,
            )
        stop_renew_at_ms = _optional_int(
            raw, "stop_renew_at_ms", base, None, minimum=0, allow_null=True
        )
        request_timeout_ms = _optional_int(raw, "request_timeout_ms", base, 1000, minimum=1)
        if mode == "fixed_delay" and (processing_ms is None or processing_ms <= 0):
            raise _scene_error(
                "fixed_delay mode requires processing_ms > 0",
                f"{base}.processing_ms",
            )

        sources.append(
            SourceSpec(
                source_id=source_id,
                address=address,
                port=port,
                lease_share=lease_share,
                reserve_quota=reserve_quota,
                mode=mode,
                processing_ms=processing_ms or 0,
                request_interval_ms=request_interval_ms or 200,
                seed=seed or 0,
                initial_issue=initial_issue,
                stop_renew_at_ms=stop_renew_at_ms,
                request_timeout_ms=request_timeout_ms or 1000,
            )
        )

    if reserve_total > capacity:
        raise _quota_error(
            "sum of source reserve quotas exceeds cluster capacity",
            "sources[*].reserve_quota",
            sum_reserve_quota=reserve_total,
            cluster_capacity=capacity,
        )
    return Scenario(
        cluster_capacity=capacity,
        lease_duration_ms=lease_duration_ms,
        renew_interval_ms=renew_interval_ms,
        reclaim_threshold=reclaim_threshold,
        load_duration_ms=load_duration_ms,
        sources=tuple(sorted(sources, key=lambda source: source.source_id)),
        format_version=format_version or SUPPORTED_SCENARIO_FORMAT,
        scenario_version=scenario_version,
        lease_params_version=lease_params_version or 1,
        reclaim_window_rounds=reclaim_window_rounds or 0,
        observation_window_ms=observation_window_ms or 250,
        watermark_scale=watermark_scale or WATERMARK_SCALE_DEFAULT,
    )


def load_scenario(path: str | Path) -> Scenario:
    try:
        document = read_json_strict(path)
    except FileNotFoundError as exc:
        raise _scene_error("scenario file does not exist", "$.file", path=str(path)) from exc
    except (OSError, UnicodeError) as exc:
        raise _scene_error("scenario file cannot be read", "$.file", path=str(path), error=str(exc)) from exc
    except Exception as exc:
        raise _scene_error(
            "scenario file is not valid JSON",
            "$",
            path=str(path),
            error=str(exc),
        ) from exc
    return validate_scenario(document)


def scenario_from_params(params: Mapping[str, Any], scenario_version: str = "ledger") -> Scenario:
    """Reconstruct replay parameters stored inside the ledger."""

    doc = dict(params)
    doc.pop("watermark_truncation", None)
    doc.setdefault("scenario_version", scenario_version)
    doc["sources"] = [
        {
            "source_id": source["source_id"],
            "address": "127.0.0.1",
            "port": 1 + index,
            "lease_share": source["lease_share"],
            "reserve_quota": source["reserve_quota"],
            "initial_issue": 0,
        }
        for index, source in enumerate(sorted(doc.get("sources", []), key=lambda item: item["source_id"]))
    ]
    # Replay only uses semantic parameters; synthetic ports satisfy the loader.
    return validate_scenario(doc)


def scenario_diff(ledger_doc: Mapping[str, Any], current: Scenario) -> list[dict]:
    """Return deterministic, human-readable field differences for drift reporting."""

    current_doc = current.scenario_digest_document()
    differences: list[dict] = []

    def walk(left: Any, right: Any, path: str) -> None:
        if type(left) is not type(right):
            differences.append({"field_path": path, "ledger": left, "current": right})
            return
        if isinstance(left, dict):
            for key in sorted(set(left) | set(right)):
                child = f"{path}.{key}" if path else key
                if key not in left:
                    differences.append({"field_path": child, "ledger": "<missing>", "current": right[key]})
                elif key not in right:
                    differences.append({"field_path": child, "ledger": left[key], "current": "<missing>"})
                else:
                    walk(left[key], right[key], child)
            return
        if isinstance(left, list):
            if len(left) != len(right):
                differences.append({"field_path": f"{path}.length", "ledger": len(left), "current": len(right)})
                return
            for index, (lval, rval) in enumerate(zip(left, right)):
                walk(lval, rval, f"{path}[{index}]")
            return
        if left != right:
            differences.append({"field_path": path, "ledger": left, "current": right})

    walk(dict(ledger_doc), current_doc, "")
    return differences
