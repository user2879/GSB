from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .errors import SpecError
from .fixedpoint import FixedPoint


STAGES = ("ingress", "relay", "egress")
STAGE_RANK = {name: index for index, name in enumerate(STAGES)}
HOST_LITERAL = "127.0.0.1"


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def pipelines(self) -> Path:
        return self.root / "pipelines.json"

    @property
    def shapers(self) -> Path:
        return self.root / "shapers.json"

    @property
    def events(self) -> Path:
        return self.root / "events.json"

    @property
    def config(self) -> Path:
        return self.root / "config.json"


def load_json_file(path: str | Path) -> Any:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SpecError(f"cannot read JSON file: {exc}", object_id=str(path), field="file") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise SpecError(
            f"invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}",
            object_id=str(path),
            field="json",
        ) from exc


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _object_id(raw: Any, collection: str, index: int) -> str:
    if isinstance(raw, dict):
        for key in ("pipeline_id", "shaper_id", "event_id", "id"):
            value = raw.get(key)
            if isinstance(value, str) and value:
                return value
    return f"{collection}[{index}]"


def _fail(object_id: str, field: str, message: str) -> None:
    raise SpecError(message, object_id=object_id, field=field)


def _require_dict(raw: Any, collection: str, index: int) -> str:
    object_id = _object_id(raw, collection, index)
    if not isinstance(raw, dict):
        _fail(object_id, "$", "object must be a JSON object")
    return object_id


def _require_nonempty_str(raw: dict[str, Any], field: str, object_id: str) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value:
        _fail(object_id, field, f"{field} must be a non-empty string")
    return value


def _require_positive_int(raw: dict[str, Any], field: str, object_id: str) -> int:
    value = raw.get(field)
    if not _is_int(value) or value <= 0:
        _fail(object_id, field, f"{field} must be a positive integer")
    return value


def _require_nonnegative_int(raw: dict[str, Any], field: str, object_id: str) -> int:
    value = raw.get(field)
    if not _is_int(value) or value < 0:
        _fail(object_id, field, f"{field} must be a non-negative integer")
    return value


def validate_config(raw: Any) -> dict[str, Any]:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise SpecError("config must be an object", object_id="$config", field="$")
    config = {
        "scale": raw.get("scale", 2),
        "window_ms": raw.get("window_ms", 20),
        "slide_ms": raw.get("slide_ms", 10),
        "global_max_inflight": raw.get("global_max_inflight", 4),
        "pipeline_max_inflight": raw.get("pipeline_max_inflight", 1),
        "call_timeout_ms": raw.get("call_timeout_ms", 1000),
        "batch_size": raw.get("batch_size", 4),
        "selection_deadband_work": raw.get("selection_deadband_work", 8),
    }
    FixedPoint(config["scale"])
    for field in ("window_ms", "slide_ms", "global_max_inflight", "call_timeout_ms", "batch_size"):
        if not _is_int(config[field]) or config[field] <= 0:
            _fail("$config", field, f"{field} must be a positive integer")
    if not _is_int(config["pipeline_max_inflight"]) or config["pipeline_max_inflight"] <= 0:
        _fail("$config", "pipeline_max_inflight", "must be a positive integer")
    if not _is_int(config["selection_deadband_work"]) or config["selection_deadband_work"] < 0:
        _fail("$config", "selection_deadband_work", "must be a non-negative integer")
    if config["slide_ms"] > config["window_ms"]:
        _fail("$config", "slide_ms", "slide_ms must not exceed window_ms")
    if config["window_ms"] % config["slide_ms"] != 0:
        _fail("$config", "window_ms", "window_ms must be an integer multiple of slide_ms")
    return config


def validate_pipelines(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, dict) or not isinstance(raw.get("pipelines"), list):
        raise SpecError("pipelines must contain a pipelines array", object_id="$pipelines", field="pipelines")
    pipelines: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw["pipelines"]):
        object_id = _require_dict(item, "pipelines", index)
        pipeline_id = _require_nonempty_str(item, "pipeline_id", object_id)
        if pipeline_id in seen:
            _fail(pipeline_id, "pipeline_id", "pipeline_id is duplicated")
        seen.add(pipeline_id)
        stages = item.get("stage_shaper_ids")
        if not isinstance(stages, dict):
            _fail(pipeline_id, "stage_shaper_ids", "stage_shaper_ids must be an object")
        normalized_stages: dict[str, str] = {}
        for stage in STAGES:
            shaper_id = stages.get(stage)
            if not isinstance(shaper_id, str) or not shaper_id:
                _fail(pipeline_id, f"stage_shaper_ids.{stage}", "must be a non-empty shaper_id")
            normalized_stages[stage] = shaper_id
        unknown = sorted(set(stages) - set(STAGES))
        if unknown:
            _fail(pipeline_id, f"stage_shaper_ids.{unknown[0]}", "unknown stage name")
        pipelines.append(
            {
                "pipeline_id": pipeline_id,
                "stage_shaper_ids": normalized_stages,
            }
        )
    if not pipelines:
        raise SpecError("at least one pipeline is required", object_id="$pipelines", field="pipelines")
    return pipelines


def validate_shapers(raw: Any, pipelines: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(raw, dict) or not isinstance(raw.get("shapers"), list):
        raise SpecError("shapers must contain a shapers array", object_id="$shapers", field="shapers")
    pipeline_by_id = {item["pipeline_id"]: item for item in pipelines}
    shapers: list[dict[str, Any]] = []
    seen_shaper_ids: set[str] = set()
    seen_ports: set[int] = set()
    expected_pairs: dict[tuple[str, str], str] = {}
    for pipeline in pipelines:
        for stage in STAGES:
            shaper_id = pipeline["stage_shaper_ids"][stage]
            key = (pipeline["pipeline_id"], stage)
            if shaper_id in expected_pairs.values():
                raise SpecError(
                    "shaper is linked from more than one pipeline stage",
                    object_id=shaper_id,
                    field="shaper_id",
                )
            expected_pairs[key] = shaper_id

    for index, item in enumerate(raw["shapers"]):
        object_id = _require_dict(item, "shapers", index)
        shaper_id = _require_nonempty_str(item, "shaper_id", object_id)
        if shaper_id in seen_shaper_ids:
            _fail(shaper_id, "shaper_id", "shaper_id is duplicated")
        seen_shaper_ids.add(shaper_id)

        pipeline_id = _require_nonempty_str(item, "pipeline_id", shaper_id)
        if pipeline_id not in pipeline_by_id:
            _fail(shaper_id, "pipeline_id", "pipeline_id does not reference a declared pipeline")
        stage = _require_nonempty_str(item, "stage", shaper_id)
        if stage not in STAGE_RANK:
            _fail(shaper_id, "stage", "stage must be ingress, relay or egress")

        linked_id = pipeline_by_id[pipeline_id]["stage_shaper_ids"][stage]
        if linked_id != shaper_id:
            _fail(shaper_id, "stage", "shaper does not match pipeline stage_shaper_ids")
        if expected_pairs.get((pipeline_id, stage)) != shaper_id:
            _fail(shaper_id, "stage", "duplicate shaper for pipeline stage")

        host = _require_nonempty_str(item, "host", shaper_id)
        if host != HOST_LITERAL:
            _fail(shaper_id, "host", f"host must be the literal {HOST_LITERAL!r}")
        port = item.get("port")
        if not _is_int(port) or isinstance(port, bool) or not (1024 <= port <= 65535):
            _fail(shaper_id, "port", "port must be an integer from 1024 through 65535")
        if port in seen_ports:
            _fail(shaper_id, "port", "port is duplicated")
        seen_ports.add(port)

        rate = _require_positive_int(item, "refill_rate_per_ms", shaper_id)
        burst = _require_positive_int(item, "burst_capacity_tokens", shaper_id)
        if rate > 1_000_000_000:
            _fail(shaper_id, "refill_rate_per_ms", "rate is too large for deterministic saturation bounds")
        if burst > 1_000_000_000:
            _fail(shaper_id, "burst_capacity_tokens", "burst is too large for deterministic saturation bounds")
        delay_ms = item.get("service_delay_ms", 0)
        if not _is_int(delay_ms) or delay_ms < 0:
            _fail(shaper_id, "service_delay_ms", "service_delay_ms must be a non-negative integer")

        shapers.append(
            {
                "shaper_id": shaper_id,
                "pipeline_id": pipeline_id,
                "stage": stage,
                "host": host,
                "port": port,
                "refill_rate_per_ms": rate,
                "burst_capacity_tokens": burst,
                "service_delay_ms": delay_ms,
            }
        )

    if len(shapers) != len(pipelines) * 3:
        raise SpecError(
            "each pipeline must have exactly three declared shapers",
            object_id="$shapers",
            field="shapers",
        )
    declared_pairs = {(item["pipeline_id"], item["stage"]) for item in shapers}
    missing = [pair for pair in expected_pairs if pair not in declared_pairs]
    if missing:
        pipeline_id, stage = sorted(missing)[0]
        raise SpecError(
            "pipeline stage is missing a shaper declaration",
            object_id=pipeline_id,
            field=f"stage_shaper_ids.{stage}",
        )
    return sorted(shapers, key=lambda item: item["shaper_id"])


def validate_events(raw: Any, pipelines: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(raw, dict) or not isinstance(raw.get("events"), list):
        raise SpecError("events must contain an events array", object_id="$events", field="events")
    pipeline_ids = {item["pipeline_id"] for item in pipelines}
    events: list[dict[str, Any]] = []
    seen_event_ids: set[str] = set()
    previous_arrival: int | None = None
    for index, item in enumerate(raw["events"]):
        object_id = _require_dict(item, "events", index)
        event_id = _require_nonempty_str(item, "event_id", object_id)
        if event_id in seen_event_ids:
            _fail(event_id, "event_id", "event_id is duplicated")
        seen_event_ids.add(event_id)
        pipeline_id = _require_nonempty_str(item, "pipeline_id", event_id)
        if pipeline_id not in pipeline_ids:
            _fail(event_id, "pipeline_id", "event references a missing pipeline")
        arrival = _require_nonnegative_int(item, "arrival_ms", event_id)
        if previous_arrival is not None and arrival < previous_arrival:
            _fail(event_id, "arrival_ms", "arrival_ms must be monotonically non-decreasing")
        previous_arrival = arrival
        length = _require_positive_int(item, "packet_length", event_id)
        events.append(
            {
                "event_seq": index,
                "event_id": event_id,
                "pipeline_id": pipeline_id,
                "arrival_ms": arrival,
                "packet_length": length,
            }
        )
    if not events:
        raise SpecError("at least one event is required", object_id="$events", field="events")
    return events


def load_bundle(spec_dir: str | Path) -> dict[str, Any]:
    paths = Paths(Path(spec_dir))
    pipelines_raw = load_json_file(paths.pipelines)
    pipelines = validate_pipelines(pipelines_raw)
    shapers_raw = load_json_file(paths.shapers)
    shapers = validate_shapers(shapers_raw, pipelines)
    events_raw = load_json_file(paths.events)
    events = validate_events(events_raw, pipelines)
    config_raw = load_json_file(paths.config) if paths.config.exists() else {}
    config = validate_config(config_raw)
    return {
        "spec_dir": str(paths.root),
        "config": config,
        "pipelines": sorted(pipelines, key=lambda item: item["pipeline_id"]),
        "shapers": shapers,
        "events": events,
        "spec_digest": None,
    }


def bundle_digest(bundle: dict[str, Any]) -> str:
    from .canonical import sha256_hex

    return sha256_hex(
        {
            "pipelines": bundle["pipelines"],
            "shapers": bundle["shapers"],
            "events": [{k: v for k, v in event.items() if k != "event_seq"} for event in bundle["events"]],
            "config": bundle["config"],
        }
    )


def shaper_index(bundle: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(item["pipeline_id"], item["stage"]): item for item in bundle["shapers"]}
