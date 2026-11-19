"""Session parsing and strict deterministic input validation."""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

from .canon import (
    as_fraction,
    as_int,
    canonical_json,
    fraction_str,
    load_json_file,
    positive_fraction,
    sha256_canonical,
)
from .errors import BusinessError


@dataclass(frozen=True)
class Representation:
    bitrate_bps: Fraction
    size_bytes: Fraction
    source_index: int

    @property
    def digest(self) -> str:
        return sha256_canonical(
            {
                "bitrate_bps": fraction_str(self.bitrate_bps),
                "size_bytes": fraction_str(self.size_bytes),
                "source_index": self.source_index,
            }
        )


@dataclass(frozen=True)
class Segment:
    index: int
    duration_ms: int
    media_start_ms: int
    media_end_ms: int
    representations: tuple[Representation, ...]

    @property
    def fragment_digest(self) -> str:
        return sha256_canonical(
            {
                "duration_ms": self.duration_ms,
                "media_start_ms": self.media_start_ms,
                "media_end_ms": self.media_end_ms,
                "representations": [
                    {
                        "bitrate_bps": fraction_str(rep.bitrate_bps),
                        "size_bytes": fraction_str(rep.size_bytes),
                    }
                    for rep in self.representations
                ],
            }
        )


@dataclass(frozen=True)
class MediaManifest:
    fragments: tuple[Segment, ...]
    total_duration_ms: int
    digest: str
    bitrate_ladder: tuple[Fraction, ...]


@dataclass(frozen=True)
class TraceSegment:
    index: int
    bandwidth_bps: Fraction
    rtt_ms: int


@dataclass(frozen=True)
class NetworkTrace:
    segments: tuple[TraceSegment, ...]
    digest: str


@dataclass(frozen=True)
class SimulationEvent:
    at_ms: int
    action: str
    order: int
    source_index: int
    target_ms: int | None = None
    segment_index: int | None = None


@dataclass(frozen=True)
class EventScript:
    events: tuple[SimulationEvent, ...]
    digest: str


@dataclass(frozen=True)
class ClientConfig:
    startup_buffer_ms: int
    max_buffer_ms: int
    safety_margin_ms: int
    strategy: str
    switch_penalty: Fraction
    concurrency: int
    memory_budget_bytes: int
    throughput_safety_ratio: Fraction


@dataclass(frozen=True)
class Session:
    session_path: Path
    network_path: Path
    media_path: Path
    event_path: Path
    state_dir: Path
    snapshot_dir: Path
    client: ClientConfig
    media: MediaManifest
    network: NetworkTrace
    events: EventScript
    version_id: str
    version_payload: dict

    def segment_at_or_after(self, position_ms: int) -> int:
        if position_ms >= self.media.total_duration_ms:
            return len(self.media.fragments)
        starts = [fragment.media_start_ms for fragment in self.media.fragments]
        index = bisect.bisect_right(starts, position_ms) - 1
        return max(0, index)


def _normalize_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize_json_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, float, Decimal)):
        return fraction_str(as_fraction(value, "session"))
    return str(value)


def _read_required_path(raw: dict, keys: tuple[str, ...], base: Path, label: str) -> Path:
    value = None
    for key in keys:
        if key in raw:
            value = raw[key]
            break
    if not isinstance(value, str) or not value.strip():
        raise BusinessError("E_INPUT", f"{label} path is required", label)
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _parse_client(raw: Any) -> tuple[ClientConfig, dict]:
    if not isinstance(raw, dict):
        raise BusinessError("E_INPUT", "client must be an object", "client")
    startup = as_int(raw.get("startup_buffer_ms"), "client.startup_buffer_ms", minimum=0)
    maximum = as_int(raw.get("max_buffer_ms"), "client.max_buffer_ms", minimum=1)
    safety = as_int(raw.get("safety_margin_ms"), "client.safety_margin_ms", minimum=0)
    if startup > maximum:
        raise BusinessError(
            "E_INPUT",
            "startup_buffer_ms must not exceed max_buffer_ms",
            "client.startup_buffer_ms",
        )
    strategy = raw.get("strategy")
    if strategy not in {"hybrid", "throughput", "buffer"}:
        raise BusinessError(
            "E_INPUT",
            "strategy must be one of: hybrid, throughput, buffer",
            "client.strategy",
        )
    penalty = as_fraction(raw.get("switch_penalty"), "client.switch_penalty")
    if penalty < 0:
        raise BusinessError("E_INPUT", "switch_penalty must be non-negative", "client.switch_penalty")
    concurrency = as_int(raw.get("concurrency"), "client.concurrency", minimum=1)
    memory_budget = as_int(
        raw.get("memory_budget_bytes"), "client.memory_budget_bytes", minimum=1
    )
    ratio_raw = raw.get("throughput_safety_ratio", Decimal("0.90"))
    ratio = as_fraction(ratio_raw, "client.throughput_safety_ratio")
    if ratio <= 0 or ratio > 1:
        raise BusinessError(
            "E_INPUT",
            "throughput_safety_ratio must be in (0, 1]",
            "client.throughput_safety_ratio",
        )
    config = ClientConfig(
        startup_buffer_ms=startup,
        max_buffer_ms=maximum,
        safety_margin_ms=safety,
        strategy=strategy,
        switch_penalty=penalty,
        concurrency=concurrency,
        memory_budget_bytes=memory_budget,
        throughput_safety_ratio=ratio,
    )
    normalized = _normalize_json_value(raw)
    normalized.update(
        {
            "startup_buffer_ms": startup,
            "max_buffer_ms": maximum,
            "safety_margin_ms": safety,
            "strategy": strategy,
            "switch_penalty": fraction_str(penalty),
            "concurrency": concurrency,
            "memory_budget_bytes": memory_budget,
            "throughput_safety_ratio": fraction_str(ratio),
        }
    )
    return config, normalized


def _parse_media(raw: Any) -> MediaManifest:
    if not isinstance(raw, dict) or not isinstance(raw.get("fragments"), list):
        raise BusinessError("E_MEDIA", "media manifest must contain fragments[]", "media.fragments")
    if not raw["fragments"]:
        raise BusinessError("E_MEDIA", "media manifest must contain at least one fragment", "media.fragments")
    fragments: list[Segment] = []
    cursor = 0
    all_rates: set[Fraction] = set()
    normalized_fragments: list[dict] = []
    for index, item in enumerate(raw["fragments"]):
        path = f"media.fragments[{index}]"
        if not isinstance(item, dict):
            raise BusinessError("E_MEDIA", "fragment must be an object", path)
        duration = as_int(item.get("duration_ms"), f"{path}.duration_ms", minimum=1)
        reps_raw = item.get("representations", item.get("variants"))
        if reps_raw is None and "bitrate_bps" in item and "size_bytes" in item:
            reps_raw = [
                {
                    "bitrate_bps": item["bitrate_bps"],
                    "size_bytes": item["size_bytes"],
                }
            ]
        if not isinstance(reps_raw, list) or not reps_raw:
            raise BusinessError(
                "E_MEDIA",
                "fragment must contain a non-empty representations[] list",
                f"{path}.representations",
            )
        reps: list[Representation] = []
        for rep_index, rep_item in enumerate(reps_raw):
            rep_path = f"{path}.representations[{rep_index}]"
            if not isinstance(rep_item, dict):
                raise BusinessError("E_MEDIA", "representation must be an object", rep_path)
            bitrate = positive_fraction(rep_item.get("bitrate_bps"), f"{rep_path}.bitrate_bps")
            size = positive_fraction(rep_item.get("size_bytes"), f"{rep_path}.size_bytes")
            reps.append(Representation(bitrate, size, rep_index))
            all_rates.add(bitrate)
        reps.sort(key=lambda rep: (rep.bitrate_bps, rep.size_bytes, rep.source_index))
        segment = Segment(
            index=index,
            duration_ms=duration,
            media_start_ms=cursor,
            media_end_ms=cursor + duration,
            representations=tuple(reps),
        )
        fragments.append(segment)
        cursor += duration
        normalized_fragments.append(
            {
                "duration_ms": duration,
                "representations": [
                    {
                        "bitrate_bps": fraction_str(rep.bitrate_bps),
                        "size_bytes": fraction_str(rep.size_bytes),
                    }
                    for rep in reps
                ],
            }
        )
    payload = {"fragments": normalized_fragments}
    return MediaManifest(
        fragments=tuple(fragments),
        total_duration_ms=cursor,
        digest=sha256_canonical(payload),
        bitrate_ladder=tuple(sorted(all_rates)),
    )


def _parse_network(raw: Any) -> NetworkTrace:
    if not isinstance(raw, dict) or not isinstance(raw.get("segments"), list):
        raise BusinessError("E_NETWORK", "network trace must contain segments[]", "network.segments")
    if not raw["segments"]:
        raise BusinessError("E_NETWORK", "network trace must contain at least one segment", "network.segments")
    segments: list[TraceSegment] = []
    normalized: list[dict] = []
    for index, item in enumerate(raw["segments"]):
        path = f"network.segments[{index}]"
        if not isinstance(item, dict):
            raise BusinessError("E_NETWORK", "trace segment must be an object", path)
        bandwidth = positive_fraction(item.get("bandwidth_bps"), f"{path}.bandwidth_bps")
        rtt = as_int(item.get("rtt_ms"), f"{path}.rtt_ms", minimum=0)
        segments.append(TraceSegment(index, bandwidth, rtt))
        normalized.append(
            {"bandwidth_bps": fraction_str(bandwidth), "rtt_ms": rtt}
        )
    payload = {"segments": normalized}
    return NetworkTrace(tuple(segments), sha256_canonical(payload))


def _parse_events(raw: Any, media: MediaManifest, network: NetworkTrace) -> EventScript:
    if not isinstance(raw, dict) or not isinstance(raw.get("events"), list):
        raise BusinessError("E_EVENT", "event script must contain events[]", "events")
    parsed: list[SimulationEvent] = []
    normalized: list[dict] = []
    for source_index, item in enumerate(raw["events"]):
        path = f"events[{source_index}]"
        if not isinstance(item, dict):
            raise BusinessError("E_EVENT", "event must be an object", path)
        at_ms = as_int(item.get("at_ms"), f"{path}.at_ms", minimum=0)
        action_raw = item.get("action")
        action_aliases = {
            "pause": "pause",
            "resume": "resume",
            "jump": "jump",
            "seek": "jump",
            "trace_switch": "trace_switch",
            "network_switch": "trace_switch",
            "switch_trace": "trace_switch",
            "terminate": "terminate",
            "signal": "terminate",
            "interrupt": "terminate",
        }
        if not isinstance(action_raw, str) or action_raw not in action_aliases:
            raise BusinessError(
                "E_EVENT",
                "unsupported event action",
                f"{path}.action",
            )
        action = action_aliases[action_raw]
        target_ms: int | None = None
        segment_index: int | None = None
        if action == "jump":
            target_value = item.get("target_ms", item.get("position_ms"))
            target_ms = as_int(target_value, f"{path}.target_ms", minimum=0)
            if target_ms > media.total_duration_ms:
                raise BusinessError(
                    "E_EVENT",
                    "jump target_ms exceeds media duration",
                    f"{path}.target_ms",
                )
        elif action == "trace_switch":
            segment_value = item.get("segment_index", item.get("trace_segment"))
            segment_index = as_int(segment_value, f"{path}.segment_index", minimum=0)
            if segment_index >= len(network.segments):
                raise BusinessError(
                    "E_EVENT",
                    "trace segment index is out of range",
                    f"{path}.segment_index",
                )
        order = as_int(item.get("order", source_index), f"{path}.order", minimum=0)
        parsed.append(
            SimulationEvent(
                at_ms=at_ms,
                action=action,
                order=order,
                source_index=source_index,
                target_ms=target_ms,
                segment_index=segment_index,
            )
        )
        entry = {"at_ms": at_ms, "action": action, "order": order}
        if target_ms is not None:
            entry["target_ms"] = target_ms
        if segment_index is not None:
            entry["segment_index"] = segment_index
        normalized.append(entry)
    parsed.sort(key=lambda event: (event.at_ms, event.order, event.source_index))
    normalized.sort(key=lambda event: (event["at_ms"], event["order"]))
    return EventScript(tuple(parsed), sha256_canonical(normalized))


def load_session(session_path: str | Path) -> Session:
    resolved = Path(session_path).resolve()
    raw = load_json_file(resolved)
    if not isinstance(raw, dict):
        raise BusinessError("E_INPUT", "session description must be a JSON object", "session")
    base = resolved.parent
    network_path = _read_required_path(
        raw, ("network_trace", "network_trace_file", "network"), base, "network_trace"
    )
    media_path = _read_required_path(
        raw, ("media_manifest", "media_manifest_file", "media"), base, "media_manifest"
    )
    event_path = _read_required_path(
        raw, ("event_script", "event_script_file", "events"), base, "event_script"
    )
    state_dir = _read_required_path(
        raw, ("state_dir", "status_dir"), base, "state_dir"
    )
    snapshot_dir = _read_required_path(
        raw, ("snapshot_dir", "snapshot_archive_dir"), base, "snapshot_dir"
    )

    client, normalized_client = _parse_client(raw.get("client"))
    media = _parse_media(load_json_file(media_path, code="E_MEDIA"))
    network = _parse_network(load_json_file(network_path, code="E_NETWORK"))
    events = _parse_events(load_json_file(event_path, code="E_EVENT"), media, network)

    version_payload = {
        "schema": "streamsim-v1",
        "media": {
            "fragments": [
                {
                    "duration_ms": fragment.duration_ms,
                    "representations": [
                        {
                            "bitrate_bps": fraction_str(rep.bitrate_bps),
                            "size_bytes": fraction_str(rep.size_bytes),
                        }
                        for rep in fragment.representations
                    ],
                }
                for fragment in media.fragments
            ]
        },
        "network": {
            "segments": [
                {"bandwidth_bps": fraction_str(segment.bandwidth_bps), "rtt_ms": segment.rtt_ms}
                for segment in network.segments
            ]
        },
        "client": normalized_client,
    }
    version_id = sha256_canonical(version_payload)
    return Session(
        session_path=resolved,
        network_path=network_path,
        media_path=media_path,
        event_path=event_path,
        state_dir=state_dir,
        snapshot_dir=snapshot_dir,
        client=client,
        media=media,
        network=network,
        events=events,
        version_id=version_id,
        version_payload=version_payload,
    )
