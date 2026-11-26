from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .errors import SpecError


FORMAT_VERSION = 1
SUPPORTED_BLOCK_SIZE_MIN = 64
SUPPORTED_BLOCK_SIZE_MAX = 1 << 20
SUPPORTED_BLOCK_SIZES = tuple(
    1 << exponent
    for exponent in range(6, 21)
)
EVENT_ALIASES = {
    "write": "write",
    "snapshot_create": "snapshot_create",
    "create_snapshot": "snapshot_create",
    "snapshot_created": "snapshot_create",
    "snapshot_delete": "snapshot_delete",
    "delete_snapshot": "snapshot_delete",
    "snapshot_deleted": "snapshot_delete",
    "rollback": "rollback",
    "rollback_start": "rollback",
}


def _pick(obj: dict[str, Any], *names: str, default: Any = None, required: bool = False) -> Any:
    for name in names:
        if name in obj and obj[name] is not None:
            return obj[name]
    if required:
        raise SpecError("missing required field", entry_id=_entry_id(obj), field=names[0])
    return default


def _entry_id(obj: Any) -> str | None:
    if not isinstance(obj, dict):
        return None
    value = obj.get("id") or obj.get("volume_id") or obj.get("snapshot_id") or obj.get("rollback_id")
    if value is None and "seq" in obj:
        value = f"event:{obj['seq']}"
    elif value is None and "sequence" in obj:
        value = f"event:{obj['sequence']}"
    return str(value) if value is not None else None


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_int(
    obj: dict[str, Any],
    names: tuple[str, ...],
    *,
    positive: bool = False,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    field = names[0]
    value = _pick(obj, *names, required=True)
    entry_id = _entry_id(obj)
    if not _is_int(value):
        raise SpecError("field must be an integer", entry_id=entry_id, field=field)
    if positive and value <= 0:
        raise SpecError("field must be a positive integer", entry_id=entry_id, field=field)
    if minimum is not None and value < minimum:
        raise SpecError(f"field must be >= {minimum}", entry_id=entry_id, field=field)
    if maximum is not None and value > maximum:
        raise SpecError(f"field must be <= {maximum}", entry_id=entry_id, field=field)
    return value


def _require_str(obj: dict[str, Any], names: tuple[str, ...]) -> str:
    field = names[0]
    value = _pick(obj, *names, required=True)
    if not isinstance(value, str) or not value:
        raise SpecError("field must be a non-empty string", entry_id=_entry_id(obj), field=field)
    return value


def normalize_intervals(
    raw_intervals: Any,
    block_count: int,
    *,
    entry_id: str | None = None,
    field: str = "intervals",
) -> list[list[int]]:
    if not isinstance(raw_intervals, list) or not raw_intervals:
        raise SpecError("must be a non-empty list of half-open intervals", entry_id=entry_id, field=field)

    intervals: list[tuple[int, int]] = []
    for item in raw_intervals:
        if isinstance(item, list) and len(item) == 2:
            start_raw, end_raw = item
        elif isinstance(item, dict):
            start_raw = item.get("start", item.get("first", item.get("start_block")))
            end_raw = item.get("end", item.get("end_exclusive", item.get("last_exclusive")))
            if end_raw is None and "last" in item and _is_int(item["last"]):
                end_raw = item["last"] + 1
        else:
            raise SpecError("interval must be [start,end) or an object", entry_id=entry_id, field=field)

        if not _is_int(start_raw) or not _is_int(end_raw):
            raise SpecError("interval endpoints must be integers", entry_id=entry_id, field=field)
        if start_raw < 0 or end_raw > block_count or start_raw >= end_raw:
            raise SpecError(
                f"interval [{start_raw},{end_raw}) is outside 0..{block_count} or empty",
                entry_id=entry_id,
                field=field,
            )
        intervals.append((start_raw, end_raw))

    intervals.sort()
    normalized: list[list[int]] = []
    for start, end in intervals:
        if normalized and start < normalized[-1][1]:
            raise SpecError("intervals must not overlap", entry_id=entry_id, field=field)
        if normalized and start == normalized[-1][1]:
            normalized[-1][1] = end
        else:
            normalized.append([start, end])
    return normalized


def expand_intervals(intervals: Iterable[tuple[int, int] | list[int]]) -> list[int]:
    blocks: list[int] = []
    for start, end in intervals:
        blocks.extend(range(int(start), int(end)))
    return blocks


def _read_json_file(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SpecError(f"spec file not found: {path}", field="path") from exc
    except json.JSONDecodeError as exc:
        raise SpecError(
            f"invalid JSON in {path.name}: {exc.msg}",
            entry_id=path.name,
            field=f"json:{exc.lineno}:{exc.colno}",
        ) from exc


def load_json_path(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if source.is_dir():
        return load_directory(source)
    raw = _read_json_file(source)
    if not isinstance(raw, dict):
        raise SpecError("top-level JSON must be an object", entry_id=source.name, field="root")
    return normalize_spec(raw)


def load_directory(directory: Path) -> dict[str, Any]:
    candidate = directory / "spec.json"
    if candidate.exists():
        return load_json_path(candidate)

    def section(name: str) -> Any:
        path = directory / name
        if not path.exists():
            return None
        raw = _read_json_file(path)
        return raw.get(name.split(".")[0]) if isinstance(raw, dict) else raw

    volumes = section("volumes.json")
    snapshots = section("snapshots.json")
    events = section("events.json")
    budget = section("budget.json")
    raw: dict[str, Any] = {}
    if volumes is not None:
        raw["volumes"] = volumes
    if snapshots is not None:
        raw["snapshots"] = snapshots
    if events is not None:
        raw["events"] = events
    if budget is not None:
        raw["budget"] = budget
    return normalize_spec(raw)


def normalize_spec(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise SpecError("spec must be a JSON object", field="root")

    format_version = raw.get("format_version", FORMAT_VERSION)
    if not _is_int(format_version):
        raise SpecError("format_version must be an integer", field="format_version")
    if format_version != FORMAT_VERSION:
        raise SpecError(
            f"unsupported format_version {format_version}; expected {FORMAT_VERSION}",
            field="format_version",
        )

    volumes_raw = raw.get("volumes")
    if not isinstance(volumes_raw, list) or not volumes_raw:
        raise SpecError("volumes must be a non-empty list", field="volumes")

    volumes: list[dict[str, Any]] = []
    seen_volume_ids: set[str] = set()
    for item in volumes_raw:
        if not isinstance(item, dict):
            raise SpecError("volume entry must be an object", field="volumes[]")
        volume_id = _require_str(item, ("id", "volume_id"))
        if volume_id in seen_volume_ids:
            raise SpecError("duplicate volume identifier", entry_id=volume_id, field="id")
        seen_volume_ids.add(volume_id)

        block_size = _require_int(item, ("block_size",), positive=True)
        if (
            block_size < SUPPORTED_BLOCK_SIZE_MIN
            or block_size > SUPPORTED_BLOCK_SIZE_MAX
            or (block_size & (block_size - 1)) != 0
        ):
            raise SpecError(
                "block_size must be a power of two between "
                f"{SUPPORTED_BLOCK_SIZE_MIN} and {SUPPORTED_BLOCK_SIZE_MAX}",
                entry_id=volume_id,
                field="block_size",
            )
        block_count = _require_int(item, ("block_count",), positive=True)
        seed = _require_int(item, ("seed", "initial_seed"))
        baseline_version = _require_int(
            item, ("baseline_version", "baseline"), positive=True
        )
        volumes.append(
            {
                "id": volume_id,
                "block_size": block_size,
                "block_count": block_count,
                "seed": seed,
                "baseline_version": baseline_version,
            }
        )

    volume_by_id = {volume["id"]: volume for volume in volumes}

    snapshots_raw = raw.get("snapshots", [])
    if not isinstance(snapshots_raw, list):
        raise SpecError("snapshots must be a list", field="snapshots")

    snapshots: list[dict[str, Any]] = []
    seen_snapshot_ids: set[str] = set()
    snapshot_versions_by_volume: dict[str, set[int]] = {vid: set() for vid in volume_by_id}
    for item in snapshots_raw:
        if not isinstance(item, dict):
            raise SpecError("snapshot entry must be an object", field="snapshots[]")
        snapshot_id = _require_str(item, ("id", "snapshot_id"))
        if snapshot_id in seen_snapshot_ids:
            raise SpecError("duplicate snapshot identifier", entry_id=snapshot_id, field="id")
        seen_snapshot_ids.add(snapshot_id)

        volume_id = _require_str(item, ("volume_id", "volume", "volume_ref"))
        if volume_id not in volume_by_id:
            raise SpecError(
                "snapshot references a non-existent volume",
                entry_id=snapshot_id,
                field="volume_id",
            )
        version = _require_int(item, ("version", "snapshot_version"), positive=True)
        if version in snapshot_versions_by_volume[volume_id]:
            raise SpecError(
                "snapshot version is duplicated for volume",
                entry_id=snapshot_id,
                field="version",
            )
        snapshot_versions_by_volume[volume_id].add(version)

        volume = volume_by_id[volume_id]
        intervals = normalize_intervals(
            item.get("intervals", item.get("ranges", item.get("block_intervals"))),
            volume["block_count"],
            entry_id=snapshot_id,
        )
        blocks = expand_intervals(intervals)
        digests_raw = item.get("digests", item.get("content_digests", {}))
        if not isinstance(digests_raw, dict):
            raise SpecError("digests must be an object keyed by block index", entry_id=snapshot_id, field="digests")
        digests: dict[int, str] = {}
        for key, value in digests_raw.items():
            try:
                block_index = int(key)
            except (TypeError, ValueError) as exc:
                raise SpecError("digest key must be a block index", entry_id=snapshot_id, field="digests") from exc
            if block_index not in blocks:
                raise SpecError(
                    "digest references block outside captured intervals",
                    entry_id=snapshot_id,
                    field=f"digests.{key}",
                )
            if not isinstance(value, str) or len(value) != 64:
                raise SpecError(
                    "digest must be a 64-character SHA-256 hex string",
                    entry_id=snapshot_id,
                    field=f"digests.{key}",
                )
            try:
                int(value, 16)
            except ValueError as exc:
                raise SpecError(
                    "digest must be lowercase hexadecimal",
                    entry_id=snapshot_id,
                    field=f"digests.{key}",
                ) from exc
            digests[block_index] = value.lower()
        missing = sorted(set(blocks) - set(digests))
        if missing:
            raise SpecError(
                f"missing digest for block {missing[0]}",
                entry_id=snapshot_id,
                field=f"digests.{missing[0]}",
            )

        snapshots.append(
            {
                "id": snapshot_id,
                "volume_id": volume_id,
                "version": version,
                "intervals": intervals,
                "blocks": blocks,
                "digests": {str(index): digests[index] for index in sorted(digests)},
            }
        )

    snapshot_by_id = {snapshot["id"]: snapshot for snapshot in snapshots}

    events_raw = raw.get("events", [])
    if not isinstance(events_raw, list):
        raise SpecError("events must be a list", field="events")

    events: list[dict[str, Any]] = []
    last_seq: int | None = None
    seen_event_ids: set[str] = set()
    created_snapshots: set[str] = set()
    deleted_snapshots: set[str] = set()
    created_versions_by_volume: dict[str, list[int]] = {vid: [] for vid in volume_by_id}
    seen_rollback_ids: set[str] = set()

    for position, item in enumerate(events_raw):
        if not isinstance(item, dict):
            raise SpecError("event must be an object", entry_id=f"events[{position}]", field="events[]")
        seq = _require_int(item, ("seq", "sequence", "arrival_seq"), minimum=0)
        entry_id = _entry_id(item) or f"event:{seq}"
        if last_seq is not None and seq <= last_seq:
            raise SpecError(
                "event seq must be strictly increasing and unique",
                entry_id=entry_id,
                field="seq",
            )
        last_seq = seq

        event_id = item.get("event_id", item.get("id"))
        if event_id is not None:
            event_id = str(event_id)
            if event_id in seen_event_ids:
                raise SpecError("duplicate event identifier", entry_id=event_id, field="event_id")
            seen_event_ids.add(event_id)

        raw_type = _require_str(item, ("type", "event_type", "kind"))
        event_type = EVENT_ALIASES.get(raw_type)
        if event_type is None:
            raise SpecError(f"unsupported event type {raw_type!r}", entry_id=entry_id, field="type")

        normalized: dict[str, Any] = {"seq": seq, "type": event_type}
        if event_id is not None:
            normalized["event_id"] = event_id

        if event_type == "write":
            volume_id = _require_str(item, ("volume_id", "volume", "volume_ref"))
            if volume_id not in volume_by_id:
                raise SpecError("write references non-existent volume", entry_id=entry_id, field="volume_id")
            volume = volume_by_id[volume_id]
            intervals = normalize_intervals(
                item.get("intervals", item.get("ranges", item.get("block_intervals"))),
                volume["block_count"],
                entry_id=entry_id,
            )
            seed = _require_int(item, ("seed", "content_seed", "write_seed"))
            normalized.update({"volume_id": volume_id, "intervals": intervals, "seed": seed})

        elif event_type == "snapshot_create":
            snapshot_id = _require_str(item, ("snapshot_id", "snapshot", "ref", "id"))
            if snapshot_id not in snapshot_by_id:
                raise SpecError("create event references undeclared snapshot", entry_id=entry_id, field="snapshot_id")
            if snapshot_id in created_snapshots:
                raise SpecError("snapshot was already created", entry_id=snapshot_id, field="snapshot_id")
            snapshot = snapshot_by_id[snapshot_id]
            volume_id = snapshot["volume_id"]
            prior_versions = created_versions_by_volume[volume_id]
            if prior_versions and snapshot["version"] <= prior_versions[-1]:
                raise SpecError(
                    "snapshot create versions for a volume must be strictly increasing in arrival order",
                    entry_id=snapshot_id,
                    field="version",
                )
            prior_versions.append(snapshot["version"])
            created_snapshots.add(snapshot_id)
            normalized["snapshot_id"] = snapshot_id

        elif event_type == "snapshot_delete":
            snapshot_id = _require_str(item, ("snapshot_id", "snapshot", "ref", "id"))
            if snapshot_id not in snapshot_by_id:
                raise SpecError("delete event references undeclared snapshot", entry_id=entry_id, field="snapshot_id")
            if snapshot_id not in created_snapshots:
                raise SpecError("snapshot cannot be deleted before its create event", entry_id=snapshot_id, field="seq")
            if snapshot_id in deleted_snapshots:
                raise SpecError("snapshot is already deleted", entry_id=snapshot_id, field="snapshot_id")
            deleted_snapshots.add(snapshot_id)
            normalized["snapshot_id"] = snapshot_id

        else:
            snapshot_id = _require_str(item, ("snapshot_id", "snapshot", "ref"))
            if snapshot_id not in snapshot_by_id:
                raise SpecError("rollback references undeclared snapshot", entry_id=entry_id, field="snapshot_id")
            if snapshot_id not in created_snapshots or snapshot_id in deleted_snapshots:
                raise SpecError(
                    "rollback target snapshot must exist and be active at event arrival",
                    entry_id=_entry_id(item),
                    field="snapshot_id",
                )
            rollback_id = item.get("rollback_id", item.get("id", f"rb-{seq}"))
            rollback_id = str(rollback_id)
            if rollback_id in seen_rollback_ids:
                raise SpecError("duplicate rollback identifier", entry_id=rollback_id, field="rollback_id")
            seen_rollback_ids.add(rollback_id)
            normalized.update({"snapshot_id": snapshot_id, "rollback_id": rollback_id})
            if "max_blocks" in item or "max_blocks_per_rollback" in item:
                max_blocks = _require_int(item, ("max_blocks", "max_blocks_per_rollback"), positive=True)
                normalized["max_blocks"] = max_blocks

        events.append(normalized)

    for snapshot in snapshots:
        if snapshot["id"] not in created_snapshots:
            raise SpecError(
                "declared snapshot is missing its snapshot_create event",
                entry_id=snapshot["id"],
                field="events",
            )

    budget_raw = raw.get("budget", {})
    if budget_raw is None:
        budget_raw = {}
    if not isinstance(budget_raw, dict):
        raise SpecError("budget must be an object", entry_id="budget", field="budget")

    def budget_value(*names: str, default: int | None = None) -> int | None:
        for name in names:
            if name in budget_raw:
                value = budget_raw[name]
                if not _is_int(value):
                    raise SpecError("budget value must be an integer", entry_id="budget", field=name)
                return value
        return default

    per_volume = budget_value("per_volume_rollback_concurrency", "per_volume_concurrency", default=1)
    global_concurrency = budget_value("global_rollback_concurrency", "global_concurrency", default=2)
    max_blocks = budget_value("max_blocks_per_rollback", "max_blocks", default=4)
    permille = budget_value("verification_sample_permille", "sample_permille")
    if permille is None:
        percent = budget_value("verification_sample_percent", "sample_percent", default=100)
        if percent is None or percent < 0 or percent > 100:
            raise SpecError("sample percent must be 0..100", entry_id="budget", field="verification_sample_percent")
        permille = percent * 10
    for name, value, low, high in (
        ("per_volume_rollback_concurrency", per_volume, 1, 64),
        ("global_rollback_concurrency", global_concurrency, 1, 128),
        ("max_blocks_per_rollback", max_blocks, 1, 1_000_000),
        ("verification_sample_permille", permille, 0, 1000),
    ):
        if value is None or value < low or value > high:
            raise SpecError(f"budget {name} out of range {low}..{high}", entry_id="budget", field=name)
    if global_concurrency < per_volume:
        raise SpecError(
            "global_rollback_concurrency must be >= per_volume_rollback_concurrency",
            entry_id="budget",
            field="global_rollback_concurrency",
        )

    timeout_ms = budget_value("verification_timeout_ms", "worker_timeout_ms", default=1000)
    if timeout_ms is None or timeout_ms < 1 or timeout_ms > 600_000:
        raise SpecError("verification_timeout_ms out of range", entry_id="budget", field="verification_timeout_ms")

    budget = {
        "per_volume_rollback_concurrency": per_volume,
        "global_rollback_concurrency": global_concurrency,
        "max_blocks_per_rollback": max_blocks,
        "verification_sample_permille": permille,
        "verification_timeout_ms": timeout_ms,
    }

    return {"format_version": FORMAT_VERSION, "volumes": volumes, "snapshots": snapshots, "events": events, "budget": budget}


def volume_summary(spec: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for volume in sorted(spec["volumes"], key=lambda item: item["id"]):
        snapshots = sorted(
            (snapshot["id"] for snapshot in spec["snapshots"] if snapshot["volume_id"] == volume["id"])
        )
        baseline_digest = __import__("blockledger.canonical", fromlist=["sha256_hex"]).sha256_hex(
            {
                "volume_id": volume["id"],
                "baseline_version": volume["baseline_version"],
                "seed": volume["seed"],
                "block_count": volume["block_count"],
                "block_size": volume["block_size"],
            }
        )
        result.append(
            {
                "volume_id": volume["id"],
                "block_count": volume["block_count"],
                "block_size": volume["block_size"],
                "snapshot_refs": snapshots,
                "snapshot_count": len(snapshots),
                "baseline_version": volume["baseline_version"],
                "baseline_digest": baseline_digest,
            }
        )
    return result
