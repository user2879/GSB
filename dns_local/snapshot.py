"""Immutable, named session snapshots with checksums and whole-state restore."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .cache import CacheEntry, CacheKey
from .clock import Clock
from .errors import BusinessError
from .names import normalize_name
from .wire import CLASS_IN, SOAData, TYPE_BY_NAME, ResourceRecord
from .zone import ZoneImage, image_from_snapshot


SNAPSHOT_FORMAT = "dns-local-snapshot/1"
_SNAPSHOT_ID_RE = re.compile(r"^(?P<revision>\d{4})-(?P<slug>[a-z0-9_-]+)$")


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _checksum(data: Mapping[str, Any]) -> str:
    body = {key: value for key, value in data.items() if key != "checksum"}
    return hashlib.sha256(_canonical_json(body)).hexdigest()


def rr_to_json(record: ResourceRecord) -> dict[str, Any]:
    if isinstance(record.rdata, SOAData):
        rdata: object = {
            "mname": record.rdata.mname,
            "rname": record.rdata.rname,
            "serial": record.rdata.serial,
            "refresh": record.rdata.refresh,
            "retry": record.rdata.retry,
            "expire": record.rdata.expire,
            "minimum": record.rdata.minimum,
        }
    elif isinstance(record.rdata, bytes):
        rdata = {"hex": record.rdata.hex()}
    else:
        rdata = record.rdata
    return {
        "name": record.name,
        "type": record.type_name,
        "class": record.rclass,
        "ttl": record.ttl,
        "rdata": rdata,
    }


def rr_from_json(value: object) -> ResourceRecord:
    if not isinstance(value, dict):
        raise BusinessError("E_SNAPSHOT", "snapshot RR must be an object", field="rdata")
    try:
        name = normalize_name(value["name"], allow_wildcard=True)
        type_name = str(value["type"]).upper()
        rclass = int(value["class"])
        ttl = int(value["ttl"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BusinessError("E_SNAPSHOT", f"invalid snapshot RR: {exc}", field="rdata") from exc
    if type_name not in TYPE_BY_NAME:
        raise BusinessError("E_SNAPSHOT", f"unsupported snapshot RR type: {type_name}", field="type")
    if ttl < 0 or ttl > 0x7FFFFFFF:
        raise BusinessError("E_SNAPSHOT", "snapshot RR TTL is outside the DNS range", field="ttl")
    raw = value.get("rdata")
    if type_name == "A":
        try:
            rdata: str | SOAData | bytes = str(ipaddress.IPv4Address(str(raw)))
        except ipaddress.AddressValueError as exc:
            raise BusinessError("E_SNAPSHOT", "invalid A record in snapshot", field="rdata") from exc
    elif type_name == "AAAA":
        try:
            rdata = str(ipaddress.IPv6Address(str(raw)))
        except ipaddress.AddressValueError as exc:
            raise BusinessError("E_SNAPSHOT", "invalid AAAA record in snapshot", field="rdata") from exc
    elif type_name in ("CNAME", "NS"):
        try:
            rdata = normalize_name(str(raw))
        except ValueError as exc:
            raise BusinessError("E_SNAPSHOT", "invalid name RDATA in snapshot", field="rdata") from exc
    elif type_name == "SOA":
        if not isinstance(raw, dict):
            raise BusinessError("E_SNAPSHOT", "SOA RDATA must be an object", field="rdata")
        try:
            soa_values = [
                normalize_name(str(raw["mname"])),
                normalize_name(str(raw["rname"])),
                int(raw["serial"]),
                int(raw["refresh"]),
                int(raw["retry"]),
                int(raw["expire"]),
                int(raw["minimum"]),
            ]
            if any(not 0 <= number <= 0xFFFFFFFF for number in soa_values[2:]):
                raise ValueError("SOA numeric field outside 32-bit range")
            rdata = SOAData(*soa_values)
        except (KeyError, TypeError, ValueError) as exc:
            raise BusinessError("E_SNAPSHOT", f"invalid SOA in snapshot: {exc}", field="rdata") from exc
    else:
        if not isinstance(raw, dict) or not isinstance(raw.get("hex"), str):
            raise BusinessError("E_SNAPSHOT", "raw RDATA must contain a hex field", field="rdata")
        try:
            rdata = bytes.fromhex(raw["hex"])
        except ValueError as exc:
            raise BusinessError("E_SNAPSHOT", "invalid raw RDATA hex", field="rdata") from exc
    return ResourceRecord(name, TYPE_BY_NAME[type_name], rclass, ttl, rdata)


def zone_to_json(zone: ZoneImage) -> dict[str, Any]:
    return {
        "origin": zone.origin,
        "version": zone.version,
        "file_digest": zone.digest,
        "content_digest": zone.content_digest,
        "records": [rr_to_json(record) for record in zone.records],
    }


def zone_from_json(value: object, *, max_ttl: int) -> ZoneImage:
    if not isinstance(value, dict):
        raise BusinessError("E_SNAPSHOT", "snapshot zone must be an object", field="zone")
    try:
        version = int(value["version"])
        digest = str(value["file_digest"])
        content_digest = str(value["content_digest"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BusinessError("E_SNAPSHOT", f"invalid snapshot zone metadata: {exc}", field="zone") from exc
    document = {"origin": value.get("origin"), "records": []}
    records = value.get("records")
    if not isinstance(records, list):
        raise BusinessError("E_SNAPSHOT", "snapshot zone records must be an array", field="zone")
    for rr in records:
        restored = rr_from_json(rr)
        record: dict[str, Any] = {
            "name": restored.name,
            "type": restored.type_name,
            "ttl": restored.ttl,
            "rdata": restored.rdata_text(),
        }
        document["records"].append(record)
    image = image_from_snapshot(document, version=version, file_digest=digest, max_ttl=max_ttl)
    if image.content_digest != content_digest:
        raise BusinessError("E_SNAPSHOT", "snapshot zone content digest mismatch", field="zone")
    return image


def cache_entry_to_json(entry: CacheEntry, remaining_ttl: int) -> dict[str, Any]:
    age = max(0, entry.ttl - remaining_ttl)
    answers = [rr_to_json(record) for record in entry.answers]
    authority = [rr_to_json(record) for record in entry.authority]
    return {
        "name": entry.key.name,
        "qtype": entry.key.qtype,
        "qclass": entry.key.qclass,
        "rcode": entry.rcode,
        "ttl": entry.ttl,
        "remaining_ttl": remaining_ttl,
        "stored_age_seconds": age,
        "first_write_basis": "CLOCK.monotonic_ns",
        "source": entry.source,
        "source_serial": entry.source_serial,
        "negative": entry.negative,
        "answers": answers,
        "authority": authority,
        "attribution": list(entry.attribution),
        "metadata": list(entry.metadata),
    }


def cache_entry_from_json(value: object, *, now_ns: int) -> CacheEntry:
    if not isinstance(value, dict):
        raise BusinessError("E_SNAPSHOT", "snapshot cache entry must be an object", field="cache")
    try:
        key = CacheKey(
            normalize_name(str(value["name"])),
            int(value["qtype"]),
            int(value.get("qclass", CLASS_IN)),
        )
        rcode = int(value["rcode"])
        ttl = int(value["ttl"])
        source = str(value["source"])
        source_serial = int(value["source_serial"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BusinessError("E_SNAPSHOT", f"invalid snapshot cache entry: {exc}", field="cache") from exc
    if source not in ("ZONE", "NEGATIVE", "UPSTREAM"):
        raise BusinessError("E_SNAPSHOT", f"invalid cache source: {source}", field="source")
    if ttl < 0 or ttl > 0x7FFFFFFF:
        raise BusinessError("E_SNAPSHOT", "snapshot cache TTL is outside the DNS range", field="ttl")
    try:
        age = int(value.get("stored_age_seconds", 0))
    except (TypeError, ValueError) as exc:
        raise BusinessError("E_SNAPSHOT", "snapshot cache age is invalid", field="stored_age_seconds") from exc
    if age < 0:
        raise BusinessError("E_SNAPSHOT", "snapshot cache age is negative", field="stored_age_seconds")
    raw_answers = value.get("answers", [])
    raw_authority = value.get("authority", [])
    if not isinstance(raw_answers, list) or not isinstance(raw_authority, list):
        raise BusinessError("E_SNAPSHOT", "snapshot cache records must be arrays", field="cache")
    answers = tuple(rr_from_json(record) for record in raw_answers)
    authority = tuple(rr_from_json(record) for record in raw_authority)
    attribution = value.get("attribution", [])
    metadata = value.get("metadata", [])
    if not isinstance(attribution, list) or not all(isinstance(item, str) for item in attribution):
        raise BusinessError("E_SNAPSHOT", "cache attribution must be strings", field="attribution")
    if not isinstance(metadata, list) or not all(isinstance(item, str) for item in metadata):
        raise BusinessError("E_SNAPSHOT", "cache metadata must be strings", field="metadata")
    return CacheEntry(
        key=key,
        rcode=rcode,
        answers=answers,
        authority=authority,
        ttl=ttl,
        stored_ns=now_ns - age * 1_000_000_000,
        source=source,  # type: ignore[arg-type]
        source_serial=source_serial,
        attribution=tuple(attribution),
        negative=bool(value.get("negative", False)),
        metadata=tuple(metadata),
    )


@dataclass(frozen=True)
class SnapshotInfo:
    path: Path
    snapshot_id: str
    name: str
    revision: int
    zone_version: int
    zone_digest: str
    cache_entries: int
    upstream_version: int
    created_utc: str

    @property
    def readonly(self) -> bool:
        return True


class SnapshotStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _slug(self, name: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
        if not slug:
            slug = "snapshot-" + hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
        return slug[:48]

    def next_revision(self, name: str) -> int:
        slug = self._slug(name)
        revisions = []
        for path in self.root.glob(f"*-{slug}.json"):
            match = _SNAPSHOT_ID_RE.match(path.stem)
            if match and match.group("slug") == slug:
                revisions.append(int(match.group("revision")))
        return max(revisions, default=0) + 1

    def save(
        self,
        *,
        name: str,
        zone: ZoneImage,
        cache_entries: Iterable[tuple[CacheEntry, int]],
        history: list[dict[str, Any]],
        negative_policy: Mapping[str, Any],
        upstream_version: int,
        cache_generation: int,
        state_epoch: int,
        clock: Clock,
        created_utc: str | None = None,
    ) -> tuple[SnapshotInfo, dict[str, Any]]:
        if not isinstance(name, str) or not name.strip():
            raise BusinessError("E_SNAPSHOT", "snapshot name must not be empty", field="name")
        revision = self.next_revision(name)
        snapshot_id = f"{revision:04d}-{self._slug(name)}"
        payload: dict[str, Any] = {
            "format": SNAPSHOT_FORMAT,
            "snapshot_id": snapshot_id,
            "name": name.strip(),
            "revision": revision,
            "created_utc": created_utc or datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "clock_source": clock.source,
            "zone": zone_to_json(zone),
            "cache_generation": cache_generation,
            "state_epoch": state_epoch,
            "negative_policy": {
                "max_ttl": int(negative_policy["max_ttl"]),
                "basis": str(negative_policy.get("basis", "min(SOA TTL, SOA MINIMUM) with cap")),
            },
            "upstream_version": int(upstream_version),
            "cache_entries": [
                cache_entry_to_json(entry, remaining_ttl)
                for entry, remaining_ttl in sorted(cache_entries, key=lambda item: item[0].key.display)
            ],
            "history": history,
        }
        payload["checksum"] = _checksum(payload)
        path = self.root / f"{snapshot_id}.json"
        encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        temp = path.with_suffix(".json.tmp")
        with temp.open("wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        try:
            path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        except OSError:
            pass
        info = self._info_from_payload(path, payload)
        return info, payload

    def _read(self, path: Path) -> dict[str, Any]:
        try:
            raw = path.read_text(encoding="utf-8")
            payload = json.loads(raw)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BusinessError("E_SNAPSHOT", f"cannot read snapshot: {exc}", field="file") from exc
        if not isinstance(payload, dict):
            raise BusinessError("E_SNAPSHOT", "snapshot root must be an object", field="file")
        if payload.get("format") != SNAPSHOT_FORMAT:
            raise BusinessError("E_SNAPSHOT", "unsupported snapshot format", field="format")
        if payload.get("checksum") != _checksum(payload):
            raise BusinessError("E_SNAPSHOT", "snapshot checksum mismatch", field="checksum")
        return payload

    def _info_from_payload(self, path: Path, payload: Mapping[str, Any]) -> SnapshotInfo:
        try:
            zone = payload["zone"]
            assert isinstance(zone, dict)
            entries = payload["cache_entries"]
            assert isinstance(entries, list)
            return SnapshotInfo(
                path=path,
                snapshot_id=str(payload["snapshot_id"]),
                name=str(payload["name"]),
                revision=int(payload["revision"]),
                zone_version=int(zone["version"]),
                zone_digest=str(zone["file_digest"]),
                cache_entries=len(entries),
                upstream_version=int(payload["upstream_version"]),
                created_utc=str(payload["created_utc"]),
            )
        except (KeyError, TypeError, ValueError, AssertionError) as exc:
            raise BusinessError("E_SNAPSHOT", f"invalid snapshot metadata: {exc}", field="file") from exc

    def load(self, locator: str) -> dict[str, Any]:
        path = Path(locator)
        if not path.exists():
            direct = self.root / locator
            candidates = [direct, direct.with_suffix(".json"), self.root / f"{locator}.json"]
            path = next((candidate for candidate in candidates if candidate.exists()), direct)
        if not path.exists():
            matches = [item.path for item in self.list() if item.name == locator]
            if not matches:
                hashed = self.root / f"{locator}.json"
                matches = list(self.root.glob(f"*-{locator}.json")) if hashed.exists() is False else [hashed]
            if not matches:
                raise BusinessError("E_SNAPSHOT", f"snapshot not found: {locator}", field="locator")
            path = sorted(matches)[-1]
        return self._read(path)

    def list(self) -> list[SnapshotInfo]:
        items: list[SnapshotInfo] = []
        for path in sorted(self.root.glob("*.json")):
            try:
                payload = self._read(path)
                items.append(self._info_from_payload(path, payload))
            except BusinessError:
                continue
        return sorted(items, key=lambda item: (item.name, item.revision, item.snapshot_id))

    def payload_info(self, payload: Mapping[str, Any]) -> SnapshotInfo:
        path = self.root / f"{payload['snapshot_id']}.json"
        return self._info_from_payload(path, payload)


def compare_state(
    payload: Mapping[str, Any],
    *,
    current_zone: ZoneImage,
    current_cache_entries: Iterable[CacheEntry],
    current_negative_max_ttl: int,
    current_upstream_version: int,
) -> tuple[bool, list[str]]:
    diffs: list[str] = []
    zone = payload.get("zone")
    if not isinstance(zone, dict):
        raise BusinessError("E_SNAPSHOT", "snapshot has no zone object", field="zone")
    if int(zone.get("version", -1)) != current_zone.version:
        diffs.append(f"zone-version snapshot={zone.get('version')} current={current_zone.version}")
    if str(zone.get("file_digest", "")) != current_zone.digest:
        diffs.append("zone-digest mismatch")
    policy = payload.get("negative_policy", {})
    if not isinstance(policy, dict):
        diffs.append("negative-policy invalid")
    elif int(policy.get("max_ttl", -1)) != int(current_negative_max_ttl):
        diffs.append(
            f"negative-max-ttl snapshot={policy.get('max_ttl')} current={current_negative_max_ttl}"
        )
    snapshot_upstream = int(payload.get("upstream_version", -1))
    if snapshot_upstream != current_upstream_version:
        diffs.append(f"upstream-version snapshot={snapshot_upstream} current={current_upstream_version}")

    raw_entries = payload.get("cache_entries", [])
    if not isinstance(raw_entries, list):
        diffs.append("cache-entries invalid")
        raw_entries = []
    snapshot_map: dict[str, tuple[str, int]] = {}
    for raw in raw_entries:
        if not isinstance(raw, dict):
            diffs.append("cache-entry invalid")
            continue
        try:
            key = CacheKey(normalize_name(str(raw["name"])), int(raw["qtype"]), int(raw.get("qclass", 1))).display
            snapshot_map[key] = (str(raw["source"]), int(raw["source_serial"]))
        except (KeyError, TypeError, ValueError):
            diffs.append("cache-entry key invalid")
    current_map = {
        entry.key.display: (entry.source, entry.source_serial)
        for entry in current_cache_entries
    }
    for key in sorted(set(snapshot_map) | set(current_map)):
        if key not in snapshot_map:
            diffs.append(f"entry-extra current={key}")
            continue
        if key not in current_map:
            diffs.append(f"entry-missing current={key}")
            continue
        if snapshot_map[key][0] != current_map[key][0]:
            diffs.append(
                f"entry-source key={key} snapshot={snapshot_map[key][0]} current={current_map[key][0]}"
            )
        if snapshot_map[key][1] != current_map[key][1]:
            diffs.append(
                f"entry-source-serial key={key} snapshot={snapshot_map[key][1]} current={current_map[key][1]}"
            )
    return (not diffs, sorted(diffs))
