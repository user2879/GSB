"""Local JSON authoritative-zone loading, validation and immutable images."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .errors import BusinessError, zone_error
from .names import is_subdomain, normalize_name, normalize_zone_name
from .wire import (
    CLASS_IN,
    SOAData,
    TYPE_BY_NAME,
    ResourceRecord,
    type_to_int,
)


DEFAULT_ZONE_MAX_TTL = 2_147_483_647
SUPPORTED_ZONE_TYPES = frozenset(TYPE_BY_NAME)


@dataclass(frozen=True)
class ZoneImage:
    origin: str
    version: int
    digest: str
    content_digest: str
    source_path: str | None
    records: tuple[ResourceRecord, ...]
    by_name: Mapping[str, tuple[ResourceRecord, ...]]
    owner_names: frozenset[str]
    soa: ResourceRecord

    def is_covered(self, name: str) -> bool:
        return is_subdomain(name, self.origin)

    @property
    def negative_ttl(self) -> int:
        if not isinstance(self.soa.rdata, SOAData):
            return 0
        return max(0, min(self.soa.ttl, self.soa.rdata.minimum))

    def records_at(self, name: str) -> tuple[ResourceRecord, ...]:
        return self.by_name.get(normalize_name(name), ())

    def identity(self) -> tuple[int, str, str]:
        return (self.version, self.digest, self.content_digest)


@dataclass(frozen=True)
class ZoneLoadResult:
    image: ZoneImage
    changed: bool


def _required(mapping: Mapping[str, Any], field: str, index: int) -> Any:
    if field not in mapping:
        raise zone_error("missing required field", index, field)
    return mapping[field]


def _parse_soa(value: str, index: int) -> SOAData:
    parts = value.split()
    if len(parts) != 7:
        raise zone_error("SOA RDATA must contain 7 fields", index, "rdata")
    try:
        mname = normalize_name(parts[0], allow_wildcard=False)
        rname = normalize_name(parts[1], allow_wildcard=False)
    except ValueError as exc:
        raise zone_error(f"invalid SOA name: {exc}", index, "rdata") from exc
    numbers: list[int] = []
    for position, token in enumerate(parts[2:], start=2):
        try:
            number = int(token, 10)
        except ValueError as exc:
            raise zone_error(f"SOA field {position} must be an integer", index, "rdata") from exc
        if not 0 <= number <= 0xFFFFFFFF:
            raise zone_error(f"SOA field {position} is outside the 32-bit range", index, "rdata")
        numbers.append(number)
    return SOAData(mname, rname, *numbers)


def _normalize_rdata(type_name: str, value: Any, index: int) -> str | SOAData:
    if not isinstance(value, str):
        raise zone_error("RDATA must be a string", index, "rdata")
    text = value.strip()
    if type_name == "A":
        try:
            return str(ipaddress.IPv4Address(text))
        except ipaddress.AddressValueError as exc:
            raise zone_error(f"invalid IPv4 address: {exc}", index, "rdata") from exc
    if type_name == "AAAA":
        try:
            return str(ipaddress.IPv6Address(text))
        except ipaddress.AddressValueError as exc:
            raise zone_error(f"invalid IPv6 address: {exc}", index, "rdata") from exc
    if type_name in ("CNAME", "NS"):
        try:
            return normalize_name(text, allow_wildcard=False)
        except ValueError as exc:
            raise zone_error(f"invalid {type_name} target: {exc}", index, "rdata") from exc
    if type_name == "SOA":
        return _parse_soa(text, index)
    raise zone_error(f"unsupported record type: {type_name}", index, "type")


def _zone_digest(origin: str, records: tuple[ResourceRecord, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(origin.encode("ascii"))
    digest.update(b"\n")
    for record in sorted(records, key=lambda rr: rr.canonical()):
        digest.update(record.canonical().encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def parse_zone_document(
    document: object,
    *,
    file_digest: str,
    version: int,
    source_path: str | None = None,
    max_ttl: int = DEFAULT_ZONE_MAX_TTL,
) -> ZoneImage:
    if not isinstance(document, dict):
        raise BusinessError("E_ZONE", "zone document root must be a JSON object", field="root")
    if "origin" not in document:
        raise BusinessError("E_ZONE", "missing origin", field="origin")
    try:
        origin = normalize_zone_name(document["origin"])
    except (TypeError, ValueError) as exc:
        raise BusinessError("E_ZONE", f"invalid origin: {exc}", field="origin") from exc
    raw_records = document.get("records")
    if not isinstance(raw_records, list):
        raise BusinessError("E_ZONE", "records must be a JSON array", field="records")

    records: list[ResourceRecord] = []
    seen: set[tuple[str, int, str]] = set()
    soa_records: list[ResourceRecord] = []
    names_with_cname: set[str] = set()
    for index, entry in enumerate(raw_records):
        if not isinstance(entry, dict):
            raise zone_error("record must be a JSON object", index, "record")
        name_value = _required(entry, "name", index)
        type_value = _required(entry, "type", index)
        ttl_value = _required(entry, "ttl", index)
        rdata_value = _required(entry, "rdata", index)
        if not isinstance(type_value, str):
            raise zone_error("type must be a string", index, "type")
        type_name = type_value.upper()
        if type_name not in SUPPORTED_ZONE_TYPES:
            raise zone_error("unsupported record type", index, "type", type=type_value)
        try:
            name = normalize_name(name_value, allow_wildcard=True)
        except (TypeError, ValueError) as exc:
            raise zone_error(f"invalid owner name: {exc}", index, "name") from exc
        if not is_subdomain(name, origin):
            raise zone_error("record owner is outside the zone origin", index, "name")
        if isinstance(ttl_value, bool) or not isinstance(ttl_value, int):
            raise zone_error("TTL must be a non-negative integer", index, "ttl")
        if ttl_value < 0 or ttl_value > max_ttl:
            raise zone_error(f"TTL must be between 0 and {max_ttl}", index, "ttl")
        rdata = _normalize_rdata(type_name, rdata_value, index)
        rtype = type_to_int(type_name)
        record = ResourceRecord(name, rtype, CLASS_IN, ttl_value, rdata)
        duplicate_key = (name, rtype, record.rdata_text())
        if duplicate_key in seen:
            raise zone_error("duplicate record", index, "record")
        seen.add(duplicate_key)
        records.append(record)
        if type_name == "SOA":
            soa_records.append(record)
        if type_name == "CNAME":
            names_with_cname.add(name)

    if len(soa_records) != 1:
        raise BusinessError("E_ZONE", "zone must contain exactly one SOA record", field="records")
    soa = soa_records[0]
    if soa.name != origin:
        raise BusinessError("E_ZONE", "SOA owner must equal the zone origin", field="records")
    if names_with_cname and any(
        record.rtype != TYPE_BY_NAME["CNAME"] and record.name in names_with_cname for record in records
    ):
        raise BusinessError("E_ZONE", "CNAME owner cannot also carry other record types", field="records")
    cname_counts: dict[str, int] = {}
    for record in records:
        if record.rtype == TYPE_BY_NAME["CNAME"]:
            cname_counts[record.name] = cname_counts.get(record.name, 0) + 1
    if any(count != 1 for count in cname_counts.values()):
        raise BusinessError("E_ZONE", "a CNAME owner must have exactly one CNAME record", field="records")

    record_tuple = tuple(records)
    by_name: dict[str, list[ResourceRecord]] = {}
    for record in record_tuple:
        by_name.setdefault(record.name, []).append(record)
    immutable_by_name = MappingProxyType({name: tuple(values) for name, values in by_name.items()})
    return ZoneImage(
        origin=origin,
        version=version,
        digest=file_digest,
        content_digest=_zone_digest(origin, record_tuple),
        source_path=source_path,
        records=record_tuple,
        by_name=immutable_by_name,
        owner_names=frozenset(by_name),
        soa=soa,
    )


def image_from_snapshot(
    document: object,
    *,
    version: int,
    file_digest: str,
    max_ttl: int = DEFAULT_ZONE_MAX_TTL,
) -> ZoneImage:
    image = parse_zone_document(
        document,
        file_digest=file_digest,
        version=version,
        source_path=None,
        max_ttl=max_ttl,
    )
    return image


class ZoneManager:
    """Loads zone files and assigns monotonic versions based on content digests."""

    def __init__(self, *, max_ttl: int = DEFAULT_ZONE_MAX_TTL) -> None:
        self.max_ttl = max_ttl
        self._next_version = 1
        self._active: ZoneImage | None = None
        self._last_digest: str | None = None
        self._lock = threading.RLock()

    @property
    def active(self) -> ZoneImage | None:
        with self._lock:
            return self._active

    def load_file(self, path: str | Path) -> ZoneLoadResult:
        source = Path(path)
        try:
            raw = source.read_bytes()
        except OSError as exc:
            raise BusinessError("E_ZONE", f"cannot read zone file: {exc}", field="file") from exc
        digest = hashlib.sha256(raw).hexdigest()
        try:
            document = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BusinessError("E_ZONE", f"invalid zone JSON: {exc}", field="file") from exc
        with self._lock:
            active = self._active
            if active is not None and digest == active.digest:
                return ZoneLoadResult(active, changed=False)
            version = self._next_version
            image = parse_zone_document(
                document,
                file_digest=digest,
                version=version,
                source_path=str(source.resolve()),
                max_ttl=self.max_ttl,
            )
            self._next_version += 1
            self._active = image
            self._last_digest = digest
            return ZoneLoadResult(image, changed=True)

    def install_image(self, image: ZoneImage) -> None:
        with self._lock:
            self._next_version = max(self._next_version, image.version + 1)
            self._active = image
            self._last_digest = image.digest

    def next_version(self) -> int:
        with self._lock:
            return self._next_version

