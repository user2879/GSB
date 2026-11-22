"""本地确定性 JSON 区文件加载与权威解析。

区文件结构：
{
  "zone_version": 3,
  "origin": "example.test",
  "soa": {"mname": ..., "rname": ..., "serial": ..., "minimum": 60, ...},
  "records": [{"name": "www.example.test", "qtype": 1, "rclass": 1,
               "ttl": 300, "value": "10.0.0.5", "rdata_hex": "0a000005"}, ...]
}
区文件损坏（JSON 语法、结构、必需字段、版本倒退）统一抛 E_ZONE。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .errors import ZoneError
from .wire import CLASS_IN, TYPE_CNAME, TYPE_SOA


@dataclass
class ZoneRecord:
    name: str
    qtype: int
    value: str
    ttl: int
    rclass: int = CLASS_IN
    rdata_hex: str = ""

    def key(self) -> tuple[str, int]:
        return (self.name, self.qtype)


@dataclass
class Zone:
    origin: str
    zone_version: int
    soa: dict[str, int | str]
    records: list[ZoneRecord] = field(default_factory=list)

    def soa_minimum(self) -> int:
        return int(self.soa.get("minimum", 0))

    def soa_record_value(self) -> str:
        order = ("mname", "rname", "serial", "refresh", "retry", "expire", "minimum")
        return " ".join(str(self.soa[name]) for name in order)


def load_zone(path: str | Path) -> Zone:
    path = Path(path)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ZoneError(f"区文件无法读取: {path}", context={"path": str(path)}) from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ZoneError(
            "区文件 JSON 语法损坏",
            offset=exc.pos,
            fields=["<json>"],
            context={"path": str(path), "line": exc.lineno, "column": exc.colno},
        ) from exc
    return parse_zone(payload, source=str(path))


def parse_zone(payload: object, *, source: str = "<memory>") -> Zone:
    if not isinstance(payload, dict):
        raise ZoneError("区文件根节点必须为对象", fields=["<root>"], context={"path": source})
    missing = [name for name in ("zone_version", "origin", "soa", "records") if name not in payload]
    if missing:
        raise ZoneError("区文件必需字段缺失", fields=missing, context={"path": source})
    if not isinstance(payload["zone_version"], int) or isinstance(payload["zone_version"], bool) or payload["zone_version"] < 0:
        raise ZoneError("区版本必须为非负整数", fields=["zone_version"], context={"path": source})
    if not isinstance(payload["origin"], str) or not payload["origin"]:
        raise ZoneError("区 origin 非法", fields=["origin"], context={"path": source})
    soa = payload["soa"]
    if not isinstance(soa, dict) or "minimum" not in soa:
        raise ZoneError("SOA 段非法或缺少 minimum", fields=["soa"], context={"path": source})
    if not isinstance(soa["minimum"], int) or soa["minimum"] < 0:
        raise ZoneError("SOA minimum 必须为非负整数", fields=["soa.minimum"], context={"path": source})
    raw_records = payload["records"]
    if not isinstance(raw_records, list):
        raise ZoneError("records 必须为数组", fields=["records"], context={"path": source})
    records: list[ZoneRecord] = []
    for index, item in enumerate(raw_records):
        prefix = f"records[{index}]"
        if not isinstance(item, dict):
            raise ZoneError("区记录必须为对象", fields=[prefix], context={"path": source})
        missing_fields = [f"{prefix}.{name}" for name in ("name", "qtype", "ttl", "value") if name not in item]
        if missing_fields:
            raise ZoneError("区记录必需字段缺失", fields=missing_fields, context={"path": source})
        if not isinstance(item["name"], str) or not isinstance(item["value"], str):
            raise ZoneError("区记录 name/value 必须为字符串", fields=[f"{prefix}.name"], context={"path": source})
        if not isinstance(item["qtype"], int) or item["qtype"] <= 0:
            raise ZoneError("区记录 qtype 非法", fields=[f"{prefix}.qtype"], context={"path": source})
        if not isinstance(item["ttl"], int) or item["ttl"] < 0:
            raise ZoneError("区记录 ttl 非法", fields=[f"{prefix}.ttl"], context={"path": source})
        records.append(
            ZoneRecord(
                name=item["name"].lower(),
                qtype=item["qtype"],
                value=item["value"],
                ttl=item["ttl"],
                rclass=int(item.get("rclass", CLASS_IN)),
                rdata_hex=str(item.get("rdata_hex", "")),
            )
        )
    return Zone(
        origin=payload["origin"].lower(),
        zone_version=payload["zone_version"],
        soa=dict(soa),
        records=records,
    )


@dataclass
class AuthoritativeResult:
    rcode: int
    answers: list[ZoneRecord]
    soa: ZoneRecord | None
    cname_chain: list[ZoneRecord] = field(default_factory=list)


class AuthorityResolver:
    """对区数据执行确定性权威解析，含 CNAME 链跟随与 NXDOMAIN/无数据判定。"""

    def __init__(self, zone: Zone) -> None:
        self.zone = zone
        self._by_key: dict[tuple[str, int], list[ZoneRecord]] = {}
        self._names: set[str] = set()
        for record in zone.records:
            self._by_key.setdefault(record.key(), []).append(record)
            self._names.add(record.name)

    def resolve(self, qname: str, qtype: int) -> AuthoritativeResult:
        qname = qname.lower()
        chain: list[ZoneRecord] = []
        visited: set[str] = set()
        current = qname
        for _ in range(len(self._names) + 2):
            if current in visited:
                break
            visited.add(current)
            cname_records = self._by_key.get((current, TYPE_CNAME), [])
            exact = self._by_key.get((current, qtype), [])
            if exact:
                return AuthoritativeResult(0, exact, None, chain)
            if cname_records:
                cname = cname_records[0]
                chain.append(cname)
                current = cname.value.lower()
                continue
            if current in self._names:
                return AuthoritativeResult(0, [], self._soa_record(), chain)
            return AuthoritativeResult(3, [], self._soa_record(), chain)
        return AuthoritativeResult(3, [], self._soa_record(), chain)

    def _soa_record(self) -> ZoneRecord:
        return ZoneRecord(
            name=self.zone.origin,
            qtype=TYPE_SOA,
            value=self.zone.soa_record_value(),
            ttl=self.zone.soa_minimum(),
        )
