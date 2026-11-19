"""Authoritative-zone lookup with case folding, wildcards and CNAME chains."""

from __future__ import annotations

from dataclasses import dataclass, replace

from .errors import BusinessError
from .names import labels, normalize_name
from .wire import (
    RCODE_NOERROR,
    RCODE_NXDOMAIN,
    ResourceRecord,
    TYPE_BY_NAME,
    type_to_int,
)
from .zone import ZoneImage


@dataclass(frozen=True)
class Resolution:
    rcode: int
    answers: tuple[ResourceRecord, ...]
    authority: tuple[ResourceRecord, ...]
    ttl: int
    negative: bool
    source_serial: int
    attribution: tuple[str, ...]
    cname_hops: int = 0
    incomplete_target: str | None = None
    wildcard_used: bool = False


@dataclass(frozen=True)
class _Lookup:
    records: tuple[ResourceRecord, ...]
    exact_owner: bool
    wildcard_owner: str | None
    match: str


class ZoneResolver:
    def __init__(self, max_cname_hops: int = 8) -> None:
        if max_cname_hops < 0:
            raise ValueError("max_cname_hops must not be negative")
        self.max_cname_hops = max_cname_hops

    def _closest_encloser(self, zone: ZoneImage, name: str) -> str:
        parts = labels(name)
        for index in range(1, len(parts) + 1):
            suffix = ".".join(parts[index:]) + "." if parts[index:] else "."
            if suffix in zone.owner_names:
                return suffix
        return zone.origin

    def _lookup(self, zone: ZoneImage, name: str) -> _Lookup:
        exact = zone.by_name.get(name)
        if exact is not None:
            return _Lookup(tuple(exact), True, None, "exact")
        closest = self._closest_encloser(zone, name)
        wildcard = "*." + closest if closest != "." else "*."
        wildcard_records = zone.by_name.get(wildcard)
        if wildcard_records is None:
            return _Lookup((), False, None, "none")
        synthesized = tuple(replace(record, name=name) for record in wildcard_records)
        return _Lookup(synthesized, False, wildcard, "wildcard")

    def resolve(self, zone: ZoneImage, query_name: str, query_type: str | int) -> Resolution:
        qname = normalize_name(query_name)
        qtype = type_to_int(query_type)
        if not zone.is_covered(qname):
            raise BusinessError("E_STATE", "query name is outside the authoritative zone", field="name")

        answers: list[ResourceRecord] = []
        attribution: list[str] = []
        current = qname
        seen: set[str] = set()
        hops = 0
        wildcard_used = False

        while True:
            if current in seen:
                raise BusinessError(
                    "E_CNAME",
                    "CNAME loop detected",
                    details={"name": current, "hops": hops, "attribution": attribution},
                )
            seen.add(current)
            lookup = self._lookup(zone, current)
            if lookup.match == "wildcard":
                wildcard_used = True
                attribution.append(
                    f"match=wildcard source={lookup.wildcard_owner} synth={current} qname={qname}"
                )
            elif lookup.match == "exact":
                attribution.append(f"match=exact owner={current}")

            cname_records = [record for record in lookup.records if record.rtype == TYPE_BY_NAME["CNAME"]]
            typed_records = [record for record in lookup.records if record.rtype == qtype]

            if qtype == TYPE_BY_NAME["CNAME"]:
                if cname_records:
                    answers.extend(cname_records)
                    attribution.append(f"cname_query=terminal owner={current}")
                    return self._positive(
                        zone,
                        qname,
                        answers,
                        attribution,
                        hops,
                        wildcard_used,
                    )
                if lookup.exact_owner or lookup.match == "wildcard":
                    return self._negative(zone, qname, answers, attribution, hops, wildcard_used, nxdomain=False)
                return self._negative(zone, qname, answers, attribution, hops, wildcard_used, nxdomain=True)

            if typed_records:
                answers.extend(typed_records)
                attribution.append(f"answer_type={typed_records[0].type_name} count={len(typed_records)}")
                return self._positive(zone, qname, answers, attribution, hops, wildcard_used)

            if cname_records:
                cname = cname_records[0]
                answers.append(cname)
                hops += 1
                target = cname.target
                assert target is not None
                attribution.append(f"cname={current} target={target} hop={hops}")
                if hops > self.max_cname_hops:
                    raise BusinessError(
                        "E_CNAME",
                        "CNAME hop limit exceeded",
                        details={
                            "limit": self.max_cname_hops,
                            "hops": hops,
                            "target": target,
                            "attribution": attribution,
                        },
                    )
                if target in seen:
                    raise BusinessError(
                        "E_CNAME",
                        "CNAME loop detected",
                        details={"name": target, "hops": hops, "attribution": attribution},
                    )
                if not zone.is_covered(target):
                    attribution.append(f"delegate=out-of-zone target={target}")
                    return self._positive(
                        zone,
                        qname,
                        answers,
                        attribution,
                        hops,
                        wildcard_used,
                        incomplete_target=target,
                    )
                current = target
                continue

            if lookup.exact_owner or lookup.match == "wildcard":
                return self._negative(zone, qname, answers, attribution, hops, wildcard_used, nxdomain=False)
            return self._negative(zone, qname, answers, attribution, hops, wildcard_used, nxdomain=True)

    def _positive(
        self,
        zone: ZoneImage,
        qname: str,
        answers: list[ResourceRecord],
        attribution: list[str],
        hops: int,
        wildcard_used: bool,
        *,
        incomplete_target: str | None = None,
    ) -> Resolution:
        ttl = min((record.ttl for record in answers), default=0)
        attribution.append(f"cname_hops={hops}")
        return Resolution(
            rcode=RCODE_NOERROR,
            answers=tuple(answers),
            authority=(),
            ttl=ttl,
            negative=False,
            source_serial=zone.version,
            attribution=tuple(attribution),
            cname_hops=hops,
            incomplete_target=incomplete_target,
            wildcard_used=wildcard_used,
        )

    def _negative(
        self,
        zone: ZoneImage,
        qname: str,
        answers: list[ResourceRecord],
        attribution: list[str],
        hops: int,
        wildcard_used: bool,
        *,
        nxdomain: bool,
    ) -> Resolution:
        negative_ttl = zone.negative_ttl
        soa = replace(zone.soa, ttl=negative_ttl)
        authority = (soa,)
        ttl = negative_ttl
        if answers:
            ttl = min(ttl, min(answer.ttl for answer in answers))
        attribution.append(f"negative={'nxdomain' if nxdomain else 'nodata'} soa_min_ttl={negative_ttl}")
        attribution.append(f"cname_hops={hops}")
        return Resolution(
            rcode=RCODE_NXDOMAIN if nxdomain else RCODE_NOERROR,
            answers=tuple(answers),
            authority=authority,
            ttl=ttl,
            negative=True,
            source_serial=zone.version,
            attribution=tuple(attribution),
            cname_hops=hops,
            wildcard_used=wildcard_used,
        )


def rr_chain_summary(records: tuple[ResourceRecord, ...] | list[ResourceRecord]) -> str:
    return ",".join(record.canonical() for record in records)
