from __future__ import annotations

from pathlib import Path
import os
import time

import pytest

from dns_local.errors import BusinessError
from dns_local.resolver import ZoneResolver
from dns_local.zone import ZoneManager


@pytest.fixture
def loaded_zone(zone_v1: Path):
    manager = ZoneManager()
    result = manager.load_file(zone_v1)
    return manager, result.image, ZoneResolver(max_cname_hops=8)


def test_zone_versions_follow_content_digest(zone_v1: Path, zone_v2: Path) -> None:
    manager = ZoneManager()
    first = manager.load_file(zone_v1)
    assert first.changed is True
    assert first.image.version == 1
    same = manager.load_file(zone_v1)
    assert same.changed is False
    assert same.image.version == 1
    changed = manager.load_file(zone_v2)
    assert changed.changed is True
    assert changed.image.version == 2
    assert changed.image.digest != first.image.digest


def test_mtime_alone_does_not_create_a_new_zone_version(zone_v1: Path, tmp_path: Path) -> None:
    manager = ZoneManager()
    original = manager.load_file(zone_v1)
    copied = tmp_path / "same-content.json"
    copied.write_bytes(zone_v1.read_bytes())
    os.utime(copied, (time.time() + 1000, time.time() + 1000))
    reloaded = manager.load_file(copied)
    assert reloaded.changed is False
    assert reloaded.image.version == original.image.version
    assert reloaded.image.digest == original.image.digest


@pytest.mark.parametrize(
    "filename,index,field",
    [
        ("invalid_ttl.json", 1, "ttl"),
        ("invalid_type.json", 1, "type"),
        ("invalid_name.json", 1, "name"),
        ("invalid_rdata.json", 1, "rdata"),
    ],
)
def test_invalid_zone_is_rejected_with_index_and_field(
    root: Path, filename: str, index: int, field: str
) -> None:
    manager = ZoneManager()
    with pytest.raises(BusinessError) as caught:
        manager.load_file(root / "data" / "zones" / filename)
    assert caught.value.code == "E_ZONE"
    assert caught.value.record_index == index
    assert caught.value.field == field
    assert manager.active is None


def test_invalid_reload_keeps_the_previous_active_zone(zone_v1: Path, root: Path) -> None:
    manager = ZoneManager()
    valid = manager.load_file(zone_v1)
    with pytest.raises(BusinessError):
        manager.load_file(root / "data" / "zones" / "invalid_rdata.json")
    assert manager.active is not None
    assert manager.active.version == valid.image.version
    assert manager.active.digest == valid.image.digest


def test_case_insensitive_exact_lookup(loaded_zone) -> None:
    _, zone, resolver = loaded_zone
    result = resolver.resolve(zone, "WWW.LAB.TEST.", "A")
    assert result.rcode == 0
    assert result.answers[0].rdata == "192.0.2.10"
    assert "match=exact" in " ".join(result.attribution)


def test_wildcard_synthesis_uses_query_name_and_exact_wins(loaded_zone) -> None:
    _, zone, resolver = loaded_zone
    wildcard = resolver.resolve(zone, "anything.wild.lab.test.", "A")
    assert wildcard.answers[0].name == "anything.wild.lab.test."
    assert wildcard.answers[0].rdata == "192.0.2.99"
    assert wildcard.wildcard_used is True
    assert any("synth=anything.wild.lab.test." in item for item in wildcard.attribution)

    exact = resolver.resolve(zone, "keep.wild.lab.test.", "A")
    assert exact.answers[0].rdata == "192.0.2.88"
    assert exact.wildcard_used is False

    below_exact = resolver.resolve(zone, "x.keep.wild.lab.test.", "A")
    assert below_exact.rcode == 3


def test_cname_chain_and_attribution(loaded_zone) -> None:
    _, zone, resolver = loaded_zone
    result = resolver.resolve(zone, "alias.lab.test.", "A")
    assert [record.type_name for record in result.answers] == ["CNAME", "A"]
    assert result.answers[0].rdata == "www.lab.test."
    assert result.answers[1].rdata == "192.0.2.10"
    assert result.cname_hops == 1
    assert "cname_hops=1" in result.attribution


def test_cname_query_does_not_follow(loaded_zone) -> None:
    _, zone, resolver = loaded_zone
    result = resolver.resolve(zone, "alias.lab.test.", "CNAME")
    assert len(result.answers) == 1
    assert result.answers[0].type_name == "CNAME"
    assert result.cname_hops == 0


def test_cname_loop_and_hop_limit(loaded_zone) -> None:
    _, zone, resolver = loaded_zone
    with pytest.raises(BusinessError) as loop:
        resolver.resolve(zone, "loop1.lab.test.", "A")
    assert loop.value.code == "E_CNAME"
    with pytest.raises(BusinessError) as hops:
        resolver.resolve(zone, "c1.lab.test.", "A")
    assert hops.value.code == "E_CNAME"
    assert hops.value.details["limit"] == 8


def test_nodata_is_authoritative_negative_with_soa(loaded_zone) -> None:
    _, zone, resolver = loaded_zone
    result = resolver.resolve(zone, "nodata.lab.test.", "A")
    assert result.rcode == 0
    assert result.answers == ()
    assert result.negative is True
    assert result.authority[0].type_name == "SOA"
    assert result.ttl == 60


def test_multiple_same_type_records_are_returned(loaded_zone) -> None:
    _, zone, resolver = loaded_zone
    result = resolver.resolve(zone, "bulk.lab.test.", "A")
    assert len(result.answers) == 45
    assert result.ttl == 60
