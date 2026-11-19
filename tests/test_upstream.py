from __future__ import annotations

from pathlib import Path

import pytest

from dns_local.errors import BusinessError
from dns_local.upstream import UpstreamClient, UpstreamSimulator
from dns_local.wire import TYPE_BY_NAME, Question

from .conftest import free_tcp_port


def test_upstream_chain_validation_and_version_probe(upstream_v1: Path) -> None:
    port = free_tcp_port()
    with UpstreamSimulator(upstream_v1, port=port):
        client = UpstreamClient("127.0.0.1", port, timeout=1)
        result = client.fetch(Question("api.upstream.test.", TYPE_BY_NAME["A"]), 0x1234)
        assert result.message.header.transaction_id == 0x1234
        assert [record.type_name for record in result.answers] == ["CNAME", "A"]
        assert result.answers[-1].rdata == "198.51.100.11"
        assert result.upstream_version == 1
        assert client.probe_version("upstream.test.") == 1


@pytest.mark.parametrize("mode", ["bad-txid", "bad-question", "bad-chain"])
def test_upstream_validation_modes_raise_chain_error(upstream_v1: Path, mode: str) -> None:
    port = free_tcp_port()
    with UpstreamSimulator(upstream_v1, port=port, mode=mode):
        client = UpstreamClient("127.0.0.1", port, timeout=1)
        with pytest.raises(BusinessError) as caught:
            client.fetch(Question("api.upstream.test.", TYPE_BY_NAME["A"]), 0x7777)
        assert caught.value.code == "E_CHAIN"


def test_unreachable_upstream_is_e_upstream() -> None:
    port = free_tcp_port()
    client = UpstreamClient("127.0.0.1", port, timeout=0.2)
    with pytest.raises(BusinessError) as caught:
        client.fetch(Question("api.upstream.test.", TYPE_BY_NAME["A"]), 1)
    assert caught.value.code == "E_UPSTREAM"


def test_simulator_returns_a_wire_valid_conflicting_chain(upstream_v1: Path) -> None:
    port = free_tcp_port()
    with UpstreamSimulator(upstream_v1, port=port):
        client = UpstreamClient("127.0.0.1", port, timeout=1)
        result = client.fetch(Question("conflict.upstream.test.", TYPE_BY_NAME["A"]), 9)
        assert [record.type_name for record in result.answers] == ["CNAME", "A"]
        assert result.answers[-1].name == "www.lab.test."
        assert result.answers[-1].rdata == "203.0.113.66"
