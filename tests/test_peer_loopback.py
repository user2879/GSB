from __future__ import annotations

import socket

import pytest

from tcp_workbench.errors import E_CONNECT, E_PORT, BusinessError
from tcp_workbench.peer import DeterministicPeer, send_batch
from tcp_workbench.runner import run_loopback_session


def test_loopback_peer_ack_dupack_drop_delay_and_reorder(service_factory):
    service = service_factory()
    transcript, peer = run_loopback_session(
        service, port=0, send_segments=7, close_mode="close_wait"
    )
    assert peer.port >= 1024
    assert transcript.peer_responses
    actions = {response["action"] for response in transcript.peer_responses}
    assert "ack" in actions
    assert "dupack" in actions
    assert 6001 in transcript.dropped_seqs
    assert any(response["reordered"] for response in transcript.peer_responses)
    assert all("arrival_index" in response for response in transcript.peer_responses)
    assert service.items["connection_state"].value == "CLOSED"
    service.verify_incremental_equivalence()


def test_occupied_port_is_e_port(sample_config_script):
    _, script = sample_config_script
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        blocker.bind(("127.0.0.1", 0))
        blocker.listen(1)
        port = blocker.getsockname()[1]
        with pytest.raises(BusinessError) as shown:
            DeterministicPeer(script, "127.0.0.1", port).start()
        assert shown.value.code == E_PORT
    finally:
        blocker.close()


def test_connect_failure_is_e_connect():
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    with pytest.raises(BusinessError) as shown:
        send_batch(
            "127.0.0.1", port, [{"seq": 1001, "payload_bytes": 1000}], timeout_s=0.2
        )
    assert shown.value.code == E_CONNECT
