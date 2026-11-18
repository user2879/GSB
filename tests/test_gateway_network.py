from __future__ import annotations

import dataclasses
import socket
from concurrent.futures import ThreadPoolExecutor

from graygate.gateway import GatewayStack
from graygate.ledger import LedgerStore
from graygate.network import send_json_line


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_jsonl_gateway_admits_and_routes_on_loopback(sample_spec, tmp_path):
    ports = []
    while len(ports) < 5:
        port = _free_port()
        if port not in ports:
            ports.append(port)
    models = tuple(
        dataclasses.replace(model, port=port)
        for model, port in zip(sample_spec.models, ports[1:])
    )
    spec = dataclasses.replace(sample_spec, port=ports[0], models=models)
    ledger = LedgerStore(tmp_path / "gateway.db")
    stack = GatewayStack(spec, ledger)
    try:
        stack.start()

        def call(index: int) -> dict:
            return send_json_line(
                spec.port,
                {
                    "client_key": f"client-{index % 4}",
                    "feature": "alpha:payload" if index % 2 == 0 else "beta:payload",
                    "sample": [((index + offset) % 11) / 11.0 for offset in range(8)],
                },
            )

        with ThreadPoolExecutor(max_workers=8) as gate:
            responses = list(gate.map(call, range(16)))
        assert all(response["ok"] for response in responses)
        assert ledger.current_version() == 16
        assert all(response["batch_seq"] >= 1 for response in responses)
        assert all(response["stage_id"].endswith("-0") for response in responses)
        assert all(response["side"] == "stable" for response in responses)
        duplicate_key = [
            response
            for index, response in enumerate(responses)
            if index % 4 == 0 and index % 2 == 0
        ]
        assert len(duplicate_key) >= 2
        assert len({item["side"] for item in duplicate_key}) == 1
        assert ledger.verify_ledger_chain()[0] is True
    finally:
        stack.stop()
        ledger.close()

