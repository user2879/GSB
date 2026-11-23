from __future__ import annotations

import socket
from pathlib import Path

from gray_merge_lab.engine import GatewayEngine, InProcessTransport, ModelBank
from gray_merge_lab.ledger import Ledger
from gray_merge_lab.specs import BackendSpec, LoadedSpec, RuleSpec, SampleSpec


def free_ports(count: int) -> list[int]:
    sockets = []
    ports = []
    try:
        for _ in range(count):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", 0))
            sockets.append(sock)
            ports.append(int(sock.getsockname()[1]))
    finally:
        for sock in sockets:
            sock.close()
    return ports


def test_spec(
    ports: list[int] | None = None,
    *,
    sample_count: int = 4,
    batch_size: int = 2,
    weight_bp: int = 10000,
    client_key: str = "client-A",
) -> LoadedSpec:
    ports = ports or free_ports(2)
    backends = (
        BackendSpec("stable", "127.0.0.1", ports[0], 4, 5, 11, "fp32-v1"),
        BackendSpec("canary", "127.0.0.1", ports[1], 4, 6, 22, "int8-sym-v1"),
    )
    rules = (RuleSpec("rule-1", "/api", "stable", "canary", weight_bp, batch_size),)
    samples = tuple(
        SampleSpec(
            sample_id=f"s{index:03d}",
            sample_seq=index,
            client_key=client_key,
            rule_id="rule-1",
            request_path=f"/api/{index}",
            input_seed=1000 + index,
        )
        for index in range(1, sample_count + 1)
    )
    return LoadedSpec(backends, rules, samples)


def build_engine(
    tmp_path: Path,
    *,
    spec: LoadedSpec | None = None,
    duration: int = 7,
    snapshot: bool = True,
    **spec_kwargs,
) -> tuple[GatewayEngine, Ledger, LoadedSpec]:
    spec = spec or test_spec(**spec_kwargs)
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    bank = ModelBank(spec)
    transport = InProcessTransport(bank, duration_provider=lambda *_: duration)
    engine = GatewayEngine(
        spec,
        ledger,
        scale_exp=3,
        snapshot_path=(tmp_path / "snapshot.json") if snapshot else None,
        transport=transport,
    )
    return engine, ledger, spec
