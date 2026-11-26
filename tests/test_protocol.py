from __future__ import annotations

import json
import socket
import threading
import time

import pytest

from ckptsim.errors import CkptError
from ckptsim.model import build_batch_samples, generate_parameter_set
from ckptsim.process import ShardProcess
from ckptsim.protocol import TcpShardClient
from ckptsim.spec import load_bundle


def _bundle(files):
    return load_bundle(files.shards, files.layers, files.checkpoints, files.events)


def _start_shard(files, shard_id: str = "shard-a") -> ShardProcess:
    bundle = _bundle(files)
    shard = bundle.shards.by_id()[shard_id]
    return ShardProcess(
        shard=shard,
        shards_path=str(files.shards),
        layers_path=str(files.layers),
        startup_timeout=6,
        lease_ttl_ms=5_000,
        client_timeout=3,
    ).start()


def test_real_worker_round_trip_and_optimistic_renew(spec_files) -> None:
    bundle = _bundle(spec_files)
    shard = bundle.shards.by_id()["shard-a"]
    version = bundle.layers.by_version()[1]
    parameters = generate_parameter_set(shard, version)
    samples = build_batch_samples(shard, parameters, "b-a1", [1, 2, 3])
    process = _start_shard(spec_files)
    try:
        client = process.client
        assert client is not None
        response = client.batch(
            shard_id="shard-a",
            batch_id="b-a1",
            sample_ids=[1, 2, 3],
            parameter_version=1,
            samples=samples,
            lease_id="lease-test",
            lease_version=1,
            lease_ttl_ms=5_000,
        )
        assert response["loss_digest"]
        assert response["parameter_digest"] == parameters.digest
        assert isinstance(response["duration_ms"], int)
        with pytest.raises(CkptError) as caught:
            client.renew(
                shard_id="shard-a",
                lease_id="lease-test",
                expected_version=99,
                lease_ttl_ms=5_000,
            )
        assert caught.value.code == "E_LEASE"
        renewed = client.renew(
            shard_id="shard-a",
            lease_id="lease-test",
            expected_version=1,
            lease_ttl_ms=5_000,
        )
        assert renewed["lease_version"] == 2
        with pytest.raises(CkptError) as caught:
            client.renew(
                shard_id="shard-a",
                lease_id="lease-test",
                expected_version=1,
                lease_ttl_ms=5_000,
            )
        assert caught.value.code == "E_LEASE"
    finally:
        process.stop()


def test_occupied_port_maps_to_e_port(spec_files) -> None:
    first = _start_shard(spec_files)
    second = None
    try:
        bundle = _bundle(spec_files)
        second = ShardProcess(
            shard=bundle.shards.by_id()["shard-a"],
            shards_path=str(spec_files.shards),
            layers_path=str(spec_files.layers),
            startup_timeout=4,
        )
        with pytest.raises(CkptError) as caught:
            second.start()
        assert caught.value.code == "E_PORT"
    finally:
        if second is not None:
            second.stop()
        first.stop()


def test_socket_read_timeout_maps_to_e_timeout() -> None:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    done = threading.Event()

    def server() -> None:
        try:
            connection, _ = listener.accept()
            with connection:
                connection.recv(65536)
                time.sleep(0.4)
        finally:
            done.set()

    thread = threading.Thread(target=server, daemon=True)
    thread.start()
    client = TcpShardClient(host="127.0.0.1", port=port, timeout=0.1)
    try:
        with pytest.raises(CkptError) as caught:
            client.request({"op": "health"})
        assert caught.value.code == "E_TIMEOUT"
    finally:
        done.wait(2)
        listener.close()


def test_missing_response_field_maps_to_e_field() -> None:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]

    def server() -> None:
        connection, _ = listener.accept()
        with connection:
            connection.recv(65536)
            connection.sendall(
                json.dumps(
                    {
                        "ok": True,
                        "shard_id": "shard-a",
                        "batch_id": "b-a1",
                        "parameter_version": 1,
                    }
                ).encode()
                + b"\n"
            )

    thread = threading.Thread(target=server, daemon=True)
    thread.start()
    client = TcpShardClient(host="127.0.0.1", port=port, timeout=1)
    try:
        with pytest.raises(CkptError) as caught:
            client.batch(
                shard_id="shard-a",
                batch_id="b-a1",
                sample_ids=[1],
                parameter_version=1,
                samples={},
                lease_id="lease",
                lease_version=1,
                lease_ttl_ms=1000,
            )
        assert caught.value.code == "E_FIELD"
        assert caught.value.field in {"generation", "parameter_digest", "loss"}
    finally:
        thread.join(2)
        listener.close()
