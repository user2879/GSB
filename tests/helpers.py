from __future__ import annotations

import socket

from gsb.model import BatchSpec, CheckpointSpec, EventSpec, LayerSpec, SessionSpec, ShardSpec


def free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


def make_shard(
    shard_id: str = "shard-a",
    *,
    port: int | None = None,
    seed: int = 1001,
    parameter_version: int = 2,
    lease_limit: int = 2,
    concurrency_limit: int = 2,
    sample_dim: int = 4,
    scale: int = 1000,
) -> ShardSpec:
    return ShardSpec(
        shard_id=shard_id,
        host="127.0.0.1",
        port=port or free_port(),
        lease_limit=lease_limit,
        concurrency_limit=concurrency_limit,
        seed=seed,
        parameter_version=parameter_version,
        sample_dim=sample_dim,
        lease_ttl_ms=30000,
        layers=(
            LayerSpec("embed", sample_dim, 3, scale),
            LayerSpec("head", 3, 2, 500, ("embed",)),
        ),
    )


def make_session(
    shards: tuple[ShardSpec, ...] | None = None,
    *,
    checkpoints: tuple[CheckpointSpec, ...] | None = None,
    batches: tuple[BatchSpec, ...] | None = None,
    events: tuple[EventSpec, ...] | None = None,
    global_budget: int = 4,
) -> SessionSpec:
    shards = shards or (make_shard(),)
    if checkpoints is None:
        checkpoints = tuple(
            CheckpointSpec(
                f"cp-{shard.shard_id}",
                shard.shard_id,
                shard.parameter_version,
                "0" * 64,
            )
            for shard in shards
        )
    return SessionSpec(
        format_version=1,
        global_budget=global_budget,
        shards=tuple(shards),
        checkpoints=tuple(checkpoints),
        batches=tuple(batches or ()),
        events=tuple(events or ()),
    )


def simple_session(*, events: tuple[EventSpec, ...] = ()) -> SessionSpec:
    shard = make_shard()
    checkpoint = CheckpointSpec("cp", shard.shard_id, shard.parameter_version, "0" * 64)
    batches = (
        BatchSpec("b1", shard.shard_id, (1, 2), 1, 1, 100),
        BatchSpec("b2", shard.shard_id, (3,), 1, 2, 100),
    )
    return make_session(
        (shard,),
        checkpoints=(checkpoint,),
        batches=batches,
        events=events,
    )
