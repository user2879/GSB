from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from .leases import LeaseService
from .protocol import NodeClient
from .spec import SpecBundle


@dataclass(frozen=True)
class ShardRun:
    shard_seq: int
    node_id: str
    lease_id: str
    digest: str
    content_bytes_digest: str
    exchange_format: str
    node_elapsed_ms: int
    queue_waited_ms: int


class VerificationScheduler:
    def __init__(self, spec: SpecBundle, service: LeaseService):
        self.spec = spec
        self.service = service

    def default_node_id(self, seq: int) -> str:
        nodes = sorted(self.spec.nodes, key=lambda node: node.node_id)
        return nodes[seq % len(nodes)].node_id

    def process_shard(
        self,
        shard_seq: int,
        node_id: str | None = None,
        *,
        crash_before_reply: bool = False,
        renew_with_version: int | None = None,
    ) -> ShardRun:
        content = self.spec.content
        node_id = node_id or self.default_node_id(shard_seq)
        shard_id = content.shard_id(shard_seq)
        acquisition = self.service.acquire(node_id, shard_id, shard_seq, content.shard_size)
        lease = acquisition.lease
        node = self.spec.node(node_id)
        client = NodeClient(node, self.spec.policy.socket_timeout_ms)
        client.grant_lease(lease)
        if renew_with_version is not None:
            lease = self.service.renew(lease, renew_with_version, client)
        response = client.verify_shard(
            lease=lease,
            content_key=content.content_key,
            shard_seq=shard_seq,
            crash_before_reply=crash_before_reply,
        )
        proof = response["proof"]
        exchange = response.get("exchange_format_version", "v3")
        self.service.complete(lease, proof, source_format=exchange)
        return ShardRun(
            shard_seq=shard_seq,
            node_id=node_id,
            lease_id=lease.lease_id,
            digest=proof["digest"],
            content_bytes_digest=proof["content_bytes_digest"],
            exchange_format=exchange,
            node_elapsed_ms=int(response.get("elapsed_ms", 0)),
            queue_waited_ms=acquisition.waited_ms,
        )

    def run(
        self,
        shard_seqs: Iterable[int] | None = None,
        *,
        max_workers: int = 4,
        node_selector: Callable[[int], str | None] | None = None,
    ) -> list[ShardRun]:
        content = self.spec.content
        sequences = list(range(content.shard_count)) if shard_seqs is None else list(shard_seqs)
        results: list[ShardRun] = []
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="edge-scheduler") as pool:
            futures = {
                pool.submit(
                    self.process_shard,
                    seq,
                    None if node_selector is None else node_selector(seq),
                ): seq
                for seq in sequences
            }
            for future in as_completed(futures):
                results.append(future.result())
        return sorted(results, key=lambda run: run.shard_seq)

    def run_on_node(self, node_id: str, shard_seqs: Iterable[int], *, max_workers: int = 2) -> list[ShardRun]:
        sequences = list(shard_seqs)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(self.process_shard, seq, node_id) for seq in sequences]
            results = [future.result() for future in as_completed(futures)]
        return sorted(results, key=lambda run: run.shard_seq)


def assert_quota_invariants(service: LeaseService) -> dict[str, int]:
    rows = service.ledger.active_lease_rows()
    global_active = len(rows)
    if global_active > service.spec.policy.global_max_active_leases:
        raise AssertionError("global active lease cap exceeded")
    by_node: dict[str, int] = {}
    slots: set[tuple[str, int]] = set()
    shards: set[str] = set()
    for row in rows:
        node_id = row["node_id"]
        by_node[node_id] = by_node.get(node_id, 0) + 1
        node = service.spec.node(node_id)
        if by_node[node_id] > node.effective_lease_cap:
            raise AssertionError(f"node cap exceeded for {node_id}")
        slot_key = (node_id, int(row["slot_index"]))
        if slot_key in slots:
            raise AssertionError(f"slot double-held: {slot_key}")
        slots.add(slot_key)
        if row["shard_id"] in shards:
            raise AssertionError(f"shard double-held: {row['shard_id']}")
        shards.add(row["shard_id"])
    budget = service.ledger.conn.execute(
        "SELECT bytes_reserved,bytes_verified FROM budget_state WHERE budget_id=1"
    ).fetchone()
    total_bytes = int(budget["bytes_reserved"]) + int(budget["bytes_verified"])
    if total_bytes > service.spec.policy.global_byte_budget:
        raise AssertionError("byte budget exceeded")
    return {
        "global_active": global_active,
        "total_bytes": total_bytes,
        "node_count": len(by_node),
    }
