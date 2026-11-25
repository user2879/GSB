"""End-to-end orchestration over real TCP nodes, scheduler and ledger."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .analytics import wait_statistics
from .canonical import canonical_bytes, canonical_text, sha256_hex
from .cluster import NodeCluster
from .engine import ReplayEngine, result_digest
from .ledger import Ledger, write_snapshot
from .propagation import changed_node_ids
from .scheduler import QuotaScheduler
from .spec import SpecBundle, load_sample_dir
from .timeutil import monotonic_ms
from .verify import verify_ledger


class LiveScenarioRunner:
    def __init__(self, bundle: SpecBundle, output_dir: str | Path) -> None:
        self.bundle = bundle
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ledger = Ledger(self.output_dir / "ledger.sqlite")
        quotas = bundle.nodes_doc["quotas"]
        self.scheduler = QuotaScheduler(
            global_in_flight=int(quotas["global_in_flight"]),
            node_in_flight={
                node["node_id"]: int(node["in_flight_limit"]) for node in bundle.nodes
            },
        )
        self.cluster = NodeCluster(bundle.nodes, log_dir=self.output_dir / "node-logs")
        self.engine = ReplayEngine(bundle)

    def close(self) -> None:
        self.cluster.stop_all()
        self.scheduler.close()
        self.ledger.close()

    def _network(self, request_id: str, node_id: str, fn: Any) -> Any:
        return self.scheduler.submit(request_id, node_id, fn).result()

    def _intent_id(self, seq: int, lease_id: str) -> str:
        return f"intent-seq-{seq}-{lease_id}"

    def _record_intent(self, event: dict[str, Any], lease_id: str, op: str) -> str:
        intent_id = self._intent_id(int(event["seq"]), lease_id)
        self.ledger.record_intent(
            intent_id=intent_id,
            event_seq=int(event["seq"]),
            lease_id=lease_id,
            op=op,
            payload={"event": event, "lease_id": lease_id},
            monotonic_ms=monotonic_ms(),
        )
        return intent_id

    def _preflight_parallel_grants(self) -> dict[int, Any]:
        futures: dict[int, Any] = {}
        for event in self.bundle.events:
            if event["type"] != "grant" or not event.get("parallel"):
                continue
            request = self.bundle.requests and next(
                item for item in self.bundle.requests if item["lease_id"] == event["lease_id"]
            )
            client = self.cluster.clients[request["node_id"]]
            self._record_intent(event, request["lease_id"], "grant")
            futures[event["seq"]] = self.scheduler.submit(
                f"grant-seq-{event['seq']}",
                request["node_id"],
                lambda client=client, request=request: client.grant(
                    request["lease_id"], request["equity_id"], int(request["ttl_ms"])
                ),
            )
        return futures

    def _network_for_event(self, event: dict[str, Any]) -> list[str]:
        event_type = event["type"]
        seq = event["seq"]
        if event_type == "grant":
            request = next(
                item for item in self.bundle.requests if item["lease_id"] == event["lease_id"]
            )
            client = self.cluster.clients[request["node_id"]]
            intent_id = self._record_intent(event, request["lease_id"], "grant")
            self._network(
                f"grant-seq-{seq}",
                request["node_id"],
                lambda: client.grant(
                    request["lease_id"], request["equity_id"], int(request["ttl_ms"])
                ),
            )
            return [intent_id]

        if event_type in {"renew", "release"}:
            lease_id = event["lease_id"]
            lease = self.engine.leases[lease_id]
            tag = event.get("version_tag")
            tag = int(lease["generation"]) if tag is None else int(tag)
            client = self.cluster.clients[lease["node_id"]]
            method = client.renew if event_type == "renew" else client.release
            intent_id = self._record_intent(event, lease_id, event_type)
            self._network(
                f"{event_type}-seq-{seq}-{lease_id}",
                lease["node_id"],
                lambda: method(lease_id, tag),
            )
            return [intent_id]

        if event_type == "expire_scan":
            at_ms = int(event["at_ms"])
            targets = sorted(
                (
                    lease
                    for lease in self.engine.leases.values()
                    if lease["state"] in {"granted", "renewed"}
                    and int(lease["expires_at_ms"]) <= at_ms
                ),
                key=lambda lease: lease["lease_id"],
            )
            intent_ids: list[str] = []
            for lease in targets:
                client = self.cluster.clients[lease["node_id"]]
                tag = int(lease["generation"])
                intent_ids.append(self._record_intent(event, lease["lease_id"], "expire"))
                self._network(
                    f"expire-seq-{seq}-{lease['lease_id']}",
                    lease["node_id"],
                    lambda client=client, lease_id=lease["lease_id"], tag=tag: client.expire(
                        lease_id, tag, "ttl_expired"
                    ),
                )
            return intent_ids

        if event_type in {"version_evolved", "node_generation_bump"}:
            if event_type == "node_generation_bump":
                changed_nodes = [event["node_id"]]
            else:
                changed_nodes = changed_node_ids(
                    event, [node["node_id"] for node in self.bundle.nodes]
                )
            targets = sorted(
                (
                    lease
                    for lease in self.engine.leases.values()
                    if lease["node_id"] in changed_nodes
                    and lease["state"] in {"granted", "renewed"}
                ),
                key=lambda lease: lease["lease_id"],
            )
            intent_ids = []
            for lease in targets:
                client = self.cluster.clients[lease["node_id"]]
                tag = int(lease["generation"])
                intent_ids.append(
                    self._record_intent(event, lease["lease_id"], "generation_invalidate")
                )
                self._network(
                    f"invalidate-seq-{seq}-{lease['lease_id']}",
                    lease["node_id"],
                    lambda client=client, lease_id=lease["lease_id"], tag=tag: client.expire(
                        lease_id, tag, "generation_superseded"
                    ),
                )
            return intent_ids
        return []

    def run(self, start_order: list[str]) -> dict[str, Any]:
        handshakes = self.cluster.start_all(start_order)
        preflight = self._preflight_parallel_grants()
        try:
            for event in self.bundle.events:
                seq = int(event["seq"])
                before_count = len(self.engine.records)
                if seq in preflight:
                    preflight[seq].result()
                    event_for_id = event
                    intent_ids = [
                        self._intent_id(seq, event_for_id["lease_id"])
                    ] if event_for_id["type"] == "grant" else []
                else:
                    intent_ids = self._network_for_event(event)
                self.engine.apply_event(event)
                new_records = self.engine.records[before_count:]
                self.ledger.append_records(new_records, intent_ids=intent_ids)

            result = self.engine.result()
            verification = verify_ledger(self.bundle, self.ledger)
            snapshot_envelope = write_snapshot(self.output_dir / "snapshot.json", result)
            conclusion = {
                "versions": result["versions"],
                "leases": result["leases"],
                "ownership": result["ownership"],
                "capacity": result["capacity"],
            }
            conclusion_bytes = canonical_bytes(conclusion)
            (self.output_dir / "conclusion.json").write_bytes(conclusion_bytes)
            (self.output_dir / "result.json").write_bytes(
                canonical_bytes(result)
            )
            wait_records = sorted(
                self.scheduler.wait_records, key=lambda item: item["request_id"]
            )
            report = {
                "ok": True,
                "start_order": start_order,
                "handshakes": handshakes,
                "scheduler": {
                    "limits": self.scheduler.active_snapshot(),
                    "wait_records": wait_records,
                    "wait_statistics": wait_statistics(wait_records),
                },
                "ledger": self.ledger.recovery_report(),
                "snapshot_digest": snapshot_envelope["content_digest"],
                "verification": {
                    key: value
                    for key, value in verification.items()
                    if key != "canonical_conclusion"
                },
                "conclusion_digest": result_digest(result),
                "conclusion_sha256": sha256_hex(conclusion_bytes),
                "output_dir": str(self.output_dir),
            }
            (self.output_dir / "run_report.json").write_bytes(canonical_bytes(report))
            return report
        finally:
            self.close()


def prepare_dir(path: str | Path) -> Path:
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def run_order_once(
    bundle: SpecBundle,
    output_dir: str | Path,
    start_order: list[str],
) -> dict[str, Any]:
    path = prepare_dir(output_dir)
    runner = LiveScenarioRunner(bundle, path)
    try:
        return runner.run(start_order)
    finally:
        runner.close()


def run_demo(root: str | Path | None = None) -> dict[str, Any]:
    root = Path(root or Path.cwd())
    sample_dir = root / "samples"
    bundle = load_sample_dir(sample_dir)
    node_ids = [node["node_id"] for node in bundle.nodes]
    output_root = root / "artifacts" / "demo"
    output_root.mkdir(parents=True, exist_ok=True)

    forward = run_order_once(bundle, output_root / "order-forward", node_ids)
    reverse = run_order_once(
        bundle, output_root / "order-reverse", list(reversed(node_ids))
    )
    left = (output_root / "order-forward" / "conclusion.json").read_bytes()
    right = (output_root / "order-reverse" / "conclusion.json").read_bytes()
    identical = left == right
    comparison = {
        "ok": True,
        "byte_identical": identical,
        "forward_sha256": sha256_hex(left),
        "reverse_sha256": sha256_hex(right),
        "start_orders": {"forward": node_ids, "reverse": list(reversed(node_ids))},
        "business_codes": ["E_SPEC", "E_PORT", "E_TIMEOUT", "E_REPLY", "E_LEASE", "E_VERIFY", "E_SNAPSHOT"],
    }
    (output_root / "comparison.json").write_bytes(canonical_bytes(comparison))
    if not identical:
        raise AssertionError("start order changed deterministic conclusion")
    return {"forward": forward, "reverse": reverse, "comparison": comparison}
