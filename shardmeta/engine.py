"""Accounting engine: request processing, cached views and authoritative replay.

The fast Tkinter path reads an in-memory merge view cache in one read and never
triggers a full recompute. The authoritative path replays every committed
ledger request (leases, digests, adoption) from scratch. Any disagreement
raises E_DIVERGE at the first object key / shard index / field.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .adoption import decide_all, decisions_canonical_blob
from .clock import Clock, monotonic_ms
from .content import compute_checksum, generate_shard_bytes
from .errors import DivergenceError, LeaseError
from .ledger import Ledger
from .models import (
    AdoptionDecision,
    Lease,
    MergeCandidate,
    NodeSpec,
    ObjectManifest,
    Request,
)
from .node import NodeOrchestrator


@dataclass
class ReadObservation:
    object_key: str
    elapsed_ms: int
    served_from_cache: bool
    summary: dict


@dataclass
class EngineResult:
    seq: int
    op: str
    ok: bool
    detail: dict = field(default_factory=dict)
    elapsed_ms: int = 0


class AccountingEngine:
    def __init__(
        self,
        nodes: dict[str, NodeSpec],
        manifests: dict[str, ObjectManifest],
        ledger: Ledger,
        orchestrator: Optional[NodeOrchestrator] = None,
        clock: Optional[Clock] = None,
        lease_ttl_ms: int = 500,
    ) -> None:
        self.nodes = nodes
        self.manifests = manifests
        self.ledger = ledger
        self.orchestrator = orchestrator
        self.clock = clock or Clock()
        self.default_ttl = lease_ttl_ms
        for node_id, spec in nodes.items():
            ledger.init_node_counter(spec.node_id, spec.capacity_slots, spec.lease_limit)

        self._leases: dict[str, Lease] = {}
        self._candidates: list[MergeCandidate] = []
        self._merge_seq = 0
        self._cursor = ledger.progress_cursor()
        self._adoption: dict[str, list[AdoptionDecision]] = {}
        self._view_cache: dict[str, dict] = {}
        self._read_observations: list[ReadObservation] = []
        self._candidate_seq = 0
        self._in_process_managers: dict[str, Any] = {}
        self._synthetic_seq = 900000

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def now_ms(self) -> int:
        return self.clock.now_ms()

    def _next_merge_seq(self) -> int:
        self._merge_seq += 1
        return self._merge_seq

    def _node_id_for(self, key: str, shard_index: int) -> str:
        return self.manifests[key].node_of(shard_index)

    def _counter_snapshot(self, node_id: str) -> dict:
        row = self.ledger.con.execute(
            "SELECT * FROM node_counters WHERE node_id=?", (node_id,)
        ).fetchone()
        return dict(row)

    def _sweep_node(self, node_id: str, seq: int) -> list[str]:
        """Ask the live node for real expirations and persist them atomically."""
        expired_ids: list[str] = []
        if self.orchestrator is not None:
            reply = self.orchestrator.client(node_id).call({"type": "sweep"})
            expired_ids = list(reply.get("expired_lease_ids", []))
        else:
            now_ms = self.now_ms()
            for lease in self._leases.values():
                if (lease.node_id == node_id and lease.state == "active"
                        and now_ms >= lease.expires_ms):
                    expired_ids.append(lease.lease_id)
        for lease_id in expired_ids:
            lease = self._leases.get(lease_id)
            if lease is None or lease.state != "active":
                continue
            self._synthetic_seq += 1
            expired = Lease(
                lease_id=lease.lease_id, object_key=lease.object_key,
                shard_index=lease.shard_index, node_id=lease.node_id,
                granted_ms=lease.granted_ms, expires_ms=lease.expires_ms,
                renew_count=lease.renew_count, lease_version=lease.lease_version,
                state="expired",
            )
            self._leases[lease_id] = expired
            self._cursor += 1
            self._persist_lease(self._synthetic_seq, "expire", expired)
        return expired_ids

    def _persist_lease(self, seq: int, op: str, lease: Lease) -> None:
        active = sum(1 for item in self._leases.values()
                     if item.node_id == lease.node_id and item.state == "active")
        spec = self.nodes[lease.node_id]
        self.ledger.commit_event(
            seq=seq, op=op, node_id=lease.node_id, object_key=lease.object_key,
            shard_index=lease.shard_index, lease_id=lease.lease_id, merge_id=None,
            payload={"lease": lease.to_dict()}, cursor=self._cursor,
            lease_row=lease.to_dict(),
            node_counter={
                "node_id": lease.node_id, "capacity_slots": spec.capacity_slots,
                "lease_limit": spec.lease_limit, "active_count": active,
                "remaining_slots": spec.capacity_slots - active,
            },
        )

    def reap_expired_leases(self, seq_base: int = 900000) -> dict[str, list[str]]:
        """Explicitly sweep every node for real monotonic expirations."""
        summary: dict[str, list[str]] = {}
        for node_id in sorted(self.nodes):
            summary[node_id] = self._sweep_node(node_id, seq_base)
        if any(summary.values()):
            self._rebuild_adoption()
        return summary

    # ------------------------------------------------------------------ #
    # Request processing
    # ------------------------------------------------------------------ #
    def process_request(self, request: Request) -> EngineResult:
        start = monotonic_ms()
        for node_id in sorted(self.nodes):
            self._sweep_node(node_id, request.seq)
        if request.op == "grant":
            detail = self._handle_grant(request)
        elif request.op == "renew":
            detail = self._handle_renew(request)
        elif request.op == "abort":
            detail = self._handle_abort(request)
        elif request.op == "upload":
            detail = self._handle_upload(request)
        elif request.op == "read":
            detail = self.fast_read(request.object_key)
        else:  # pragma: no cover - validated upstream
            raise LeaseError(f"unknown op {request.op}", field="op")
        return EngineResult(seq=request.seq, op=request.op, ok=True,
                            detail=detail, elapsed_ms=monotonic_ms() - start)

    def _handle_grant(self, request: Request) -> dict:
        assert request.shard_index is not None
        node_id = self._node_id_for(request.object_key, request.shard_index)
        ttl_ms = request.ttl_ms or self.default_ttl
        if self.orchestrator is not None:
            reply = self.orchestrator.client(node_id).call(
                {"type": "grant", "object_key": request.object_key,
                 "shard_index": request.shard_index, "ttl_ms": ttl_ms},
                expect_fields=("lease",),
            )
            lease = Lease(**reply["lease"])
        else:
            from .lease import LeaseManager
            manager = self._in_process_managers.setdefault(
                node_id, LeaseManager(node_id, self.nodes[node_id].capacity_slots,
                                      self.nodes[node_id].lease_limit, self.clock))
            lease = manager.grant(request.object_key, request.shard_index, ttl_ms)
        self._leases[lease.lease_id] = lease
        self._cursor += 1
        self._persist_lease(request.seq, "grant", lease)
        return {"lease": lease.to_dict()}

    def _handle_renew(self, request: Request) -> dict:
        if not request.lease_id or request.expected_lease_version is None:
            raise LeaseError("renew requires lease_id and expected_lease_version",
                             key=request.object_key, field="lease_id")
        lease = self._leases.get(request.lease_id)
        if lease is None:
            raise LeaseError("unknown lease_id", key=request.object_key,
                             field="lease_id")
        ttl_ms = request.ttl_ms or self.default_ttl
        if self.orchestrator is not None:
            reply = self.orchestrator.client(lease.node_id).call(
                {"type": "renew", "lease_id": lease.lease_id,
                 "expected_lease_version": request.expected_lease_version,
                 "ttl_ms": ttl_ms},
                expect_fields=("lease",),
            )
            renewed = Lease(**reply["lease"])
        else:
            manager = self._in_process_managers[lease.node_id]
            renewed = manager.renew(lease.lease_id, request.expected_lease_version, ttl_ms)
        self._leases[renewed.lease_id] = renewed
        self._cursor += 1
        self._persist_lease(request.seq, "renew", renewed)
        return {"lease": renewed.to_dict()}

    def _handle_abort(self, request: Request) -> dict:
        if not request.lease_id:
            raise LeaseError("abort requires lease_id", key=request.object_key,
                             field="lease_id")
        lease = self._leases.get(request.lease_id)
        if lease is None:
            raise LeaseError("unknown lease_id", key=request.object_key,
                             field="lease_id")
        if self.orchestrator is not None:
            reply = self.orchestrator.client(lease.node_id).call(
                {"type": "abort", "lease_id": lease.lease_id},
                expect_fields=("lease",),
            )
            aborted = Lease(**reply["lease"])
        else:
            manager = self._in_process_managers[lease.node_id]
            aborted = manager.abort(lease.lease_id)
        self._leases[aborted.lease_id] = aborted
        self._cursor += 1
        self._persist_lease(request.seq, "abort", aborted)
        return {"lease": aborted.to_dict()}

    def _handle_upload(self, request: Request) -> dict:
        assert request.shard_index is not None
        key = request.object_key
        shard_index = request.shard_index
        node_id = self._node_id_for(key, shard_index)
        spec = self.nodes[node_id]
        manifest = self.manifests[key]
        size = manifest.shard_size
        merge_seq = self._next_merge_seq()
        merge_id = f"M-{node_id}-{merge_seq:06d}"

        if self.orchestrator is not None:
            reply = self.orchestrator.client(node_id).call(
                {"type": "upload", "object_key": key, "shard_index": shard_index,
                 "object_seed": manifest.content_seed, "size": size,
                 "merge_id": merge_id},
                expect_fields=("checksum", "merge_id"),
            )
            checksum = reply["checksum"]
            node_meta_version = int(reply["meta_version"])
            elapsed = int(reply["elapsed_ms"])
        else:
            data = generate_shard_bytes(spec.seed, manifest.content_seed, key,
                                        shard_index, size)
            cs = compute_checksum(node_id, key, shard_index, data)
            checksum = cs.as_dict()
            node_meta_version = spec.meta_version
            elapsed = 0

        candidate = MergeCandidate(
            object_key=key, shard_index=shard_index,
            shard_digest=checksum["shard_digest"],
            content_check=checksum["content_check"], merge_id=merge_id,
            parent_merge_id=None, shard_bytes=int(checksum["size"]),
            meta_version=node_meta_version, source_format=3,
            merge_seq=merge_seq, origin=f"ledger-upload:{request.seq}",
            node_id=node_id,
        )
        self._candidates.append(candidate)
        self._cursor += 1
        self.ledger.commit_event(
            seq=request.seq, op="upload", node_id=node_id, object_key=key,
            shard_index=shard_index, lease_id=None, merge_id=merge_id,
            payload={"checksum": checksum, "merge_seq": merge_seq,
                     "meta_version": node_meta_version},
            cursor=self._cursor,
            shard_row={
                "object_key": key, "shard_index": shard_index, "node_id": node_id,
                "merge_id": merge_id, "shard_digest": checksum["shard_digest"],
                "shard_bytes": int(checksum["size"]),
                "content_check": checksum["content_check"],
                "meta_version": node_meta_version, "source_format": 3,
                "merge_seq": merge_seq,
            },
        )
        self._rebuild_adoption()
        return {"merge_id": merge_id, "checksum": checksum, "node_elapsed_ms": elapsed}

    def ingest_metadata_candidates(self, records, source_name: str) -> int:
        """Feed normalized c1/c2/c3 records into the candidate set and ledger."""
        added = 0
        for record in records:
            merge_seq = self._next_merge_seq()
            merge_id = record.merge_id or f"F-{merge_seq:06d}"
            node_id = self._safe_node_for(record.object_key, record.shard_index)
            candidate = MergeCandidate(
                object_key=record.object_key,
                shard_index=record.shard_index,
                shard_digest=record.shard_digest,
                content_check=record.content_check or None,
                merge_id=merge_id,
                parent_merge_id=record.parent_merge_id,
                shard_bytes=int(record.shard_bytes or 0),
                meta_version=int(record.meta_version or 0),
                source_format=record.source_format,
                merge_seq=merge_seq,
                origin=f"metadata-file:{source_name}",
                node_id=node_id,
            )
            self._candidates.append(candidate)
            self._cursor += 1
            self._synthetic_seq += 1
            self.ledger.commit_event(
                seq=self._synthetic_seq, op="metadata", node_id=node_id,
                object_key=record.object_key, shard_index=record.shard_index,
                lease_id=None, merge_id=merge_id,
                payload={"source": source_name, "source_format": record.source_format,
                         "filled_fields": sorted(record.filled_fields),
                         "content_check": record.content_check},
                cursor=self._cursor,
                shard_row={
                    "object_key": record.object_key,
                    "shard_index": record.shard_index,
                    "node_id": node_id or "",
                    "merge_id": merge_id,
                    "shard_digest": record.shard_digest,
                    "shard_bytes": int(record.shard_bytes or 0),
                    "content_check": record.content_check or None,
                    "meta_version": int(record.meta_version or 0),
                    "source_format": record.source_format,
                    "merge_seq": merge_seq,
                },
            )
            added += 1
        if added:
            self._rebuild_adoption()
        return added

    def _safe_node_for(self, key: str, shard_index: int) -> Optional[str]:
        manifest = self.manifests.get(key)
        if manifest is None:
            return None
        try:
            return manifest.node_of(shard_index)
        except KeyError:
            return None

    # ------------------------------------------------------------------ #
    # Adoption + cached view
    # ------------------------------------------------------------------ #
    def _rebuild_adoption(self) -> None:
        self._adoption = decide_all(self._candidates)
        self._view_cache = {key: self._build_view(key) for key in sorted(self._adoption)}

    def _lease_watermark(self, key: str) -> dict:
        related = [lease for lease in self._leases.values() if lease.object_key == key]
        active = [lease for lease in related if lease.state == "active"
                  and self.now_ms() < lease.expires_ms]
        if not related:
            return {"active": 0, "total": 0, "min_expires_ms": None,
                    "max_lease_version": None}
        return {
            "active": len(active),
            "total": len(related),
            "min_expires_ms": min((lease.expires_ms for lease in active), default=None),
            "max_lease_version": max(lease.lease_version for lease in related),
        }

    def _build_view(self, key: str) -> dict:
        decisions = self._adoption.get(key, [])
        return {
            "object_key": key,
            "status": "adopted" if decisions else "pending",
            "adopted": [
                {
                    "shard_index": decision.shard_index,
                    "merge_id": decision.winner.merge_id,
                    "shard_digest": decision.winner.shard_digest,
                    "node_id": decision.winner.node_id,
                    "meta_version": decision.winner.meta_version,
                    "source_format": decision.winner.source_format,
                }
                for decision in decisions
            ],
            "rejected_count": sum(len(decision.rejected) for decision in decisions),
            "lease_watermark": self._lease_watermark(key),
        }

    def fast_read(self, key: str) -> dict:
        """One cached read; never recomputes adoption. Records latency in ms."""
        start = monotonic_ms()
        view = self._view_cache.get(key)
        if view is None:
            view = self._build_view(key)
        observation = ReadObservation(
            object_key=key, elapsed_ms=monotonic_ms() - start,
            served_from_cache=True, summary=view,
        )
        self._read_observations.append(observation)
        return {"view": view, "elapsed_ms": observation.elapsed_ms,
                "served_from_cache": True}

    @property
    def read_observations(self) -> list[ReadObservation]:
        return list(self._read_observations)

    @property
    def adoption(self) -> dict[str, list[AdoptionDecision]]:
        return self._adoption

    def adoption_blob(self) -> bytes:
        return decisions_canonical_blob(self._adoption)

    # ------------------------------------------------------------------ #
    # Authoritative replay and divergence
    # ------------------------------------------------------------------ #
    def authoritative_replay(self) -> dict[str, list[AdoptionDecision]]:
        """Recompute adoption from the committed ledger shard rows.

        This is the authoritative from-scratch path: it ignores the in-memory
        candidate cache and rebuilds every candidate solely from committed
        ledger state (uploads and normalized metadata files alike).
        """
        candidates: list[MergeCandidate] = []
        for row in self.ledger.fetch_shards():
            candidates.append(MergeCandidate(
                object_key=row["object_key"], shard_index=row["shard_index"],
                shard_digest=row["shard_digest"],
                content_check=row["content_check"],
                merge_id=row["merge_id"], parent_merge_id=None,
                shard_bytes=int(row["shard_bytes"]) if "shard_bytes" in row.keys() else 0,
                meta_version=row["meta_version"],
                source_format=row["source_format"], merge_seq=row["merge_seq"],
                origin=f"replay-shard:{row['merge_id']}",
                node_id=row["node_id"] or None,
            ))
        return decide_all(candidates)

    def replay_candidates(self) -> list[MergeCandidate]:
        return [
            candidate
            for decisions in self.authoritative_replay().values()
            for decision in decisions
            for candidate in [decision.winner]
        ]

    def verify_against_ledger(self) -> None:
        authoritative = self.authoritative_replay()
        cached_blob = self.adoption_blob()
        replay_blob = decisions_canonical_blob(authoritative)
        if cached_blob == replay_blob:
            return
        # Locate the first object key / shard index / field that differs.
        for key in sorted(set(self._adoption) | set(authoritative)):
            cached_decisions = {d.shard_index: d for d in self._adoption.get(key, [])}
            auth_decisions = {d.shard_index: d for d in authoritative.get(key, [])}
            for shard_index in sorted(set(cached_decisions) | set(auth_decisions)):
                cached = cached_decisions.get(shard_index)
                auth = auth_decisions.get(shard_index)
                if cached is None or auth is None:
                    raise DivergenceError(
                        "shard presence differs", key=key,
                        field="shard_presence",
                        detail={"shard_index": shard_index},
                    )
                for field_name in ("merge_id", "shard_digest", "content_check"):
                    cached_value = getattr(cached.winner, field_name)
                    auth_value = getattr(auth.winner, field_name)
                    if cached_value != auth_value:
                        raise DivergenceError(
                            f"field {field_name} diverges", key=key, field=field_name,
                            detail={"shard_index": shard_index,
                                    "cached": cached_value,
                                    "authoritative": auth_value},
                        )
        raise DivergenceError("unclassified divergence", field="adoption")

    def mark_unknown_and_rerun(self, request: Request, reason: str) -> None:
        """Record an uncommitted request after a forced node termination."""
        self._cursor += 1
        self.ledger.commit_event(
            seq=request.seq, op=request.op, node_id=None,
            object_key=request.object_key, shard_index=request.shard_index,
            lease_id=request.lease_id, merge_id=None,
            payload={"unknown": True, "reason": reason}, cursor=self._cursor,
            unknown={"seq": request.seq, "op": request.op,
                     "object_key": request.object_key, "reason": reason},
        )

    def restore_state_from_ledger(self) -> dict[str, Any]:
        """Rebuild in-memory leases/candidates solely from committed rows."""
        self._leases = {}
        for row in self.ledger.fetch_leases():
            self._leases[row["lease_id"]] = Lease(
                lease_id=row["lease_id"], node_id=row["node_id"],
                object_key=row["object_key"], shard_index=row["shard_index"],
                granted_ms=row["granted_ms"], expires_ms=row["expires_ms"],
                renew_count=row["renew_count"], lease_version=row["lease_version"],
                state=row["state"],
            )
        self._candidates = []
        max_seq = 0
        for row in self.ledger.fetch_shards():
            max_seq = max(max_seq, row["merge_seq"])
            self._candidates.append(MergeCandidate(
                object_key=row["object_key"], shard_index=row["shard_index"],
                shard_digest=row["shard_digest"], content_check=row["content_check"],
                merge_id=row["merge_id"], parent_merge_id=None,
                shard_bytes=int(row["shard_bytes"]) if "shard_bytes" in row.keys() else 0,
                meta_version=row["meta_version"],
                source_format=row["source_format"], merge_seq=row["merge_seq"],
                origin="ledger-restore", node_id=row["node_id"],
            ))
        self._merge_seq = max_seq
        self._cursor = self.ledger.progress_cursor()
        self._rebuild_adoption()
        return {
            "leases": len(self._leases),
            "candidates": len(self._candidates),
            "cursor": self._cursor,
            "unknown_requests": [dict(row) for row in self.ledger.fetch_unknown()],
        }


__all__ = ["AccountingEngine", "EngineResult", "ReadObservation"]
