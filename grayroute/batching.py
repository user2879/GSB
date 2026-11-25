"""Deterministic request-to-batch planning.

The plan is a pure function of the validated spec and the hit attribution.
Requests are grouped by worker and parameter version, ordered by arrival
sequence, then sliced by the revision batch size.  Batch ids are derived from
an explicit stable serialization of (worker, batch index, arrival seqs), so
arrival order on disk and batch execution order cannot change the partition or
the identifiers.
"""
from __future__ import annotations

from dataclasses import dataclass
from .canonical import canonical_bytes, short_hash
from .errors import VersionConflictError
from .receipt import BatchContext
from .sharding import Hit
from .spec import Request, Spec
from .inference import output_summary


@dataclass(frozen=True)
class Batch:
    batch_id: str
    worker_id: str
    backend_id: str
    rule_id: str
    rule_revision: int
    param_version_seq: int
    param_summary: str
    batch_index: int
    request_ids: tuple[str, ...]
    arrival_seqs: tuple[int, ...]

    def to_dict(self) -> dict:
        return {
            "batch_id": self.batch_id,
            "worker_id": self.worker_id,
            "backend_id": self.backend_id,
            "rule_id": self.rule_id,
            "rule_revision": self.rule_revision,
            "param_version_seq": self.param_version_seq,
            "param_summary": self.param_summary,
            "batch_index": self.batch_index,
            "request_ids": list(self.request_ids),
            "arrival_seqs": list(self.arrival_seqs),
        }


def _make_batch_id(worker_id: str, batch_index: int,
                   arrival_seqs: tuple[int, ...]) -> str:
    material = canonical_bytes({
        "worker_id": worker_id,
        "batch_index": batch_index,
        "arrival_seqs": list(arrival_seqs),
    })
    return f"bat-{short_hash(material, 20)}"


def plan_batches(spec: Spec, hits: dict[str, Hit]) -> tuple[Batch, ...]:
    # group key keeps hot-update versions strictly separate per worker
    groups: dict[tuple[str, int], list[Request]] = {}
    group_meta: dict[tuple[str, int], Hit] = {}
    for request in sorted(spec.requests, key=lambda r: r.arrival_seq):
        hit = hits[request.request_id]
        key = (hit.worker_id, hit.param_version_seq)
        groups.setdefault(key, []).append(request)
        group_meta.setdefault(key, hit)

    batches: list[Batch] = []
    for (worker_id, pv) in sorted(groups.keys()):
        requests = sorted(groups[(worker_id, pv)],
                          key=lambda r: r.arrival_seq)
        hit = group_meta[(worker_id, pv)]
        revision = spec.rules[hit.rule_id].revision_at(hit.revision)
        size = revision.batch_size
        index = 0
        for start in range(0, len(requests), size):
            members = requests[start:start + size]
            seqs = tuple(r.arrival_seq for r in members)
            rids = tuple(r.request_id for r in members)
            batch_id = _make_batch_id(worker_id, index, seqs)
            batches.append(Batch(
                batch_id=batch_id,
                worker_id=worker_id,
                backend_id=hit.backend_id,
                rule_id=hit.rule_id,
                rule_revision=hit.revision,
                param_version_seq=pv,
                param_summary=hit.param_summary,
                batch_index=index,
                request_ids=rids,
                arrival_seqs=seqs,
            ))
            index += 1
    return tuple(sorted(batches, key=lambda b: (b.worker_id,
                                                b.param_version_seq,
                                                b.batch_index)))


def assert_single_version(batch: Batch, version_seqs: set[int],
                          summaries: dict[int, str]) -> None:
    """Guard against mixed parameter versions inside one batch (E_VERSION)."""
    if len(version_seqs) > 1:
        ordered = sorted(version_seqs)
        raise VersionConflictError(
            "batch mixes multiple parameter versions",
            object_id=batch.batch_id, field="param_version_seq",
            versions=ordered,
            summaries={str(v): summaries.get(v) for v in ordered},
        )


def build_batch_context(spec: Spec, batch: Batch,
                        requests_by_id: dict[str, Request]) -> BatchContext:
    members = [requests_by_id[rid] for rid in
               sorted(batch.request_ids,
                      key=lambda rid: requests_by_id[rid].arrival_seq)]
    rank_by_request = {req.request_id: rank
                       for rank, req in enumerate(members)}
    backend = spec.backends[batch.backend_id]
    summaries = {
        req.request_id: output_summary(
            req.vector, spec.model_dim, backend.hidden_dim, batch.param_summary)
        for req in members
    }
    return BatchContext(
        batch_id=batch.batch_id,
        worker_id=batch.worker_id,
        param_version_seq=batch.param_version_seq,
        param_summary=batch.param_summary,
        rank_by_request=rank_by_request,
        request_ids=tuple(req.request_id for req in members),
        output_summary_by_request=summaries,
    )
