from __future__ import annotations

from typing import Any

from .canonical import fingerprint
from .errors import BusinessError, E_DUPLICATE_DELIVERY, E_LEDGER_FROZEN, E_LEDGER_MISMATCH, E_STALE_DIGEST


def receipt_content_digest(receipt: dict[str, Any]) -> str:
    payload = {
        key: value
        for key, value in receipt.items()
        if key not in {"receipt_digest", "elapsed_monotonic_ms"}
    }
    return fingerprint(payload)


def verify_receipt_digest(receipt: dict[str, Any]) -> bool:
    supplied = receipt.get("receipt_digest")
    return isinstance(supplied, str) and receipt_content_digest(receipt) == supplied


class ReceiptLedger:
    def __init__(self, *, term_ids: list[int], topic_count: int) -> None:
        self.term_ids = sorted(int(term_id) for term_id in term_ids)
        self.topic_count = int(topic_count)
        self.receipts: dict[str, dict[str, Any]] = {}
        self.receipt_order: list[str] = []
        self.rejections: list[dict[str, Any]] = []
        self.frozen_topics: set[int] = set()
        self.distribution: dict[tuple[int, int], int] = {
            (term_id, topic_id): 0
            for term_id in self.term_ids
            for topic_id in range(self.topic_count)
        }
        self.authority: list[dict[str, Any]] = []
        self.overwrites: list[dict[str, Any]] = []
        self.closure: list[dict[str, Any]] = []

    def reject(
        self,
        *,
        batch_id: str | None,
        shard_id: str | None,
        code: str,
        reason: str,
        **details: Any,
    ) -> BusinessError:
        record = {
            "batch_id": batch_id,
            "shard_id": shard_id,
            "code": code,
            "reason": reason,
            "details": details,
        }
        self.rejections.append(record)
        return BusinessError(code, reason, batch_id=batch_id, details=details or None)

    def accept_receipt(
        self,
        receipt: dict[str, Any],
        *,
        batch_id: str,
        prior_receipt_digest: str | None = None,
    ) -> None:
        shard_id = str(receipt["shard_id"])
        current = self.receipts.get(shard_id)
        if current is not None:
            current_digest = current["receipt_digest"]
            if prior_receipt_digest is None:
                raise self.reject(
                    batch_id=batch_id,
                    shard_id=shard_id,
                    code=E_DUPLICATE_DELIVERY,
                    reason="duplicate shard receipt requires prior_receipt_digest",
                    current_receipt_digest=current_digest,
                )
            if prior_receipt_digest != current_digest:
                raise self.reject(
                    batch_id=batch_id,
                    shard_id=shard_id,
                    code=E_STALE_DIGEST,
                    reason="prior_receipt_digest does not match ledger receipt",
                    current_receipt_digest=current_digest,
                    supplied_digest=prior_receipt_digest,
                )
        if not verify_receipt_digest(receipt):
            raise self.reject(
                batch_id=batch_id,
                shard_id=shard_id,
                code="E_WORKER_PROTOCOL",
                reason="receipt digest verification failed",
                supplied_digest=receipt.get("receipt_digest"),
                computed_digest=receipt_content_digest(receipt),
            )

        frozen_touch = sorted(
            {
                int(entry["topic_id"])
                for entry in receipt.get("term_topic_counts", [])
                if int(entry.get("count", 0)) != 0 and int(entry["topic_id"]) in self.frozen_topics
            }
        )
        if frozen_touch:
            raise self.reject(
                batch_id=batch_id,
                shard_id=shard_id,
                code=E_LEDGER_FROZEN,
                reason="topic is frozen after ledger closure mismatch",
                frozen_topics=frozen_touch,
            )

        if current is None:
            self.receipt_order.append(shard_id)
        self.receipts[shard_id] = receipt

    @staticmethod
    def _source_ref(receipt: dict[str, Any]) -> dict[str, Any]:
        return {
            "shard_id": receipt["shard_id"],
            "node_id": receipt["node_id"],
            "topic_parameter_version": int(receipt["topic_parameter_version"]),
            "monotonic_ms": int(receipt["monotonic_ms"]),
            "receipt_digest": receipt["receipt_digest"],
        }

    @staticmethod
    def _rank_key(source: dict[str, Any]) -> tuple[int, int, str, str, str]:
        return (
            -int(source["topic_parameter_version"]),
            -int(source["monotonic_ms"]),
            str(source["node_id"]),
            str(source["shard_id"]),
            str(source["receipt_digest"]),
        )

    def _term_sources(self, term_id: int) -> list[tuple[dict[str, Any], dict[int, int]]]:
        sources: list[tuple[dict[str, Any], dict[int, int]]] = []
        for shard_id in sorted(self.receipts):
            receipt = self.receipts[shard_id]
            vector = {topic_id: 0 for topic_id in range(self.topic_count)}
            present = False
            for entry in receipt.get("term_topic_counts", []):
                if int(entry["term_id"]) == term_id:
                    vector[int(entry["topic_id"])] = int(entry["count"])
                    present = True
            if present:
                sources.append((self._source_ref(receipt), vector))
        sources.sort(key=lambda item: self._rank_key(item[0]))
        return sources

    def rebuild(self) -> dict[str, Any]:
        authority: list[dict[str, Any]] = []
        overwrites: list[dict[str, Any]] = []
        rebuilt_distribution = {
            (term_id, topic_id): 0
            for term_id in self.term_ids
            for topic_id in range(self.topic_count)
        }
        source_sums = [0 for _ in range(self.topic_count)]

        for term_id in self.term_ids:
            sources = self._term_sources(term_id)
            if not sources:
                continue
            winner, winner_vector = sources[0]
            covered: list[dict[str, Any]] = []
            for rank, (source, vector) in enumerate(sources[1:], start=2):
                covered.append(
                    {
                        "rank": rank,
                        "source": source,
                        "topic_counts": [vector[topic_id] for topic_id in range(self.topic_count)],
                    }
                )
                overwrite = {
                    "term_id": term_id,
                    "covered_source": source,
                    "covered_topic_counts": [vector[t] for t in range(self.topic_count)],
                    "authoritative_source": winner,
                    "rule": "topic_parameter_version_desc,monotonic_ms_desc,node_id_asc",
                }
                overwrites.append(overwrite)
            authority.append(
                {
                    "term_id": term_id,
                    "authoritative_source": winner,
                    "authoritative_topic_counts": [
                        winner_vector[topic_id] for topic_id in range(self.topic_count)
                    ],
                    "covered_sources": covered,
                }
            )
            for topic_id in range(self.topic_count):
                rebuilt_distribution[(term_id, topic_id)] = winner_vector[topic_id]
                source_sums[topic_id] += winner_vector[topic_id]

        for key, value in rebuilt_distribution.items():
            term_id, topic_id = key
            if topic_id not in self.frozen_topics:
                self.distribution[key] = value

        self.authority = authority
        self.overwrites = overwrites
        return self.verify_closure(source_sums=source_sums)

    def verify_closure(self, *, source_sums: list[int] | None = None) -> dict[str, Any]:
        if source_sums is None:
            source_sums = [0 for _ in range(self.topic_count)]
            for decision in self.authority:
                counts = decision["authoritative_topic_counts"]
                for topic_id, count in enumerate(counts):
                    source_sums[topic_id] += int(count)

        closure: list[dict[str, Any]] = []
        mismatches: list[int] = []
        for topic_id in range(self.topic_count):
            distribution_sum = sum(
                int(self.distribution.get((term_id, topic_id), 0)) for term_id in self.term_ids
            )
            authoritative_sum = int(source_sums[topic_id])
            closed = distribution_sum == authoritative_sum
            if not closed:
                mismatches.append(topic_id)
            closure.append(
                {
                    "topic_id": topic_id,
                    "distribution_sum": distribution_sum,
                    "authoritative_source_sum": authoritative_sum,
                    "closed": closed,
                    "frozen": topic_id in self.frozen_topics,
                }
            )
        if mismatches:
            self.frozen_topics.update(mismatches)
        self.closure = closure
        return {
            "all_closed": not mismatches,
            "mismatched_topics": mismatches,
            "frozen_topics": sorted(self.frozen_topics),
            "by_topic": closure,
        }

    def distribution_entries(self) -> list[dict[str, int]]:
        return [
            {"term_id": term_id, "topic_id": topic_id, "count": int(self.distribution[(term_id, topic_id)])}
            for term_id in self.term_ids
            for topic_id in range(self.topic_count)
        ]

    def audit(self) -> dict[str, Any]:
        closure_result = self.verify_closure()
        return {
            "receipt_order": list(self.receipt_order),
            "receipts": {shard_id: self.receipts[shard_id] for shard_id in sorted(self.receipts)},
            "authority": self.authority,
            "overwrites": self.overwrites,
            "global_distribution": self.distribution_entries(),
            "closure": closure_result,
            "rejections": list(self.rejections),
            "frozen_topics": sorted(self.frozen_topics),
        }
