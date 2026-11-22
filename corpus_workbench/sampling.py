"""Deterministic source-weight allocation and weighted sampling records."""

from __future__ import annotations

import math
from collections import defaultdict

from .codes import E_SAMPLE_FAILED
from .config import BalanceConfig
from .hashing import StableRandom, source_seed
from .models import Document, Failure, SampleRecord


def allocate_source_counts(
    weights: dict[str, float],
    target: int,
    available_sources: set[str],
) -> dict[str, int]:
    eligible = sorted(source for source in weights if source in available_sources and weights[source] > 0)
    if not eligible or target <= 0:
        return {}
    total_weight = sum(weights[source] for source in eligible)
    exact = {source: target * weights[source] / total_weight for source in eligible}
    counts = {source: int(math.floor(exact[source])) for source in eligible}
    remaining = target - sum(counts.values())
    order = sorted(eligible, key=lambda source: (-(exact[source] - counts[source]), source))
    for source in order[:remaining]:
        counts[source] += 1
    return dict(sorted(counts.items()))


def sample_documents(
    documents: list[Document],
    config: BalanceConfig,
) -> tuple[list[SampleRecord], list[Failure]]:
    by_source: dict[str, list[Document]] = defaultdict(list)
    for document in documents:
        by_source[document.source].append(document)
    for source in by_source:
        by_source[source].sort(key=lambda document: document.doc_id)

    counts = allocate_source_counts(config.source_weights, config.target_sample_count, set(by_source))
    records: list[SampleRecord] = []
    failures: list[Failure] = []

    for source in sorted(counts):
        documents_for_source = by_source[source]
        rng = StableRandom(source_seed(source, config.sampling_salt))
        for sample_index in range(counts[source]):
            try:
                document = documents_for_source[rng.randbelow(len(documents_for_source))]
                records.append(
                    SampleRecord(
                        source=source,
                        doc_id=document.doc_id,
                        sample_index=sample_index,
                        order=-1,
                    )
                )
            except Exception as exc:  # pragma: no cover - defensive fault isolation
                failures.append(
                    Failure(
                        source=source,
                        doc_id="",
                        stage="sample",
                        code=E_SAMPLE_FAILED,
                        message=f"sampling failed at source sequence {sample_index}: {exc}",
                    )
                )

    records.sort(key=lambda record: (record.source, record.sample_index))
    ordered = [
        SampleRecord(
            source=record.source,
            doc_id=record.doc_id,
            sample_index=record.sample_index,
            order=order,
        )
        for order, record in enumerate(records)
    ]
    failures.sort(key=Failure.sort_key)
    return ordered, failures
