"""High-level import, query, consistency and dedup-validation services."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .audio import AudioDocument, decode_audio
from .config import FingerprintConfig
from .errors import DedupMergeError
from .fingerprint import (
    FingerprintSequence,
    extract_fingerprints,
    match_fingerprint_sequences,
    ranking_first_difference,
    verify_pair_consistency,
)
from .index import IndexStore, ImportOutcome, vote_histogram


@dataclass
class ImportedItem:
    relative_path: str
    document: AudioDocument
    sequence: FingerprintSequence
    outcome: ImportOutcome

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "source": self.document.source.to_dict(),
            "fingerprint": self.sequence.to_dict(),
            "index": self.outcome.to_dict(),
        }


class Library:
    def __init__(
        self,
        index_path: str | os.PathLike[str],
        config: FingerprintConfig,
        *,
        auto_migrate: bool = True,
    ) -> None:
        self.config = config
        self.store = IndexStore(index_path, config, auto_migrate=auto_migrate)

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> "Library":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def import_file(
        self,
        path: str | os.PathLike[str],
        *,
        root: str | os.PathLike[str] | None = None,
        expected: dict[str, Any] | None = None,
        display_label: str | None = None,
    ) -> ImportedItem:
        audio_path = Path(path)
        document = decode_audio(audio_path, self.config.sample_rate, expected=expected)
        sequence = extract_fingerprints(document.samples, self.config)
        if root is None:
            relative_path = audio_path.name
        else:
            try:
                relative_path = audio_path.resolve().relative_to(Path(root).resolve()).as_posix()
            except ValueError:
                relative_path = audio_path.name
        outcome = self.store.add_source(
            relative_path=relative_path,
            source_format=document.source.to_dict(),
            sequence=sequence,
            display_label=display_label,
        )
        return ImportedItem(
            relative_path=relative_path,
            document=document,
            sequence=sequence,
            outcome=outcome,
        )

    def query_file(
        self,
        path: str | os.PathLike[str],
        *,
        top_k: int = 10,
        expected: dict[str, Any] | None = None,
        search_variants: bool = True,
    ) -> dict[str, Any]:
        document = decode_audio(path, self.config.sample_rate, expected=expected)
        sequence = extract_fingerprints(document.samples, self.config)
        result = self.store.search(sequence, top_k=top_k, search_variants=search_variants)
        result["query_source"] = document.source.to_dict()
        return result

    def consistency_report(
        self,
        items: Iterable[tuple[str, str | os.PathLike[str]]],
        *,
        expected_by_label: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Compare fingerprints for known same-content local format variants."""

        decoded: list[tuple[str, FingerprintSequence, AudioDocument]] = []
        expected_by_label = expected_by_label or {}
        for label, path in items:
            document = decode_audio(
                path,
                self.config.sample_rate,
                expected=expected_by_label.get(label),
            )
            decoded.append(
                (label, extract_fingerprints(document.samples, self.config), document)
            )
        matches = []
        labels = {}
        pair_index = 0
        for left_index in range(len(decoded)):
            for right_index in range(left_index + 1, len(decoded)):
                left_label, left_sequence, _ = decoded[left_index]
                right_label, right_sequence, _ = decoded[right_index]
                pair_name = f"{left_label}__vs__{right_label}"
                labels[str(pair_index)] = pair_name
                matches.append(match_fingerprint_sequences(left_sequence, right_sequence))
                pair_index += 1
        report = verify_pair_consistency(matches, self.config, labels)
        report["sequences"] = [
            {
                "label": label,
                "source": document.source.to_dict(),
                "fingerprint": sequence.to_dict(),
            }
            for label, sequence, document in decoded
        ]
        return report

    def compare_query_rankings(
        self,
        queries: Iterable[tuple[str, str | os.PathLike[str]]],
        *,
        top_k: int = 10,
    ) -> dict[str, Any]:
        runs: list[dict[str, Any]] = []
        for label, path in queries:
            result = self.query_file(path, top_k=top_k)
            result["label"] = label
            runs.append(result)
        failure: dict[str, Any] | None = None
        if runs:
            baseline = runs[0]
            baseline_order = baseline["candidate_order"]
            for run in runs[1:]:
                difference = ranking_first_difference(
                    baseline_order,
                    run["candidate_order"],
                )
                if difference:
                    affected_group = difference["left"] or difference["right"]
                    distributions = []
                    for current in (baseline, run):
                        candidate = next(
                            (
                                item
                                for item in current["results"]
                                if item["group_id"] == affected_group
                            ),
                            None,
                        )
                        distributions.append(
                            {
                                "label": current["label"],
                                "rank": candidate["rank"] if candidate else None,
                                "vote_histogram": candidate["vote_histogram"] if candidate else [],
                            }
                        )
                    failure = {
                        "baseline": baseline["label"],
                        "other": run["label"],
                        "first_rank_difference": difference,
                        "affected_group_id": affected_group,
                        "vote_distributions": distributions,
                    }
                    break
        return {
            "ok": failure is None,
            "code": None if failure is None else "E_RANK_MISMATCH",
            "runs": runs,
            "failure": failure,
        }

    def validate_dedup_truth(self, truth_by_relative_path: dict[str, str]) -> dict[str, Any]:
        groups = self.store.list_groups()
        merged_errors: list[dict[str, Any]] = []
        for group in groups:
            truth_ids = sorted(
                {
                    truth_by_relative_path[source["relative_path"]]
                    for source in group["sources"]
                    if source["relative_path"] in truth_by_relative_path
                }
            )
            if len(truth_ids) > 1:
                source_ids = [source["source_id"] for source in group["sources"]]
                evidence: list[dict[str, Any]] = []
                for left_index in range(len(source_ids)):
                    for right_index in range(left_index + 1, len(source_ids)):
                        left = self.store.sequence_for_source(source_ids[left_index])
                        right = self.store.sequence_for_source(source_ids[right_index])
                        vote = vote_histogram(left.fingerprints, right.fingerprints)
                        evidence.append(
                            {
                                "left_source_id": source_ids[left_index],
                                "right_source_id": source_ids[right_index],
                                "dedup_score": round(vote.score, 6),
                                "votes": vote.votes,
                                "offset_frames": vote.offset_frames,
                            }
                        )
                merged_errors.append(
                    {
                        "group_id": group["group_id"],
                        "truth_ids": truth_ids,
                        "sources": group["sources"],
                        "fingerprint_evidence": sorted(
                            evidence,
                            key=lambda item: (-item["dedup_score"], item["left_source_id"]),
                        ),
                    }
                )
        if merged_errors:
            raise DedupMergeError(
                "Dedup merged sources from distinct truth tracks",
                {"merged_groups": merged_errors},
            )
        return {
            "ok": True,
            "groups": len(groups),
            "sources": sum(len(group["sources"]) for group in groups),
            "merged_sources": sum(
                max(0, len(group["sources"]) - 1)
                for group in groups
                if len(
                    {
                        truth_by_relative_path[source["relative_path"]]
                        for source in group["sources"]
                        if source["relative_path"] in truth_by_relative_path
                    }
                )
                == 1
            ),
        }

    def migration_snapshot(self) -> dict[str, Any]:
        groups = self.store.list_groups()
        canonical = [
            {
                "group_id": group["group_id"],
                "display_label": group["display_label"],
                "sources": [
                    {
                        "source_id": source["source_id"],
                        "relative_path": source["relative_path"],
                        "fingerprint_count": source["fingerprint_count"],
                    }
                    for source in group["sources"]
                ],
            }
            for group in groups
        ]
        return {
            "metadata": self.store.metadata(),
            "stats": self.store.stats(),
            "groups": canonical,
        }
