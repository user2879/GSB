"""Persistent SQLite fingerprint index with versioning and crash-safe updates."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from .config import INDEX_SCHEMA_VERSION, FingerprintConfig, config_from_json
from .errors import IndexVersionError
from .fingerprint import Fingerprint, FingerprintSequence, counter_json


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _short_hash(value: str, length: int = 20) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


@dataclass
class ImportOutcome:
    source_id: str
    group_id: str
    deduplicated: bool
    exact_duplicate: bool
    dedup_score: float
    aligned_offset_frames: int
    fingerprint_count: int
    revision: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "group_id": self.group_id,
            "deduplicated": self.deduplicated,
            "exact_duplicate": self.exact_duplicate,
            "dedup_score": round(self.dedup_score, 6),
            "aligned_offset_frames": self.aligned_offset_frames,
            "fingerprint_count": self.fingerprint_count,
            "revision": self.revision,
        }


@dataclass
class VoteEvidence:
    votes: int
    offset_frames: int
    score: float
    histogram: Counter[int]

    def to_dict(self, frames_per_second: float) -> dict[str, Any]:
        return {
            "votes": self.votes,
            "offset_frames": self.offset_frames,
            "offset_seconds": round(self.offset_frames / frames_per_second, 6),
            "score": round(self.score, 6),
            "histogram": [
                {
                    "offset_frames": offset,
                    "offset_seconds": round(offset / frames_per_second, 6),
                    "votes": votes,
                }
                for offset, votes in sorted(
                    self.histogram.items(), key=lambda item: (-item[1], item[0])
                )[:16]
            ],
        }


class IndexStore:
    """A small, deterministic inverted index stored in one SQLite database."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        config: FingerprintConfig,
        *,
        auto_migrate: bool = True,
        recover_orphans: bool = True,
    ) -> None:
        self.path = Path(path)
        self.config = config
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.recovered_files: list[str] = []
        if recover_orphans:
            self.recovered_files = self._remove_orphan_temp_files()
        self.connection = sqlite3.connect(str(self.path), timeout=30.0)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA journal_mode=DELETE")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA temp_store=MEMORY")
        self.connection.execute("PRAGMA page_size=4096")
        if self._is_new_database():
            self._initialize_schema()
        else:
            self._check_integrity()
            self._remove_recovered_journal()
            schema_version = self._read_meta("schema_version")
            if int(schema_version) != INDEX_SCHEMA_VERSION:
                raise IndexVersionError(
                    "Index schema version is not supported",
                    {
                        "index_schema_version": schema_version,
                        "supported_schema_version": INDEX_SCHEMA_VERSION,
                    },
                )
            self._ensure_config(auto_migrate=auto_migrate)

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "IndexStore":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _is_new_database(self) -> bool:
        row = self.connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='meta'"
        ).fetchone()
        return row is None

    def _initialize_schema(self) -> None:
        with self.transaction():
            self.connection.executescript(
                """
                CREATE TABLE meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE work_groups (
                    group_id TEXT PRIMARY KEY,
                    display_label TEXT NOT NULL,
                    fingerprint_count INTEGER NOT NULL,
                    duration REAL NOT NULL
                );
                CREATE TABLE sources (
                    source_id TEXT PRIMARY KEY,
                    group_id TEXT NOT NULL REFERENCES work_groups(group_id) ON DELETE CASCADE,
                    relative_path TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    fingerprint_content_hash TEXT NOT NULL,
                    container TEXT NOT NULL,
                    codec TEXT NOT NULL,
                    sample_rate INTEGER NOT NULL,
                    bit_depth INTEGER,
                    channels INTEGER NOT NULL,
                    declared_frames INTEGER NOT NULL,
                    decoded_frames INTEGER NOT NULL,
                    duration REAL NOT NULL,
                    format_signature TEXT NOT NULL,
                    fingerprint_count INTEGER NOT NULL,
                    import_sequence INTEGER NOT NULL,
                    format_json TEXT NOT NULL
                );
                CREATE TABLE fingerprints (
                    id INTEGER PRIMARY KEY,
                    fingerprint_hash TEXT NOT NULL,
                    source_id TEXT NOT NULL REFERENCES sources(source_id) ON DELETE CASCADE,
                    group_id TEXT NOT NULL REFERENCES work_groups(group_id) ON DELETE CASCADE,
                    anchor_frame INTEGER NOT NULL,
                    target_frame INTEGER NOT NULL,
                    band1 INTEGER NOT NULL,
                    band2 INTEGER NOT NULL,
                    delta_frames INTEGER NOT NULL
                );
                CREATE INDEX idx_fingerprints_hash ON fingerprints(fingerprint_hash, source_id, anchor_frame);
                CREATE INDEX idx_fingerprints_source ON fingerprints(source_id);
                CREATE INDEX idx_fingerprints_group ON fingerprints(group_id);
                CREATE INDEX idx_sources_group ON sources(group_id);
                """
            )
            meta = {
                "schema_version": str(INDEX_SCHEMA_VERSION),
                "fingerprint_version": self.config.version,
                "config_digest": self.config.digest(),
                "encoding_space": self.config.encoding_space,
                "config_json": self.config.canonical_json(),
                "revision": "0",
                "content_hash": self.logical_content_hash(),
            }
            self.connection.executemany(
                "INSERT INTO meta(key, value) VALUES (?, ?)",
                sorted(meta.items()),
            )
            self.connection.execute(f"PRAGMA user_version={INDEX_SCHEMA_VERSION}")

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            self.connection.execute("BEGIN IMMEDIATE")
            yield self.connection
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

    def _check_integrity(self) -> None:
        result = self.connection.execute("PRAGMA integrity_check").fetchone()
        if not result or result[0] != "ok":
            raise IndexVersionError(
                "Index integrity check failed after crash recovery",
                {"path": self.path.name, "integrity_check": result[0] if result else None},
            )

    def _remove_orphan_temp_files(self) -> list[str]:
        removed: list[str] = []
        patterns = [
            f"{self.path.name}.tmp-*",
            f"{self.path.name}.orphan-*",
            f".{self.path.name}.tmp-*",
        ]
        for pattern in patterns:
            for candidate in self.path.parent.glob(pattern):
                if candidate.is_file():
                    candidate.unlink(missing_ok=True)
                    removed.append(candidate.name)
        return sorted(removed)

    def _remove_recovered_journal(self) -> None:
        journal = Path(str(self.path) + "-journal")
        if journal.is_file() and journal.stat().st_size > 0:
            # Reaching here means integrity_check opened and recovered the
            # database.  SQLite leaves no live transaction in this process, so
            # the now-inert rollback journal can be reclaimed deterministically.
            journal.unlink(missing_ok=True)
            self.recovered_files.append(journal.name)
            self.recovered_files.sort()

    def metadata(self) -> dict[str, str]:
        rows = self.connection.execute("SELECT key, value FROM meta ORDER BY key").fetchall()
        return {row["key"]: row["value"] for row in rows}

    def _read_meta(self, key: str) -> str:
        row = self.connection.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        if row is None:
            raise IndexVersionError(
                "Index metadata is incomplete",
                {"path": self.path.name, "missing_key": key},
            )
        return str(row["value"])

    def _set_meta(self, values: dict[str, str]) -> None:
        self.connection.executemany(
            "INSERT INTO meta(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            sorted(values.items()),
        )

    def _ensure_config(self, auto_migrate: bool) -> None:
        old_digest = self._read_meta("config_digest")
        if old_digest == self.config.digest():
            return
        old_config = config_from_json(self._read_meta("config_json"))
        if not auto_migrate:
            raise IndexVersionError(
                "Index fingerprint version does not match the requested configuration",
                {
                    "index_version": old_config.version,
                    "requested_version": self.config.version,
                    "index_config_digest": old_digest,
                    "requested_config_digest": self.config.digest(),
                    "migratable": self.config.migration_compatible_from(old_config),
                },
            )
        if not self.config.migration_compatible_from(old_config):
            raise IndexVersionError(
                "Index extraction parameters are incompatible; full re-extraction is required",
                {
                    "index_version": old_config.version,
                    "requested_version": self.config.version,
                    "index_encoding_space": old_config.encoding_space,
                    "requested_encoding_space": self.config.encoding_space,
                    "reason": "stft_or_peak_configuration_changed",
                },
            )
        self._migrate_encoding(old_config)

    def _migrate_encoding(self, old_config: FingerprintConfig) -> None:
        rows = self.connection.execute(
            "SELECT id, band1, band2, delta_frames FROM fingerprints ORDER BY id"
        ).fetchall()
        updates = [
            (
                self.config.hash_for(row["band1"], row["band2"], row["delta_frames"]),
                row["id"],
            )
            for row in rows
        ]
        with self.transaction():
            if updates:
                self.connection.executemany(
                    "UPDATE fingerprints SET fingerprint_hash=? WHERE id=?",
                    updates,
                )
            source_ids = [
                str(row[0])
                for row in self.connection.execute(
                    "SELECT source_id FROM sources ORDER BY source_id"
                )
            ]
            for source_id in source_ids:
                fingerprint_rows = self.connection.execute(
                    "SELECT fingerprint_hash, anchor_frame, target_frame, band1, band2, delta_frames "
                    "FROM fingerprints WHERE source_id=? "
                    "ORDER BY anchor_frame, target_frame, band1, band2, fingerprint_hash",
                    (source_id,),
                ).fetchall()
                sequence_digest = hashlib.sha256()
                sequence_digest.update(self.config.digest().encode("ascii"))
                for row in fingerprint_rows:
                    sequence_digest.update(
                        (
                            "|".join(str(row[key]) for key in row.keys()) + "\n"
                        ).encode("ascii")
                    )
                self.connection.execute(
                    "UPDATE sources SET fingerprint_content_hash=? WHERE source_id=?",
                    (sequence_digest.hexdigest(), source_id),
                )
            revision = int(self._read_meta("revision")) + 1
            self._set_meta(
                {
                    "fingerprint_version": self.config.version,
                    "config_digest": self.config.digest(),
                    "encoding_space": self.config.encoding_space,
                    "config_json": self.config.canonical_json(),
                    "revision": str(revision),
                }
            )
            content_hash = self.logical_content_hash()
            self._set_meta({"content_hash": content_hash})
        self.connection.execute(f"PRAGMA user_version={INDEX_SCHEMA_VERSION}")

    def logical_content_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.config.digest().encode("ascii"))
        for row in self.connection.execute(
            "SELECT group_id, display_label, fingerprint_count, duration "
            "FROM work_groups ORDER BY group_id"
        ):
            digest.update(
                ("G|" + "|".join(str(row[key]) for key in row.keys()) + "\n").encode("utf-8")
            )
        for row in self.connection.execute(
            "SELECT source_id, group_id, relative_path, content_sha256, "
            "fingerprint_content_hash, container, codec, sample_rate, bit_depth, "
            "channels, declared_frames, decoded_frames, duration, format_signature, "
            "fingerprint_count, import_sequence FROM sources ORDER BY source_id"
        ):
            digest.update(
                ("S|" + "|".join(str(row[key]) for key in row.keys()) + "\n").encode("utf-8")
            )
        for row in self.connection.execute(
            "SELECT fingerprint_hash, source_id, group_id, anchor_frame, target_frame, "
            "band1, band2, delta_frames FROM fingerprints "
            "ORDER BY group_id, source_id, fingerprint_hash, anchor_frame, target_frame, band1, band2, delta_frames"
        ):
            digest.update(
                ("F|" + "|".join(str(row[key]) for key in row.keys()) + "\n").encode("utf-8")
            )
        return digest.hexdigest()

    def _bump_revision(self) -> int:
        revision = int(self._read_meta("revision")) + 1
        content_hash = self.logical_content_hash()
        self._set_meta({"revision": str(revision), "content_hash": content_hash})
        return revision

    def _next_import_sequence(self) -> int:
        row = self.connection.execute("SELECT COALESCE(MAX(import_sequence), 0) + 1 FROM sources").fetchone()
        return int(row[0])

    def _all_source_sequences(self) -> list[tuple[str, str, FingerprintSequence]]:
        result: list[tuple[str, str, FingerprintSequence]] = []
        for source in self.connection.execute(
            "SELECT source_id, group_id, duration, fingerprint_count FROM sources ORDER BY import_sequence, source_id"
        ):
            sequence = self.sequence_for_source(source["source_id"])
            result.append((source["source_id"], source["group_id"], sequence))
        return result

    def sequence_for_source(self, source_id: str) -> FingerprintSequence:
        rows = self.connection.execute(
            "SELECT fingerprint_hash, anchor_frame, target_frame, band1, band2, delta_frames "
            "FROM fingerprints WHERE source_id=? "
            "ORDER BY anchor_frame, target_frame, band1, band2, fingerprint_hash",
            (source_id,),
        ).fetchall()
        source = self.connection.execute(
            "SELECT duration FROM sources WHERE source_id=?", (source_id,)
        ).fetchone()
        if source is None:
            raise KeyError(source_id)
        fingerprints = [
            Fingerprint(
                hash=row["fingerprint_hash"],
                anchor_frame=row["anchor_frame"],
                target_frame=row["target_frame"],
                band1=row["band1"],
                band2=row["band2"],
                delta_frames=row["delta_frames"],
            )
            for row in rows
        ]
        return FingerprintSequence(
            config=self.config,
            duration=float(source["duration"]),
            peaks=[],
            fingerprints=fingerprints,
        )

    def _find_matching_group(
        self,
        sequence: FingerprintSequence,
    ) -> tuple[str | None, float, int]:
        group_scores: dict[str, float] = {}
        group_offsets: dict[str, int] = {}
        for source_id, group_id, existing in self._all_source_sequences():
            evidence = vote_histogram(sequence.fingerprints, existing.fingerprints)
            if evidence.score > group_scores.get(group_id, -1.0):
                group_scores[group_id] = evidence.score
                group_offsets[group_id] = evidence.offset_frames
        if not group_scores:
            return None, 0.0, 0
        group_id = sorted(group_scores, key=lambda key: (-group_scores[key], key))[0]
        score = group_scores[group_id]
        if score < self.config.dedup_score_threshold:
            return None, score, group_offsets[group_id]
        return group_id, score, group_offsets[group_id]

    def add_source(
        self,
        *,
        relative_path: str,
        source_format: dict[str, Any],
        sequence: FingerprintSequence,
        display_label: str | None = None,
    ) -> ImportOutcome:
        content_sha = str(source_format["sha256"])
        format_signature = str(source_format["format_signature"])
        source_material = "|".join(
            [
                content_sha,
                format_signature,
            ]
        )
        source_id = "src_" + _short_hash(source_material)
        existing = self.connection.execute(
            "SELECT group_id, fingerprint_count, fingerprint_content_hash FROM sources WHERE source_id=?",
            (source_id,),
        ).fetchone()
        if existing is not None and existing["fingerprint_content_hash"] == sequence.content_hash():
            return ImportOutcome(
                source_id=source_id,
                group_id=str(existing["group_id"]),
                deduplicated=True,
                exact_duplicate=True,
                dedup_score=1.0,
                aligned_offset_frames=0,
                fingerprint_count=int(existing["fingerprint_count"]),
                revision=int(self._read_meta("revision")),
            )

        with self.transaction():
            group_id, dedup_score, offset = self._find_matching_group(sequence)
            new_group = group_id is None
            if group_id is None:
                group_id = "grp_" + _short_hash(source_id)
                existing_group = self.connection.execute(
                    "SELECT group_id FROM work_groups WHERE group_id=?", (group_id,)
                ).fetchone()
                if existing_group is not None:
                    group_id = "grp_" + _short_hash(group_id + "|" + relative_path)
            if new_group:
                self.connection.execute(
                    "INSERT INTO work_groups(group_id, display_label, fingerprint_count, duration) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        group_id,
                        display_label or Path(relative_path).stem,
                        len(sequence.fingerprints),
                        sequence.duration,
                    ),
                )
            else:
                self.connection.execute(
                    "UPDATE work_groups SET fingerprint_count=fingerprint_count+?, "
                    "duration=MAX(duration, ?) WHERE group_id=?",
                    (len(sequence.fingerprints), sequence.duration, group_id),
                )

            import_sequence = self._next_import_sequence()
            self.connection.execute(
                "INSERT INTO sources(source_id, group_id, relative_path, content_sha256, "
                "fingerprint_content_hash, container, codec, sample_rate, bit_depth, channels, "
                "declared_frames, decoded_frames, duration, format_signature, fingerprint_count, "
                "import_sequence, format_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    source_id,
                    group_id,
                    relative_path,
                    content_sha,
                    sequence.content_hash(),
                    str(source_format["container"]),
                    str(source_format["codec"]),
                    int(source_format["sample_rate"]),
                    source_format.get("bit_depth"),
                    int(source_format["channels"]),
                    int(source_format["declared_frames"]),
                    int(source_format["decoded_frames"]),
                    float(source_format["duration"]),
                    format_signature,
                    len(sequence.fingerprints),
                    import_sequence,
                    _json(source_format),
                ),
            )
            self.connection.executemany(
                "INSERT INTO fingerprints(fingerprint_hash, source_id, group_id, anchor_frame, "
                "target_frame, band1, band2, delta_frames) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        item.hash,
                        source_id,
                        group_id,
                        item.anchor_frame,
                        item.target_frame,
                        item.band1,
                        item.band2,
                        item.delta_frames,
                    )
                    for item in sequence.fingerprints
                ],
            )
            revision = self._bump_revision()
        return ImportOutcome(
            source_id=source_id,
            group_id=group_id,
            deduplicated=not new_group,
            exact_duplicate=False,
            dedup_score=dedup_score,
            aligned_offset_frames=offset,
            fingerprint_count=len(sequence.fingerprints),
            revision=revision,
        )

    def _source_rows_for_group(self, group_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT source_id, relative_path, container, codec, sample_rate, bit_depth, "
            "channels, duration, format_signature, fingerprint_count, content_sha256 "
            "FROM sources WHERE group_id=? ORDER BY source_id",
            (group_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def list_groups(self) -> list[dict[str, Any]]:
        groups = self.connection.execute(
            "SELECT group_id, display_label, fingerprint_count, duration "
            "FROM work_groups ORDER BY group_id"
        ).fetchall()
        return [
            {
                "group_id": row["group_id"],
                "display_label": row["display_label"],
                "fingerprint_count": row["fingerprint_count"],
                "duration": round(float(row["duration"]), 9),
                "sources": self._source_rows_for_group(row["group_id"]),
            }
            for row in groups
        ]

    def group_details(self, group_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT group_id, display_label, fingerprint_count, duration "
            "FROM work_groups WHERE group_id=?",
            (group_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "group_id": row["group_id"],
            "display_label": row["display_label"],
            "fingerprint_count": row["fingerprint_count"],
            "duration": round(float(row["duration"]), 9),
            "sources": self._source_rows_for_group(group_id),
        }

    def _fetch_occurrences(self, hashes: list[str]) -> list[sqlite3.Row]:
        unique_hashes = sorted(set(hashes))
        rows: list[sqlite3.Row] = []
        for start in range(0, len(unique_hashes), 500):
            chunk = unique_hashes[start : start + 500]
            placeholders = ",".join("?" for _ in chunk)
            rows.extend(
                self.connection.execute(
                    "SELECT f.fingerprint_hash, f.source_id, f.group_id, f.anchor_frame "
                    f"FROM fingerprints f WHERE f.fingerprint_hash IN ({placeholders}) "
                    "ORDER BY f.fingerprint_hash, f.source_id, f.anchor_frame",
                    chunk,
                ).fetchall()
            )
        return rows

    def search(
        self,
        query: FingerprintSequence,
        top_k: int = 10,
        *,
        search_variants: bool = True,
    ) -> dict[str, Any]:
        exact = self._search_core(query, top_k=top_k)
        exact_top_score = max(
            (float(item["score"]) for item in exact["results"]),
            default=0.0,
        )
        if not search_variants or exact_top_score >= 0.12:
            exact["variant_search"] = {
                "triggered": False,
                "reason": "exact_match_score_sufficient",
                "exact_top_score": round(exact_top_score, 6),
            }
            return exact

        candidates: dict[str, dict[str, Any]] = {
            item["group_id"]: item for item in exact["results"]
        }
        transformed_runs: list[dict[str, Any]] = []
        # The range is intentionally bounded.  A query that is already a close
        # match never reaches this path, so this cannot inflate same-format recall.
        for time_scale in (0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1.0, 1.02, 1.04, 1.06, 1.08, 1.10, 1.12):
            for pitch_shift in range(-6, 7):
                if time_scale == 1.0 and pitch_shift == 0:
                    continue
                transformed = transform_query_sequence(query, pitch_shift, time_scale)
                if not transformed.fingerprints:
                    continue
                run = self._search_core(transformed, top_k=top_k)
                run_top = max((float(item["score"]) for item in run["results"]), default=0.0)
                if run_top >= 0.08:
                    transformed_runs.append(
                        {
                            "pitch_shift_bands": pitch_shift,
                            "time_scale": time_scale,
                            "top_score": round(run_top, 6),
                            "candidate_order": run["candidate_order"],
                        }
                    )
                for result in run["results"]:
                    result["transform"] = {
                        "pitch_shift_bands": pitch_shift,
                        "time_scale": time_scale,
                        "label": _transform_label(pitch_shift, time_scale),
                    }
                    previous = candidates.get(result["group_id"])
                    if previous is None or (
                        float(result["score"]),
                        int(result["votes"]),
                        -int(result["transform"]["pitch_shift_bands"]),
                    ) > (
                        float(previous["score"]),
                        int(previous["votes"]),
                        -int(previous.get("transform", {}).get("pitch_shift_bands", 0)),
                    ):
                        candidates[result["group_id"]] = result
        ordered = sorted(
            candidates.values(),
            key=lambda item: (-float(item["score"]), -int(item["votes"]), item["group_id"]),
        )[: max(1, top_k)]
        for rank, result in enumerate(ordered, start=1):
            result["rank"] = rank
        return {
            "ok": True,
            "query": query.to_dict(),
            "results": ordered,
            "candidate_order": [item["group_id"] for item in ordered],
            "variant_search": {
                "triggered": True,
                "reason": "exact_match_score_below_probe_threshold",
                "exact_top_score": round(exact_top_score, 6),
                "probe_threshold": 0.12,
                "transformed_runs": sorted(
                    transformed_runs,
                    key=lambda item: (-item["top_score"], item["time_scale"], item["pitch_shift_bands"]),
                )[:32],
            },
        }

    def _search_core(self, query: FingerprintSequence, top_k: int = 10) -> dict[str, Any]:
        query_by_hash: dict[str, list[int]] = defaultdict(list)
        for item in query.fingerprints:
            query_by_hash[item.hash].append(item.anchor_frame)
        for anchors in query_by_hash.values():
            anchors.sort()
        occurrences = self._fetch_occurrences(list(query_by_hash.keys()))
        source_histograms: dict[str, Counter[int]] = defaultdict(Counter)
        row_by_source: dict[str, sqlite3.Row] = {}
        for row in occurrences:
            source_id = str(row["source_id"])
            row_by_source[source_id] = row
            for query_anchor in query_by_hash.get(row["fingerprint_hash"], []):
                offset = int(row["anchor_frame"]) - query_anchor
                source_histograms[source_id][offset] += 1

        source_evidence: list[tuple[str, VoteEvidence]] = []
        for source_id, histogram in source_histograms.items():
            evidence = _best_peak(histogram, len(query.fingerprints))
            source_evidence.append((source_id, evidence))

        group_best: dict[str, tuple[str, VoteEvidence]] = {}
        for source_id, evidence in source_evidence:
            group_id = str(row_by_source[source_id]["group_id"])
            current = group_best.get(group_id)
            if current is None or (evidence.score, evidence.votes, source_id) > (
                current[1].score,
                current[1].votes,
                current[0],
            ):
                group_best[group_id] = (source_id, evidence)

        ranked = sorted(
            group_best.items(),
            key=lambda item: (-item[1][1].score, -item[1][1].votes, item[0]),
        )[: max(1, top_k)]
        results: list[dict[str, Any]] = []
        frames_per_second = self.config.frames_per_second
        for rank, (group_id, (source_id, evidence)) in enumerate(ranked, start=1):
            group = self.group_details(group_id)
            if group is None:
                continue
            results.append(
                {
                    "rank": rank,
                    "group_id": group_id,
                    "display_label": group["display_label"],
                    "score": round(evidence.score, 6),
                    "votes": evidence.votes,
                    "best_source_id": source_id,
                    "alignment_start_seconds": round(
                        evidence.offset_frames / frames_per_second, 6
                    ),
                    "alignment_duration_seconds": round(query.duration, 6),
                    "formats": group["sources"],
                    "vote_histogram": evidence.to_dict(frames_per_second)["histogram"],
                    "score_explanation": {
                        "formula": "dominant_time_delta_votes / query_fingerprint_count",
                        "dominant_votes": evidence.votes,
                        "query_fingerprint_count": len(query.fingerprints),
                        "frames_per_second": frames_per_second,
                        "smoothing_radius_frames": 1,
                    },
                }
            )
        return {
            "ok": True,
            "query": query.to_dict(),
            "results": results,
            "candidate_order": [item["group_id"] for item in results],
        }

    def delete_group(self, group_id: str) -> dict[str, Any]:
        row = self.group_details(group_id)
        if row is None:
            return {
                "ok": True,
                "deleted": False,
                "group_id": group_id,
                "deleted_sources": 0,
                "deleted_fingerprints": 0,
            }
        source_count = len(row["sources"])
        fingerprint_count = int(row["fingerprint_count"])
        with self.transaction():
            self.connection.execute("DELETE FROM work_groups WHERE group_id=?", (group_id,))
            revision = self._bump_revision()
        return {
            "ok": True,
            "deleted": True,
            "group_id": group_id,
            "deleted_sources": source_count,
            "deleted_fingerprints": fingerprint_count,
            "revision": revision,
            "content_hash": self._read_meta("content_hash"),
        }

    def delete_source(self, source_id: str) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT group_id FROM sources WHERE source_id=?", (source_id,)
        ).fetchone()
        if row is None:
            return {"ok": True, "deleted": False, "source_id": source_id}
        group_id = str(row["group_id"])
        with self.transaction():
            self.connection.execute("DELETE FROM sources WHERE source_id=?", (source_id,))
            aggregate = self.connection.execute(
                "SELECT COUNT(*) AS source_count, COALESCE(SUM(fingerprint_count), 0) AS fingerprint_count, "
                "COALESCE(MAX(duration), 0) AS duration FROM sources WHERE group_id=?",
                (group_id,),
            ).fetchone()
            if int(aggregate["source_count"]) == 0:
                self.connection.execute("DELETE FROM work_groups WHERE group_id=?", (group_id,))
            else:
                self.connection.execute(
                    "UPDATE work_groups SET fingerprint_count=?, duration=? WHERE group_id=?",
                    (
                        int(aggregate["fingerprint_count"]),
                        float(aggregate["duration"]),
                        group_id,
                    ),
                )
            revision = self._bump_revision()
        return {
            "ok": True,
            "deleted": True,
            "source_id": source_id,
            "group_id": group_id,
            "revision": revision,
            "content_hash": self._read_meta("content_hash"),
        }

    def stats(self) -> dict[str, Any]:
        groups = int(self.connection.execute("SELECT COUNT(*) FROM work_groups").fetchone()[0])
        sources = int(self.connection.execute("SELECT COUNT(*) FROM sources").fetchone()[0])
        fingerprints = int(self.connection.execute("SELECT COUNT(*) FROM fingerprints").fetchone()[0])
        meta = self.metadata()
        return {
            "path": self.path.name,
            "groups": groups,
            "sources": sources,
            "fingerprints": fingerprints,
            "revision": int(meta["revision"]),
            "content_hash": meta["content_hash"],
            "fingerprint_version": meta["fingerprint_version"],
            "config_digest": meta["config_digest"],
            "encoding_space": meta["encoding_space"],
            "recovered_orphans": self.recovered_files,
        }

    def compare_config(self, new_config: FingerprintConfig) -> dict[str, Any]:
        old = config_from_json(self._read_meta("config_json"))
        return {
            "index_version": old.version,
            "requested_version": new_config.version,
            "index_config_digest": old.digest(),
            "requested_config_digest": new_config.digest(),
            "migratable": new_config.migration_compatible_from(old),
            "same_config": old.digest() == new_config.digest(),
        }


def vote_histogram(
    query_fingerprints: Iterable[Fingerprint],
    database_fingerprints: Iterable[Fingerprint],
) -> VoteEvidence:
    query = list(query_fingerprints)
    database = list(database_fingerprints)
    by_hash: dict[str, list[int]] = defaultdict(list)
    for item in database:
        by_hash[item.hash].append(item.anchor_frame)
    for anchors in by_hash.values():
        anchors.sort()
    histogram: Counter[int] = Counter()
    for item in query:
        for anchor in by_hash.get(item.hash, []):
            histogram[anchor - item.anchor_frame] += 1
    return _best_peak(histogram, len(query))


def transform_query_sequence(
    sequence: FingerprintSequence,
    pitch_shift_bands: int,
    time_scale: float,
) -> FingerprintSequence:
    """Apply a deterministic query-side pitch/time hypothesis.

    The database fingerprints remain untouched. Every transformed fingerprint
    is re-hashed from its actual band and frame-time features, so the hash
    remains independently reproducible.
    """

    transformed: list[Fingerprint] = []
    band_count = sequence.config.band_count
    for item in sequence.fingerprints:
        band1 = item.band1 + pitch_shift_bands
        band2 = item.band2 + pitch_shift_bands
        if band1 < 0 or band1 >= band_count or band2 < 0 or band2 >= band_count:
            continue
        anchor = int(round(item.anchor_frame * time_scale))
        target = int(round(item.target_frame * time_scale))
        delta = target - anchor
        if delta <= 0:
            continue
        transformed.append(
            Fingerprint(
                hash=sequence.config.hash_for(band1, band2, delta),
                anchor_frame=anchor,
                target_frame=target,
                band1=band1,
                band2=band2,
                delta_frames=delta,
            )
        )
    transformed.sort(
        key=lambda item: (
            item.anchor_frame,
            item.target_frame,
            item.band1,
            item.band2,
            item.hash,
        )
    )
    return FingerprintSequence(
        config=sequence.config,
        duration=sequence.duration * time_scale,
        peaks=[],
        fingerprints=transformed,
    )


def _transform_label(pitch_shift_bands: int, time_scale: float) -> str:
    if pitch_shift_bands and abs(time_scale - 1.0) > 1e-9:
        return f"pitch{ pitch_shift_bands:+d}bands_tempo{time_scale:.2f}x"
    if pitch_shift_bands:
        return f"pitch{pitch_shift_bands:+d}bands"
    return f"tempo{time_scale:.2f}x"


def _best_peak(histogram: Counter[int], query_count: int) -> VoteEvidence:
    if not histogram:
        return VoteEvidence(votes=0, offset_frames=0, score=0.0, histogram=Counter())
    # Smooth by one frame to tolerate decode-grid shifts without changing the
    # score's meaning: all votes still come from time-delta agreement.
    smoothed: Counter[int] = Counter()
    for offset, votes in histogram.items():
        for neighbour in (offset - 1, offset, offset + 1):
            smoothed[neighbour] += votes
    best_offset = min(
        smoothed,
        key=lambda offset: (-smoothed[offset], -histogram.get(offset, 0), abs(offset), offset),
    )
    votes = int(smoothed[best_offset])
    score = votes / query_count if query_count else 0.0
    return VoteEvidence(
        votes=votes,
        offset_frames=int(best_offset),
        score=min(1.0, score),
        histogram=histogram,
    )
