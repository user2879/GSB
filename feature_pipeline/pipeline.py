"""End-to-end orchestration for the feature exchange pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from .cache import IncrementalCache
from .concurrency import run_entity_tasks
from .config import load_feature_metadata, load_merge_config
from .conversion import validate_precision
from .diagnostics import build_interop_report
from .fingerprint import (
    cell_cache_key,
    config_fingerprint,
    metadata_fingerprint,
    source_fingerprint,
)
from .merge import merge_tables
from .models import V1, V2, V3, MergeConfig, stable_sort_key
from .parsers import v1_fixedwidth, v2_json, v3_binary
from .preprocess import apply_scaling, fill_missing, normalize_missing, scaling_statistics
from .snapshot import GracefulExit, SnapshotStore, build_manifest


PARSERS = {
    V1: v1_fixedwidth.parse_file,
    V2: v2_json.parse_file,
    V3: v3_binary.parse_file,
}


class Pipeline:
    def __init__(self, sample_dir: str | Path, config_path: str | Path,
                 metadata_path: str | Path, work_dir: str | Path):
        self.sample_dir = Path(sample_dir)
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.config: MergeConfig = load_merge_config(config_path)
        self.metadata = load_feature_metadata(metadata_path)
        self.config_hash = config_fingerprint(self.config)
        self.metadata_hash = metadata_fingerprint(self.metadata)
        self.cache = IncrementalCache(self.work_dir / "incremental_cache.json")
        self.cache.prune_whitelist(self.config.feature_whitelist)
        self.snapshots = SnapshotStore(self.work_dir / "snapshot_store")
        self.failures: list = []
        self.exit_signal = GracefulExit()

    def source_paths(self) -> dict[str, Path]:
        return {
            V1: self.sample_dir / "features_v1.txt",
            V2: self.sample_dir / "features_v2.json",
            V3: self.sample_dir / "features_v3.bin",
        }

    def parse_sources(self) -> dict[str, object]:
        tables = {}
        fingerprints = {}
        for version, path in self.source_paths().items():
            table = PARSERS[version](path)
            if version == V1:
                # v1 stores per-mille integers but the metadata table may assign
                # a different named precision to a feature; keep storage label in
                # raw_text and align the semantic label with the metadata table.
                for cell in table.cells:
                    meta = self.metadata.get(cell.feature)
                    if meta and meta.allowed_precisions and "per_mille" not in meta.allowed_precisions:
                        cell.precision_label = meta.allowed_precisions[0]
            for cell in table.cells:
                validate_precision(cell, self.metadata)
            tables[version] = table
            fingerprints[version] = source_fingerprint(version, path)
        return {"tables": tables, "fingerprints": fingerprints}

    def run(self) -> dict:
        parsed = self.parse_sources()
        tables_by_version = parsed["tables"]
        source_fingerprints = parsed["fingerprints"]
        tables = [tables_by_version[version] for version in self.config.source_order]

        normalize_missing(tables)

        # Stage 1: per-cell validation/normalization tasks, keyed by entity+stage.
        cell_tasks = []
        for table in tables:
            for cell in table.cells:
                if cell.feature not in self.config.feature_whitelist:
                    continue
                cell_tasks.append(
                    (
                        cell.entity_id,
                        "parse_validate",
                        lambda cell=cell: self._cell_stage(cell),
                    )
                )
        _, stage_failures = run_entity_tasks(cell_tasks, self.config.max_workers)
        self.failures.extend(stage_failures)

        fill_audit = fill_missing(tables, self.config)
        stats = scaling_statistics(tables, self.config)
        apply_scaling(tables, stats, self.config)

        # Stage 2: per-entity scaling tasks (deterministic key ordering).
        scale_tasks = []
        for table in tables:
            for cell in table.cells:
                if cell.feature in self.config.feature_whitelist:
                    scale_tasks.append(
                        (
                            cell.entity_id,
                            "scale",
                            lambda cell=cell: (cell.entity_id, cell.feature,
                                                float(cell.value)),
                        )
                    )
        _, scale_failures = run_entity_tasks(scale_tasks, self.config.max_workers)
        self.failures.extend(scale_failures)

        all_entities = sorted({
            cell.entity_id
            for table in tables
            for cell in table.cells
            if cell.feature in self.config.feature_whitelist
        })

        merged_rows: list[dict] = []
        conflicts: list[dict] = []
        suppressed: list[dict] = []
        provenance: list[dict] = []
        batches: list[dict] = []

        for start in range(0, len(all_entities), self.config.batch_size):
            batch_entities = all_entities[start : start + self.config.batch_size]
            batch_output = merge_tables(
                tables, self.config, self.config_hash, batch_entities
            )
            self._merge_with_cache(batch_output, batch_entities)
            merged_rows.extend(batch_output.rows)
            conflicts.extend(batch_output.conflicts)
            suppressed.extend(batch_output.suppressed)
            provenance.extend(batch_output.provenance)
            batches.append(
                {
                    "batch_index": len(batches),
                    "entity_ids": batch_entities,
                    "row_count": len(batch_output.rows),
                }
            )
            if self.exit_signal.requested:
                break

        merged_rows.sort(key=lambda row: (row["entity_id"], row["feature"]))
        conflicts.sort(key=lambda row: (row["entity_id"], row["feature"]))
        suppressed.sort(
            key=lambda row: (
                row["entity_id"], row["feature"], row["suppressed_source"]
            )
        )
        provenance.sort(key=lambda row: (row["entity_id"], row["feature"]))

        base_table = tables_by_version[V2]
        interop_report = build_interop_report(
            base_table,
            self.metadata,
            self.config,
            self.work_dir / "interop_roundtrip",
        )

        failure_payload = [failure.to_dict() for failure in self.failures]
        cache_audit = self.cache.audit()
        self.cache.save()

        prior_manifest = self.snapshots.current_manifest()
        global_version = (
            prior_manifest["global_version"] + 1 if prior_manifest else 1
        )
        manifest = build_manifest(
            global_version=global_version,
            config_hash=self.config_hash,
            metadata_hash=self.metadata_hash,
            source_fingerprints=source_fingerprints,
            stage_versions={
                "merged_table.json": global_version,
                "conflicts.json": global_version,
                "interop_report.json": global_version,
                "failures.json": global_version,
                "incremental_audit.json": global_version,
            },
        )
        staging = self.snapshots.begin_snapshot(manifest)
        _write_json(staging / "merged_table.json", {
            "rows": merged_rows,
            "batch_plan": batches,
            "fill_audit": fill_audit,
            "scaling_statistics": stats,
        })
        _write_json(staging / "conflicts.json", {
            "conflicts": conflicts,
            "suppressed": suppressed,
            "provenance": provenance,
        })
        _write_json(staging / "interop_report.json", interop_report)
        _write_json(staging / "failures.json", {"failures": failure_payload})
        _write_json(staging / "incremental_audit.json", cache_audit)
        pointer = self.snapshots.commit_snapshot(staging, manifest)

        recovery = self.snapshots.recovery_point()
        result = {
            "global_version": global_version,
            "pointer": pointer,
            "recovery_point": recovery,
            "row_count": len(merged_rows),
            "conflict_count": len(conflicts),
            "failure_count": len(failure_payload),
            "graceful_exit_requested": self.exit_signal.requested,
            "exit_signal": self.exit_signal.signal_name,
            "cache_audit": cache_audit,
            "staging_dir": str(staging),
        }
        _write_json(self.work_dir / "run_summary.json", result)
        return result

    def _cell_stage(self, cell) -> dict:
        key = cell_cache_key(cell, self.config_hash)
        coordinate = f"{cell.entity_id}|{cell.feature}|{cell.source_version}"
        cached = self.cache.get_cell(coordinate, key)
        payload = {
            "entity_id": cell.entity_id,
            "feature": cell.feature,
            "source_version": cell.source_version,
            "missing": cell.missing,
            "precision_label": cell.precision_label,
            "sort_key": list(stable_sort_key(cell.entity_id, "parse_validate")),
        }
        if cached is None:
            self.cache.put_cell(coordinate, key, payload)
        return payload

    def _merge_with_cache(self, batch_output, batch_entities: list[str]) -> None:
        for row in batch_output.rows:
            coordinate = f"{row['entity_id']}|{row['feature']}"
            cached = self.cache.get_merge(
                coordinate, row["dependency_key"]
            )
            if cached is None:
                self.cache.put_merge(
                    coordinate,
                    row["dependency_key"],
                    {"value": row["value"], "winning_source": row["winning_source"]},
                )


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True,
                   allow_nan=True),
        encoding="utf-8",
    )
