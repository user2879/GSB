from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonical import canonical_json, digest_json
from .errors import JoinLabError, fail
from .estimation import FIXED_SCALE, estimate_join
from .filters import apply_filters
from .histogram import Histogram, build_histogram
from .logging_utils import StructuredLogger
from .params import AnalysisParams
from .sampling import SampleResult, build_sample
from .snapshot import load_snapshot, save_snapshot, verify_replay
from .spec import TableSpec, WorkspaceSpec, ingest_workspace, load_table_rows, table_structure_payload
from .truth import execute_truth_join, validate_estimates
from .versioning import VersionStore


@dataclass
class LiveView:
    params: AnalysisParams
    params_version: int
    filtered_rows: dict[str, list[dict[str, Any]]]
    histograms: dict[str, Histogram]
    samples: dict[str, SampleResult]
    estimate: dict[str, Any]
    truth: dict[str, Any]
    validation: dict[str, Any]
    estimate_error: JoinLabError | None = None


class WorkbenchSession:
    def __init__(
        self,
        spec: WorkspaceSpec,
        db_path: str | Path,
        params: AnalysisParams,
        logger: StructuredLogger,
        version_state_path: str | Path | None = None,
    ) -> None:
        self.spec = spec
        self.db_path = Path(db_path).resolve()
        self.params = params
        self.logger = logger
        self.ingest = ingest_workspace(spec, self.db_path)
        self.version_store = VersionStore(version_state_path, spec.workspace_id)
        self.params_version = self.version_store.ensure(params)
        self.current: LiveView | None = None
        self.stale_epochs: list[dict[str, Any]] = []
        self.last_replacement: dict[str, Any] | None = None
        self.logger.info(
            "workspace_loaded",
            workspace_id=spec.workspace_id,
            db_path=str(self.db_path),
            params_version=self.params_version,
            tables=sorted(spec.tables),
        )

    def _load_filtered_rows(self) -> dict[str, list[dict[str, Any]]]:
        result: dict[str, list[dict[str, Any]]] = {}
        for table_name in (self.spec.join.left_table, self.spec.join.right_table):
            table = self.spec.table(table_name)
            rows = load_table_rows(self.db_path, table)
            result[table_name] = apply_filters(rows, self.spec.filters.get(table_name, ()))
        return result

    def _table_filter_payload(self, table: TableSpec, filtered_rows: list[dict[str, Any]]) -> dict[str, Any]:
        predicates = self.spec.filters.get(table.name, ())
        return {
            "structure_version": table.structure_version,
            "data_digest": table.data_digest,
            "row_count": table.row_count,
            "filter_digest": digest_json(
                [
                    {"column": predicate.column, "op": predicate.op, "value": predicate.value}
                    for predicate in predicates
                ]
            ),
            "filtered_row_count": len(filtered_rows),
            "filtered_rid_digest": digest_json(sorted(row["__rid"] for row in filtered_rows)),
        }

    def _build_view(self, params: AnalysisParams, version: int) -> tuple[LiveView, JoinLabError | None]:
        filtered = self._load_filtered_rows()
        join = self.spec.join
        left_table = self.spec.table(join.left_table)
        right_table = self.spec.table(join.right_table)
        left_column = left_table.column(join.left_key)
        right_column = right_table.column(join.right_key)
        if left_column is None or right_column is None:
            fail("E_TABLE", "联接键在装载后丢失", "join")
        left_hist = build_histogram(
            left_table.name,
            join.left_key,
            left_column.type,
            filtered[left_table.name],
            params.bucket_count,
            params.quantile_policy,
        )
        right_hist = build_histogram(
            right_table.name,
            join.right_key,
            right_column.type,
            filtered[right_table.name],
            params.bucket_count,
            params.quantile_policy,
        )
        left_sample = build_sample(
            left_table.name,
            join.left_key,
            left_column.type,
            filtered[left_table.name],
            params.sample_size,
            params.seed,
            left_table.data_digest,
        )
        right_sample = build_sample(
            right_table.name,
            join.right_key,
            right_column.type,
            filtered[right_table.name],
            params.sample_size,
            params.seed,
            right_table.data_digest,
        )
        estimate = estimate_join(
            left_hist,
            right_hist,
            left_sample,
            right_sample,
            len(filtered[left_table.name]),
            len(filtered[right_table.name]),
            params.attribution_k,
        )
        truth = execute_truth_join(
            filtered[left_table.name],
            filtered[right_table.name],
            join.left_key,
            join.right_key,
            left_column.type,
            right_column.type,
        )
        validation_error: JoinLabError | None = None
        try:
            validation = validate_estimates(
                truth,
                estimate["histogram_path"],
                estimate["sample_path"],
                params.bucket_count,
                params.sample_size,
                estimate["attribution"],
                FIXED_SCALE,
            )
        except JoinLabError as exc:
            if exc.code != "E_ESTIMATE":
                raise
            validation_error = exc
            validation = exc.details["validation"]
        view = LiveView(
            params=params,
            params_version=version,
            filtered_rows=filtered,
            histograms={left_table.name: left_hist, right_table.name: right_hist},
            samples={left_table.name: left_sample, right_table.name: right_sample},
            estimate=estimate,
            truth=truth,
            validation=validation,
            estimate_error=validation_error,
        )
        return view, validation_error

    def build_current(self) -> dict[str, Any]:
        view, error = self._build_view(self.params, self.params_version)
        self.current = view
        self.logger.info(
            "statistics_built",
            workspace_id=self.spec.workspace_id,
            params_version=view.params_version,
            histogram_digests={name: hist.digest for name, hist in view.histograms.items()},
            sample_digests={name: sample.sample_digest for name, sample in view.samples.items()},
            validation_status=view.validation["status"],
        )
        if error:
            self.logger.error(
                error.code,
                error.message,
                field_path=error.field_path,
                details=error.details,
            )
            raise error
        return self.current_report()

    def _stale_summary(self, view: LiveView, reason: str) -> dict[str, Any]:
        return {
            "stale": True,
            "reason": reason,
            "params_version": view.params_version,
            "params": view.params.as_dict(),
            "histogram_digests": {name: hist.digest for name, hist in view.histograms.items()},
            "sample_digests": {name: sample.sample_digest for name, sample in view.samples.items()},
            "histogram_estimate_fixed": view.estimate["histogram_path"]["estimate_fixed"],
            "sample_estimate_fixed": view.estimate["sample_path"]["estimate_fixed"],
            "attribution_digest": digest_json(view.estimate["attribution"]),
            "validation_digest": digest_json(view.validation),
        }

    def _replacement_report(
        self,
        old: LiveView | None,
        new: LiveView,
        reason: str,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        changed_tables: list[dict[str, Any]] = []
        table_names = sorted(set(new.histograms) | set(new.samples))
        for table_name in table_names:
            old_hist = old.histograms.get(table_name) if old else None
            new_hist = new.histograms.get(table_name)
            old_sample = old.samples.get(table_name) if old else None
            new_sample = new.samples.get(table_name)
            changed = (
                old is None
                or old_hist is None
                or new_hist is None
                or old_hist.digest != new_hist.digest
                or old_sample is None
                or new_sample is None
                or old_sample.sample_digest != new_sample.sample_digest
            )
            if not changed:
                continue
            changed_tables.append(
                {
                    "table": table_name,
                    "old_bucket_count": old_hist.bucket_count if old_hist else None,
                    "new_bucket_count": new_hist.bucket_count if new_hist else None,
                    "old_histogram_digest": old_hist.digest if old_hist else None,
                    "new_histogram_digest": new_hist.digest if new_hist else None,
                    "old_sample_digest": old_sample.sample_digest if old_sample else None,
                    "new_sample_digest": new_sample.sample_digest if new_sample else None,
                }
            )
        return {
            "reason": reason,
            "from_version": old.params_version if old else None,
            "to_version": new.params_version,
            "replacement_mode": "whole_view_replace",
            "changed_tables": changed_tables,
            "affected_conclusions": [
                "histogram_estimate",
                "sample_estimate",
                "bucket_attribution",
                "truth_validation",
            ],
            "old_view_status": "STALE" if old else "NONE",
            **({"extra": extra} if extra else {}),
        }

    def replace_parameters(self, new_params: AnalysisParams) -> dict[str, Any]:
        if new_params.digest() == self.params.digest() and self.current is not None:
            return {
                "ok": True,
                "changed": False,
                "params_version": self.params_version,
                "message": "参数未变化，保留当前统计视图",
            }
        old = self.current
        if old is not None:
            self.stale_epochs.append(self._stale_summary(old, "parameters_changed"))
        self.current = None
        self.params = new_params
        self.params_version = self.version_store.bump(new_params)
        self.logger.info(
            "parameters_replaced",
            workspace_id=self.spec.workspace_id,
            from_version=old.params_version if old else None,
            to_version=self.params_version,
            old_params=old.params.as_dict() if old else None,
            new_params=new_params.as_dict(),
        )
        view, error = self._build_view(self.params, self.params_version)
        self.current = view
        self.last_replacement = self._replacement_report(old, view, "parameters_changed")
        if error:
            raise error
        return {
            "ok": True,
            "changed": True,
            "replacement": self.last_replacement,
            "report": self.current_report(),
        }

    def current_report(self) -> dict[str, Any]:
        if self.current is None:
            fail("E_REPLAY", "当前没有有效统计视图", "session.current")
        view = self.current
        tables = {
            name: self._table_filter_payload(self.spec.table(name), view.filtered_rows[name])
            for name in (self.spec.join.left_table, self.spec.join.right_table)
        }
        return {
            "ok": True,
            "workspace_id": self.spec.workspace_id,
            "db_path": str(self.db_path),
            "params_version": view.params_version,
            "params": view.params.as_dict(),
            "join": {
                "left_table": self.spec.join.left_table,
                "left_key": self.spec.join.left_key,
                "right_table": self.spec.join.right_table,
                "right_key": self.spec.join.right_key,
            },
            "tables": tables,
            "histograms": {name: hist.as_dict() for name, hist in view.histograms.items()},
            "samples": {name: sample.as_dict() for name, sample in view.samples.items()},
            "estimate": view.estimate,
            "attribution": view.estimate["attribution"],
            "truth": view.truth,
            "validation": view.validation,
            "stale_epochs": list(self.stale_epochs),
            "replacement": self.last_replacement,
        }

    def _snapshot_document(self, view: LiveView) -> dict[str, Any]:
        tables = {
            name: self._table_filter_payload(self.spec.table(name), view.filtered_rows[name])
            for name in (self.spec.join.left_table, self.spec.join.right_table)
        }
        histograms = {
            name: {
                "digest": hist.digest,
                "bucket_count": hist.bucket_count,
                "key": hist.key,
                "key_type": hist.key_type,
                "quantile_policy": hist.quantile_policy,
                "boundary_rule": hist.boundary_rule,
                "null_count": hist.null_count,
            }
            for name, hist in view.histograms.items()
        }
        samples = {
            name: {
                "sample_digest": sample.sample_digest,
                "seed": sample.seed,
                "requested_size": sample.requested_size,
                "sampled_row_count": sample.sampled_row_count,
                "table_row_count": sample.table_row_count,
                "distinct_key_count": len(sample.frequencies),
                "null_sample_count": sample.null_sample_count,
            }
            for name, sample in view.samples.items()
        }
        truth = {
            key: view.truth[key]
            for key in (
                "true_row_count",
                "grouped_row_count",
                "grouped_distribution_digest",
                "output_sequence_rule",
                "output_sequence_digest",
                "engine",
            )
        }
        return {
            "workspace_id": self.spec.workspace_id,
            "params_version": view.params_version,
            "params": view.params.as_dict(),
            "tables": tables,
            "histograms": histograms,
            "samples": samples,
            "estimate": view.estimate,
            "attribution": view.estimate["attribution"],
            "truth": truth,
            "validation": view.validation,
        }

    def save_snapshot(self, path: str | Path) -> dict[str, Any]:
        if self.current is None:
            fail("E_SNAP", "当前没有可保存的统计视图", "snapshot")
        document = self._snapshot_document(self.current)
        save_snapshot(path, document)
        loaded_digest = digest_json(
            {key: document.get(key) for key in ("params", "tables", "histograms", "samples", "estimate", "attribution", "truth", "validation")}
        )
        self.logger.info(
            "snapshot_saved",
            workspace_id=self.spec.workspace_id,
            path=str(path),
            params_version=self.current.params_version,
            conclusion_digest=loaded_digest,
        )
        return {
            "ok": True,
            "path": str(Path(path).resolve()),
            "params_version": self.current.params_version,
            "conclusion_digest": loaded_digest,
        }

    def restore_snapshot(self, path: str | Path) -> dict[str, Any]:
        snapshot = load_snapshot(path)
        from .params import normalise_params

        desired = normalise_params(snapshot["params"], None, self.spec.attribution_k)
        old = self.current
        if old is not None:
            self.stale_epochs.append(self._stale_summary(old, "snapshot_restore"))
        self.current = None
        self.params = desired
        active_version = int(snapshot["params_version"])
        view, error = self._build_view(desired, active_version)
        recomputed = self._snapshot_document(view)
        replay = verify_replay(snapshot, recomputed)
        self.current = view
        self.params_version = active_version
        self.last_replacement = self._replacement_report(
            old,
            view,
            "snapshot_restore",
            {
                "snapshot_version": snapshot["params_version"],
                "previous_live_version": old.params_version if old else None,
                "version_mismatch": old is None or old.params_version != snapshot["params_version"],
            },
        )
        self.logger.info(
            "snapshot_restored",
            workspace_id=self.spec.workspace_id,
            path=str(path),
            snapshot_version=snapshot["params_version"],
            replay_verified=True,
        )
        if error:
            raise error
        return {
            "ok": True,
            "replay": replay,
            "replacement": self.last_replacement,
            "report": self.current_report(),
        }


def workspace_digest(spec: WorkspaceSpec) -> str:
    return digest_json(
        {
            "workspace_id": spec.workspace_id,
            "tables": table_structure_payload(spec),
            "join": {
                "left_table": spec.join.left_table,
                "left_key": spec.join.left_key,
                "right_table": spec.join.right_table,
                "right_key": spec.join.right_key,
            },
        }
    )
