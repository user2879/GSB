"""Stage orchestration: validate -> fingerprint -> gather, with
versioned snapshots, precise invalidation and diagnostic reports."""

import json
import os

from .edits import apply_edit_command
from .errors import (BusinessError, E_ID_MISSING, E_SNAPSHOT_MISMATCH,
                     GracefulExit)
from .gather import GatherRequest, GatherResult
from .observability import result_metrics
from .scheduler import run_gather_stage
from .snapshot import SnapshotStore
from .table import EmbeddingTable


def load_requests(path):
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    raw = data["requests"] if isinstance(data, dict) else data
    requests = []
    for index, item in enumerate(raw):
        requests.append(GatherRequest(index, item["ids"],
                                      item.get("request_id")))
    return requests


class Pipeline:
    def __init__(self, config, injector=None, exit_flag=None):
        self.config = config
        self.injector = injector
        self.exit_flag = exit_flag
        self.store = SnapshotStore(config.snapshot_dir)
        self.table = None
        self.manifest = None
        self.results = {}            # request_index -> GatherResult
        self.budget_report = None
        self.pending = []
        self.recovery_info = None
        self.last_invalidation = {"invalidated": [], "reused": []}

    # ------------------------------------------------------------ helpers
    def _check_exit(self, context):
        if self.exit_flag is not None and self.exit_flag.requested:
            progress = {
                "context": context,
                "snapshot_version":
                    self.manifest["global_version"] if self.manifest else 0,
                "signal": self.exit_flag.signum,
            }
            self.store.write_progress(progress)
            raise GracefulExit(progress)

    # ------------------------------------------------------------ recovery
    def recover(self):
        """Check an existing snapshot chain; refuse to continue on any
        version/config/quant-level/layout mismatch."""
        manifest = self.store.load_manifest()
        if manifest is None:
            return None
        # Phase 1: config hash + quantization level (pre-table checks).
        self.store.verify_manifest(manifest, self.config)
        self.manifest = manifest
        progress = self.store.load_progress()
        self.recovery_info = {
            "version": manifest["global_version"],
            "progress": progress,
        }
        return manifest

    def _verify_layout_against_manifest(self):
        # Phase 2: block layout fingerprint (post-table-load check).
        if self.manifest is not None:
            self.store.verify_manifest(self.manifest, self.config,
                                       layout_fp=self.table.layout_fp)

    # --------------------------------------------------------------- stages
    def _stage_validate(self):
        self.table = EmbeddingTable.load(self.config, self.injector)
        self._verify_layout_against_manifest()
        if self.manifest is None:
            self.manifest = self.store.new_manifest(self.config,
                                                    self.table.layout_fp)
        self.manifest["layout_fp"] = self.table.layout_fp
        version = self.store.commit_stage(
            self.manifest, "validate",
            self.table.validate_stage_artifact(0))
        self._check_exit("validate")
        return version

    def _stage_fingerprint(self):
        version = self.store.commit_stage(
            self.manifest, "fingerprint",
            self.table.fingerprint_stage_artifact(0))
        self._check_exit("fingerprint")
        return version

    def _stage_gather(self, requests, only_indices=None):
        results, budget, pending = run_gather_stage(
            requests, self.table, self.config, self.injector,
            only_indices=only_indices)
        self.results.update(results)
        self.budget_report = budget
        self.pending = pending
        artifact = {
            "stage": "gather",
            "version": 0,
            "agg_mode": self.config.agg_mode,
            "quant_level": self.config.quant_level,
            "results": {str(idx): self.results[idx].to_dict()
                        for idx in sorted(self.results)},
        }
        version = self.store.commit_stage(self.manifest, "gather", artifact)
        self._check_exit("gather")
        return version

    def _exercise_snapshot_injection(self):
        """Fault injection: a stale snapshot reused after a block layout
        change must be recognised and refused (E_SNAPSHOT_MISMATCH)."""
        if self.injector is None or self.manifest is None:
            return
        tampered = self.injector.tampered_manifest(self.manifest)
        if tampered is None:
            return
        try:
            self.store.verify_manifest(tampered, self.config,
                                       layout_fp=self.table.layout_fp)
        except BusinessError as exc:
            if exc.code != E_SNAPSHOT_MISMATCH:
                raise
            # Recorded in the injection hit list; run continues so a
            # single fault never interrupts other requests.

    # -------------------------------------------------------------- full run
    def full_run(self, requests):
        self.recover()
        self._stage_validate()
        self._stage_fingerprint()
        self._exercise_snapshot_injection()
        invalidated = sorted(r.index for r in requests)
        self._stage_gather(requests)
        self.last_invalidation = {"invalidated": invalidated, "reused": []}
        return self.build_report("full")

    # ------------------------------------------------------ incremental run
    def incremental_session(self, requests, commands, on_step=None):
        """Run a full baseline then apply edit commands incrementally."""
        self.recover()
        self._stage_validate()
        self._stage_fingerprint()
        self._stage_gather(requests)
        steps = []
        for cmd_index, cmd in enumerate(commands):
            self._check_exit(f"command:{cmd_index}")
            step = self.apply_command(requests, cmd, cmd_index)
            steps.append(step)
            if on_step is not None:
                on_step(step)
        report = self.build_report("incremental")
        report["steps"] = steps
        return report

    def apply_command(self, requests, cmd, cmd_index):
        impact = apply_edit_command(self.table, self.config, cmd)
        self.manifest["quant_level"] = self.config.quant_level
        self.manifest["agg_mode"] = self.config.agg_mode
        if impact["table_changed"]:
            self.manifest["layout_fp"] = self.table.layout_fp
            self.store.commit_stage(self.manifest, "validate",
                                    self.table.validate_stage_artifact(0))
            self.store.commit_stage(
                self.manifest, "fingerprint",
                self.table.fingerprint_stage_artifact(0))
        invalidated = self._compute_invalidation(requests, impact)
        reused = sorted(r.index for r in requests
                        if r.index not in invalidated)
        if invalidated:
            self._stage_gather(requests, only_indices=set(invalidated))
        self.last_invalidation = {
            "invalidated": sorted(invalidated),
            "reused": reused,
        }
        return {
            "command_index": cmd_index,
            "command": cmd,
            "invalidated": sorted(invalidated),
            "reused": reused,
            "stage_invalidations": self._stage_invalidation_view(impact),
            "snapshot_version": self.manifest["global_version"],
        }

    @staticmethod
    def _stage_invalidation_view(impact):
        if impact["invalidate_all"]:
            return {"validate": False, "fingerprint": False, "gather": True}
        if impact["table_changed"]:
            return {"validate": True, "fingerprint": True, "gather": True}
        return {"validate": False, "fingerprint": False, "gather": False}

    def _compute_invalidation(self, requests, impact):
        if impact["invalidate_all"]:
            return sorted(r.index for r in requests)
        affected = impact["affected_word_ids"]
        added = impact["added_word_ids"]
        invalidated = []
        for req in requests:
            res = self.results.get(req.index)
            if res is None:
                invalidated.append(req.index)
                continue
            if res.status == "ok":
                referenced_ids = {pair[0] for pair in res.referenced}
                if referenced_ids & affected:
                    invalidated.append(req.index)
            else:
                # Retry failed requests whose fault may be resolved by
                # this edit (e.g. an appended block now provides the id).
                missing = (res.detail or {}).get("word_id")
                if missing is not None and (missing in affected
                                            or missing in added):
                    invalidated.append(req.index)
        return sorted(invalidated)

    # --------------------------------------------------------------- report
    def build_report(self, run_kind):
        request_entries = []
        max_err = 0.0
        codes = set()
        for index in sorted(self.results):
            res = self.results[index]
            metrics = result_metrics(res, self.table)
            if metrics["max_abs_error"] is not None:
                max_err = max(max_err, metrics["max_abs_error"])
            if res.code:
                codes.add(res.code)
            entry = res.to_dict()
            entry["max_abs_error"] = metrics["max_abs_error"]
            entry["deviation_positions"] = metrics["deviation_positions"]
            request_entries.append(entry)
        for failure in (self.table.failure_list if self.table else []):
            codes.add(failure["code"])
        injections = self.injector.hits if self.injector else []
        for hit in injections:
            codes.add(hit["business_code"])
        report = {
            "run_kind": run_kind,
            "snapshot_version":
                self.manifest["global_version"] if self.manifest else 0,
            "recovery": self.recovery_info,
            "config": {
                "dim": self.config.dim,
                "rows_per_block": self.config.rows_per_block,
                "quant_level": self.config.quant_level,
                "frac_bits": self.config.frac_bits,
                "agg_mode": self.config.agg_mode,
                "max_concurrency": self.config.max_concurrency,
            },
            "layout_fp": self.table.layout_fp if self.table else None,
            "requests": request_entries,
            "invalidation": {
                "invalidated": self.last_invalidation["invalidated"],
                "reused": self.last_invalidation["reused"],
                "invalidated_count":
                    len(self.last_invalidation["invalidated"]),
                "reused_count": len(self.last_invalidation["reused"]),
            },
            "budget": self.budget_report,
            "error_summary": {"max_abs_error_overall": max_err},
            "injections": injections,
            "failure_list":
                list(self.table.failure_list) if self.table else [],
            "pending_requests": list(self.pending),
            "business_codes": sorted(codes),
        }
        return report


def write_report(report, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def results_from_snapshot(store):
    artifact = store.load_stage("gather")
    if artifact is None:
        return {}
    return {int(k): GatherResult.from_dict(v)
            for k, v in artifact["results"].items()}
