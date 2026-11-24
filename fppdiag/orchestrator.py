"""End-to-end orchestration: run, persist, recover, replay and increment.

Identity / replacement policy
-----------------------------
On open, the snapshot's spec version, spec digest, sources digest and scale
policy version must all match the supplied spec. Any mismatch triggers a
*whole* ledger replacement (never incremental merging).

Recovery policy
---------------
Only ``committed`` stage batches are trusted. ``unknown`` batches (a process
killed mid-stage) and missing stages are rerun. Replay recomputation of
committed stages is compared row-by-row; the first mismatching stage/field
raises E_REPLAY.
"""

from __future__ import annotations

import os

from .engine import PipelineEngine
from .errors import DivergeError, ReplayError, SnapshotError
from . import fixedpoint as fp
from .ledger import Ledger, Snapshot
from .spec import PipelineSpec
from .stats import compute_window_stats, evaluate_feature
from .util import content_digest

SCALE_POLICY_VERSION = 1


class Orchestrator:
    def __init__(self, spec: PipelineSpec, workdir: str,
                 schedule: str = "stage_major"):
        self.spec = spec
        self.workdir = workdir
        self.schedule = schedule
        os.makedirs(workdir, exist_ok=True)
        self.ledger_path = os.path.join(workdir, "ledger.sqlite")
        self.snapshot_path = os.path.join(workdir, "snapshot.json")
        self.ledger = Ledger(self.ledger_path)
        self.engine = PipelineEngine(spec, workdir)
        self._snapshot_policy()

    # -- identity ----------------------------------------------------------
    def _snapshot_policy(self) -> None:
        current = Snapshot(
            spec_version=self.spec.version,
            spec_digest=self.spec.digest(),
            sources_digest=self.spec.sources_digest(),
            scale_policy_version=SCALE_POLICY_VERSION,
            ledger_cursor={"committed": sorted(
                self.ledger.committed_stages(self.spec.version))})
        if not os.path.exists(self.snapshot_path):
            current.write(self.snapshot_path)
            return
        try:
            prior = Snapshot.read(self.snapshot_path)
        except SnapshotError:
            # A corrupted/tampered snapshot digest is a hard E_SNAPSHOT fault;
            # callers that want whole-replacement must remove the snapshot.
            raise
        identity = (prior.spec_version == current.spec_version and
                    prior.spec_digest == current.spec_digest and
                    prior.sources_digest == current.sources_digest and
                    prior.scale_policy_version == current.scale_policy_version)
        if not identity:
            # Whole replacement, never incremental merge.
            self.ledger.replace_all()
        current.write(self.snapshot_path)

    # -- feature computation ----------------------------------------------
    def _window_values(self, run, start: int, end: int):
        vals = run.all_values()
        return vals[start:end]

    def _stats_for(self, run, start: int, end: int, decile_edges=None):
        vals = self._window_values(run, start, end)
        return compute_window_stats(vals, run.scale_exp,
                                    self.spec.session_scale_exp, decile_edges)

    def _window_stats_pair(self, run, w):
        ref_vals = self._window_values(run, w.ref_start, w.ref_end)
        from .stats import reference_decile_edges
        edges = reference_decile_edges(ref_vals, run.scale_exp,
                                       self.spec.session_scale_exp)
        ref_stats = self._stats_for(run, w.ref_start, w.ref_end, edges)
        obs_stats = self._stats_for(run, w.obs_start, w.obs_end, edges)
        return ref_stats, obs_stats

    def _feature_bound_q(self, stage_id: str) -> int:
        chain = self.engine.error_bounds[stage_id]
        bound = chain["bound_ulp"]
        stage_scale = chain["scale_exp"]
        q = self.spec.session_scale_exp
        if q >= stage_scale:
            return bound * fp.POW10[q - stage_scale]
        return fp.div_half_away(bound, fp.POW10[stage_scale - q])

    def _uncertain_rows(self, run) -> int:
        return sum(b for blk in run.blocks for b in blk.boundary_uncertain)

    def run_full(self, crash_after_stage: str | None = None) -> dict:
        """Execute every stage; optionally simulate a forced termination.

        Pass 1 computes integer values stage by stage and commits each stage
        batch atomically. A simulated crash leaves the current batch marked
        ``unknown``. Pass 2 builds error chains from observed magnitudes and
        persists feature stats and verdicts for newly computed stages.
        """
        self.engine.load_sources()
        w = self.spec.window
        for seq, stage in enumerate(self.spec.stages):
            existing = self.ledger.batch_status(self.spec.version, stage.stage_id)
            if existing is not None and existing["status"] == "committed":
                # Replay committed outputs in-memory so dependents can read them.
                self.engine.run_stage(stage)
                continue
            self.ledger.begin_batch(self.spec.version, stage.stage_id, seq)
            if crash_after_stage == stage.stage_id:
                # Simulate SIGKILL: batch left "unknown", nothing committed.
                return {"crashed": stage.stage_id}
            run = self.engine.run_stage(stage)
            block_digests = [content_digest(b.values) for b in run.blocks]
            self.ledger.commit_stage(self.spec.version, stage.stage_id, seq,
                                     run.summary(), block_digests)
        from .errorchain import build_error_chains
        self.engine.error_bounds = build_error_chains(self.spec, self.engine)
        for stage in self.spec.stages:
            already, _ = self.ledger.get_verdict(self.spec.version, stage.stage_id)
            if already is not None:
                continue
            self._persist_feature(stage, self.engine.runs[stage.stage_id], w)
        self._write_cursor()
        return self.report()

    def _persist_feature(self, stage, run, w) -> None:
        ref_stats, obs_stats = self._window_stats_pair(run, w)
        self.ledger.put_feature_stats(self.spec.version, stage.stage_id,
                                      "reference", ref_stats.to_dict())
        self.ledger.put_feature_stats(self.spec.version, stage.stage_id,
                                      "observation", obs_stats.to_dict())
        chain = self.engine.error_bounds[stage.stage_id]
        self.ledger.put_chain(self.spec.version, stage.stage_id, chain)
        verdict = evaluate_feature(
            stage.stage_id, ref_stats, obs_stats,
            self.spec.thresholds_bps[stage.stage_id],
            self._feature_bound_q(stage.stage_id),
            self._uncertain_rows(run), run.scale_exp)
        self.ledger.put_verdict(self.spec.version, stage.stage_id, verdict)

    def _write_cursor(self) -> None:
        snap = Snapshot(
            spec_version=self.spec.version,
            spec_digest=self.spec.digest(),
            sources_digest=self.spec.sources_digest(),
            scale_policy_version=SCALE_POLICY_VERSION,
            ledger_cursor={"committed": sorted(
                self.ledger.committed_stages(self.spec.version))})
        snap.write(self.snapshot_path)

    # -- recovery & replay -------------------------------------------------
    def recover(self) -> dict:
        # Rerun everything in-memory, verify committed, replace unknown/missing.
        self.engine.load_sources()
        replay_issues = []
        for seq, stage in enumerate(self.spec.stages):
            run = self.engine.run_stage(stage)
        from .errorchain import build_error_chains
        self.engine.error_bounds = build_error_chains(self.spec, self.engine)
        for seq, stage in enumerate(self.spec.stages):
            run = self.engine.runs[stage.stage_id]
            status = self.ledger.batch_status(self.spec.version, stage.stage_id)
            if status is None or status["status"] == "unknown":
                block_digests = [content_digest(b.values) for b in run.blocks]
                self.ledger.commit_stage(self.spec.version, stage.stage_id, seq,
                                         run.summary(), block_digests)
            # Chains/features may be missing if a prior crash interrupted the
            # post-value persistence pass; (re)persist them, then replay-check.
            existing_chain, _ = self.ledger.get_chain(self.spec.version, stage.stage_id)
            if existing_chain is None:
                self._persist_feature(stage, run, self.spec.window)
                continue
            self._replay_compare(stage, run, replay_issues)
        if replay_issues:
            first = replay_issues[0]
            raise ReplayError(
                f"replay mismatch at stage {first['stage_id']} field {first['field']}",
                location=first["stage_id"], field=first["field"],
                detail=first)
        self._write_cursor()
        return self.report()

    def _replay_compare(self, stage, run, issues: list) -> None:
        summary, _ = self.ledger.get_summary(self.spec.version, stage.stage_id)
        fresh_summary = run.summary()
        for field in ("digest", "nulls", "saturated", "boundary_uncertain",
                      "nonpos_skipped", "scale_exp"):
            if summary.get(field) != fresh_summary.get(field):
                issues.append({"stage_id": stage.stage_id, "field": field,
                               "ledger": summary.get(field),
                               "replay": fresh_summary.get(field)})
                return
        chain, _ = self.ledger.get_chain(self.spec.version, stage.stage_id)
        fresh_chain = self.engine.error_bounds[stage.stage_id]
        if chain.get("bound_ulp") != fresh_chain.get("bound_ulp"):
            issues.append({"stage_id": stage.stage_id, "field": "bound_ulp",
                           "ledger": chain.get("bound_ulp"),
                           "replay": fresh_chain.get("bound_ulp")})
            return
        verdict, _ = self.ledger.get_verdict(self.spec.version, stage.stage_id)
        w = self.spec.window
        ref, obs = self._window_stats_pair(run, w)
        fresh_verdict = evaluate_feature(
            stage.stage_id, ref, obs,
            self.spec.thresholds_bps[stage.stage_id],
            self._feature_bound_q(stage.stage_id),
            self._uncertain_rows(run), run.scale_exp)
        for field in ("drift_bps", "error_bps"):
            if verdict.get(field) != fresh_verdict.get(field):
                issues.append({"stage_id": stage.stage_id, "field": field,
                               "ledger": verdict.get(field),
                               "replay": fresh_verdict.get(field)})
                return
        if verdict["verdict"]["verdict"] != fresh_verdict["verdict"]["verdict"]:
            issues.append({"stage_id": stage.stage_id, "field": "verdict",
                           "ledger": verdict["verdict"]["verdict"],
                           "replay": fresh_verdict["verdict"]["verdict"]})

    # -- invalidation / increment -----------------------------------------
    def affected_stages(self, changed_stage_ids: set[str],
                        changed_sources: set[str]) -> set[str]:
        affected = set(changed_stage_ids)
        for s in self.spec.sources:
            if s in changed_sources:
                for stage in self.spec.stages:
                    if stage.source_id == s:
                        affected.add(stage.stage_id)
        # Forward propagation along refs in declared order until fixpoint.
        changed = True
        while changed:
            changed = False
            for stage in self.spec.stages:
                if stage.stage_id in affected:
                    continue
                if any(r in affected for r in stage.refs):
                    affected.add(stage.stage_id)
                    changed = True
        return affected

    def verify_incremental(self, full_report: dict,
                           incremental_report: dict) -> None:
        full_map = {f["stage_id"]: f for f in full_report["features"]}
        for f in incremental_report["features"]:
            ref = full_map.get(f["stage_id"])
            if ref is None:
                raise DivergeError("incremental produced unknown feature",
                                   location=f["stage_id"], field="stage_id")
            for field in ("drift_bps", "error_bps", "scale_exp"):
                if ref.get(field) != f.get(field):
                    raise DivergeError(
                        f"incremental/full mismatch at {field}",
                        location=f["stage_id"], field=field,
                        detail={"full": ref.get(field), "incremental": f.get(field)})
            if ref["verdict"]["verdict"] != f["verdict"]["verdict"]:
                raise DivergeError("incremental/full verdict mismatch",
                                   location=f["stage_id"], field="verdict")

    def report(self) -> dict:
        features, stages, chains = [], [], []
        for stage in self.spec.stages:
            summary, _ = self.ledger.get_summary(self.spec.version, stage.stage_id)
            chain, _ = self.ledger.get_chain(self.spec.version, stage.stage_id)
            verdict, _ = self.ledger.get_verdict(self.spec.version, stage.stage_id)
            if summary:
                stages.append(summary)
            if chain:
                chains.append(chain)
            if verdict:
                features.append(verdict)
        return {"spec_version": self.spec.version,
                "session_scale_exp": self.spec.session_scale_exp,
                "stages": stages, "chains": chains, "features": features}

    def close(self) -> None:
        self.ledger.close()
