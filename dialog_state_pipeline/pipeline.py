"""End-to-end pipeline orchestration with staged snapshot commits."""

import copy

from .config import PipelineConfig, load_config, load_fsm
from .injection import (
    apply_injections, load_injection, quota_injection_ids, quota_overrides,
)
from .loader import load_event_stream, _record_failure
from .report import build_report, write_report
from .scheduler import Scheduler
from .snapshot import SnapshotState, SnapshotStore


class Pipeline:
    def __init__(self, config: PipelineConfig, fsm, injection_config=None,
                 budget_refill_on_resume=False):
        self.config = config
        self.fsm = fsm
        self.injection_config = injection_config or {"injections": []}
        self.store = SnapshotStore(config.snapshot_dir)
        # When True, a resumed run treats the configured global_budget as a
        # fresh per-run allocation (new budget period); cumulative consumption
        # stays auditable in the snapshot/ledger.  When False, the budget is a
        # hard total ceiling that can permanently park sessions.
        self.budget_refill_on_resume = budget_refill_on_resume

    def run(self, event_path, report_path, mode="incremental", stop_after_stage=None):
        # ---------------- Stage 1: load & validate ----------------
        materialised, load_failures = load_event_stream(event_path)
        load_failures = [
            item.to_dict() if hasattr(item, "to_dict") else dict(item)
            for item in load_failures]
        materialised, applied_ids = apply_injections(materialised, self.injection_config)
        # Re-run the density rule so turn_jump injections surface as real
        # E_STREAM_INVALID faults (the bad turn is removed from the stream).
        materialised, jump_failures = _revalidate_density(materialised)
        load_failures.extend(
            item.to_dict() if hasattr(item, "to_dict") else dict(item)
            for item in jump_failures)

        # Snapshot compatibility is judged against the unmodified pipeline
        # config; injection quota overrides are applied only afterwards.
        state, cache, stale_failure = self.store.resume_or_reset(
            self.config.fingerprint, self.fsm.fingerprint)
        if state is None:
            state = SnapshotState(
                config_hash=self.config.fingerprint,
                fsm_hash=self.fsm.fingerprint,
            )
        # Use a run-scoped copy so injection overrides never mutate the
        # pipeline's declared config fingerprint across repeated runs.
        run_config = copy.deepcopy(self.config)
        overrides = quota_overrides(self.injection_config)
        if overrides:
            run_config = _with_quota_overrides(run_config, overrides)
        state.stage = "load"
        state.artifact_versions["load"] = _artefact_hash(materialised, load_failures)
        self.store.commit(state)
        if stop_after_stage == "load":
            return None

        # ---------------- Stage 2: concurrent advance ----------------
        remaining_budget = run_config.global_budget - state.budget_consumed
        if self.budget_refill_on_resume and remaining_budget <= 0:
            remaining_budget = run_config.global_budget
        prior_ledger = state.ledger
        scheduler = Scheduler(self.fsm, run_config, cache,
                              budget_remaining=remaining_budget,
                              budget_consumed=state.budget_consumed,
                              ledger=prior_ledger,
                              quota_injection_ids=quota_injection_ids(
                                  self.injection_config))
        scheduler.run(materialised, pending_ids=state.pending_sessions)

        state.stage = "advance"
        state.global_version += 1
        state.budget_consumed = scheduler.budget_consumed
        state.pending_sessions = list(scheduler.pending)
        state.ledger = list(scheduler.ledger)
        state.checkpoints = _sorted_checkpoints(cache)
        state.artifact_versions["advance"] = _hash_checkpoints(state.checkpoints)
        self.store.commit(state)
        if stop_after_stage == "advance":
            return None

        # ---------------- Stage 3: settle & report ----------------
        completed_batches = _completed_batches(materialised, scheduler)
        state.stage = "settle"
        state.completed_batches = completed_batches
        state.artifact_versions["settle"] = _hash_checkpoints(state.checkpoints)
        self.store.commit(state)

        report = build_report(
            scheduler.results, load_failures, stale_failure, scheduler,
            self.fsm, run_config, materialised, self.injection_config,
            applied_ids, completed_batches, mode)
        if report_path is not None:
            write_report(report, report_path)
        return report


def run_from_files(event_path, fsm_path, config_path, report_path,
                   injection_path=None, mode="incremental", stop_after_stage=None,
                   reset=False, refill_budget=False):
    config = load_config(config_path)
    fsm = load_fsm(fsm_path)
    injection_config = load_injection(injection_path)
    pipeline = Pipeline(config, fsm, injection_config,
                        budget_refill_on_resume=refill_budget)
    if reset:
        pipeline.store.reset()
    return pipeline.run(event_path, report_path, mode=mode,
                        stop_after_stage=stop_after_stage)


def _revalidate_density(materialised):
    failures = []
    cleaned = {}
    for sid in sorted(materialised):
        turns = materialised[sid]
        bad_turns = set()
        expected = 1
        for turn in turns:
            if turn["turn"] != expected:
                _record_failure(failures, sid, turn["turn"], "E_STREAM_INVALID",
                                f"turn sequence violation: expected {expected}, "
                                f"got {turn['turn']}",
                                batch_id=turn.get("batch_id", ""),
                                injection_id=turn.get("injection_id"))
                bad_turns.add(turn["turn"])
                continue
            expected += 1
        kept = [t for t in turns if t["turn"] not in bad_turns]
        for idx, turn in enumerate(kept, start=1):
            turn["turn"] = idx
        cleaned[sid] = kept
    return cleaned, failures


def _with_quota_overrides(config, overrides):
    cloned = copy.deepcopy(config)
    cloned.session_quotas.update(overrides)
    return cloned


def _sorted_checkpoints(cache):
    checkpoints = list(cache.values())
    checkpoints.sort(key=lambda cp: (cp.session_id, cp.turn))
    return checkpoints


def _hash_checkpoints(checkpoints):
    from .hashing import digest
    return digest([cp.to_dict() for cp in checkpoints])


def _artefact_hash(materialised, failures):
    from .hashing import digest
    return digest({"materialised": materialised, "load_failures": list(failures)})


def _completed_batches(materialised, scheduler):
    """A batch is completed when every turn carrying its id across every
    session has been processed or served from cache, and no owning session is
    still parked on the global budget."""
    batch_total = {}
    batch_done = {}
    for sid, turns in materialised.items():
        result = scheduler.results.get(sid)
        done_turns = scheduler.served_turns(sid, turns)
        for turn in turns:
            batch = turn.get("batch_id")
            if not batch:
                continue
            batch_total[batch] = batch_total.get(batch, 0) + 1
            if turn["turn"] in done_turns:
                batch_done[batch] = batch_done.get(batch, 0) + 1
    blocked_batches = set()
    for sid in scheduler.pending:
        blocked_batches.update(
            turn.get("batch_id") for turn in materialised.get(sid, [])
            if turn.get("batch_id"))
    # A session parked by its hard quota cannot complete any of its batches.
    for sid, result in scheduler.results.items():
        if result.status == "quota_exceeded":
            blocked_batches.update(
                turn.get("batch_id") for turn in materialised.get(sid, [])
                if turn.get("batch_id"))
    return sum(
        1 for batch, total in sorted(batch_total.items())
        if batch_done.get(batch, 0) == total and batch not in blocked_batches)
