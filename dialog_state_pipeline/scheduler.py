"""Concurrent scheduler with global budget and per-session quotas.

Determinism contract despite threading:

* session *ordering* is always lexicographic by session id, with budget
  pending sessions from a previous run visited first;
* the global budget ledger is written exclusively by this orchestrator
  thread - workers never mutate shared budget state;
* each round grants at most one state entry per still-unfinished session, so
  total grants in a round can never exceed the remaining global budget;
* worker outputs are merged back in stable session order, not completion
  order, which keeps the deduction ledger reproducible run after run.

One processed turn consumes one global budget entry and one per-session
quota entry.  When the global budget is exhausted, unfinished sessions keep
their remaining turns as todos and are marked ``E_BUDGET_PENDING``; a
follow-up run re-enters them first.
"""

from concurrent.futures import ThreadPoolExecutor

from .engine import compute_keys, longest_reused_prefix, run_session
from .state_machine import build_transition_index


class Scheduler:
    def __init__(self, fsm, config, cache, budget_remaining,
                 budget_consumed=0, ledger=None, quota_injection_ids=None):
        self.fsm = fsm
        self.config = config
        self.cache = cache                      # key -> TurnCheckpoint
        self.transition_index = build_transition_index(fsm)
        self.budget_remaining = int(budget_remaining)
        self.budget_consumed = int(budget_consumed)
        self.ledger = list(ledger or [])       # auditable deduction sequence
        self.pending = []                       # session ids waiting on budget
        self.results = {}                       # session id -> SessionResult
        self._seq = max((row["seq"] for row in self.ledger), default=0)
        self.quota_injection_ids = dict(quota_injection_ids or {})

    # ------------------------------------------------------------------ run
    def run(self, turns_by_session, pending_ids=None):
        pending_ids = list(pending_ids or [])
        ordered = self._ordered_ids(turns_by_session, pending_ids)

        # Zero remaining budget: nothing can advance this run.
        if self.budget_remaining <= 0:
            for sid in ordered:
                if self._remaining_work(sid, turns_by_session) > 0:
                    self._ensure_pending(sid, turns_by_session)
            return self.results

        active = [sid for sid in ordered
                  if self._remaining_work(sid, turns_by_session) > 0]
        while active and self.budget_remaining > 0:
            # One entry per active session per round; total <= remaining.
            plan = [sid for sid in active
                    if self._quota_room(sid) > 0][:self.budget_remaining]
            if not plan:
                break

            with ThreadPoolExecutor(max_workers=self.config.max_workers) as pool:
                futures = {
                    pool.submit(self._invoke, sid, turns_by_session[sid]): sid
                    for sid in plan
                }
                worker_outputs = {futures[f]: f.result() for f in futures}

            for sid in plan:  # merge in stable session order
                result, new_checkpoints, _keys = worker_outputs[sid]
                for checkpoint in new_checkpoints:
                    self.cache[checkpoint.key] = checkpoint
                self._reserve(sid)
                for turn_no in result.processed_turns:
                    self._consume(sid, turn_no)
                    self.budget_remaining -= 1
                    self.budget_consumed += 1
                    self.ledger[-1]["remaining_after"] = self.budget_remaining
                self._drop_pending_marker(result)
                self.results[sid] = result

            active = [sid for sid in ordered
                      if self._remaining_work(sid, turns_by_session) > 0
                      and self._quota_room(sid) > 0]

        # Final classification: quota failures vs budget-retained todos.
        for sid in ordered:
            result = self.results.get(sid)
            if result is None:
                continue
            self._drop_pending_marker(result)
            if result.status == "quota_exceeded":
                continue
            if self._remaining_work(sid, turns_by_session) > 0:
                result.status = "budget_pending"
                self._ensure_pending(sid, turns_by_session)
            elif sid in self.pending:
                self.pending.remove(sid)
        return self.results

    # -------------------------------------------------------------- helpers
    def _invoke(self, session_id, turns):
        return run_session(session_id, turns, self.fsm, self.config,
                           self.cache, self.transition_index,
                           quota=self.config.quota_for(session_id),
                           turn_limit=1,
                           quota_injection_id=self.quota_injection_ids.get(session_id))

    def _ordered_ids(self, turns_by_session, pending_ids):
        all_ids = sorted(turns_by_session)
        pending = [sid for sid in dict.fromkeys(pending_ids) if sid in all_ids]
        rest = [sid for sid in all_ids if sid not in pending]
        return pending + rest

    def _reused_count(self, sid, turns):
        result = self.results.get(sid)
        if result is not None:
            return len(result.reused_turns)
        keys = compute_keys(turns, self.fsm, self.config)
        return longest_reused_prefix(keys, self.cache)

    def _remaining_work(self, sid, turns_by_session):
        turns = turns_by_session[sid]
        result = self.results.get(sid)
        produced = len(result.processed_turns) if result else 0
        return len(turns) - self._reused_count(sid, turns) - produced

    def _quota_room(self, sid):
        result = self.results.get(sid)
        consumed = result.consumed_entries if result else 0
        return max(0, self.config.quota_for(sid) - consumed)

    def served_turns(self, sid, turns):
        """Turn numbers already available to this scheduler for *sid*.

        Counts turns produced in this run plus every turn whose hash-chain
        key is present in the shared cache, which covers prefixes restored
        from a snapshot even when the current run processed nothing.
        """
        result = self.results.get(sid)
        produced = set(result.processed_turns) if result else set()
        keys = compute_keys(turns, self.fsm, self.config)
        cached = {turn["turn"] for turn, key in zip(turns, keys)
                  if key in self.cache}
        return produced | cached

    def _drop_pending_marker(self, result):
        result.failures = [f for f in result.failures
                           if f["code"] != "E_BUDGET_PENDING"]

    def _ensure_pending(self, sid, turns_by_session):
        if sid not in self.pending:
            self.pending.append(sid)
        result = self.results.get(sid)
        turn_no = None
        if result is not None:
            done = set(result.processed_turns) | set(result.reused_turns)
            leftover = [t["turn"] for t in turns_by_session[sid]
                        if t["turn"] not in done]
            turn_no = leftover[0] if leftover else None
            self._drop_pending_marker(result)
            result.status = "budget_pending"
            result.failures.append({
                "session_id": sid,
                "turn": turn_no,
                "stage": "schedule",
                "code": "E_BUDGET_PENDING",
                "message": "global state entry budget exhausted; todo retained for resume",
                "batch_id": "",
                "injection_id": None,
            })

    def _reserve(self, sid):
        self._seq += 1
        self.ledger.append({
            "seq": self._seq, "kind": "reserve", "session_id": sid,
            "entries": 1, "remaining_after": self.budget_remaining,
        })

    def _consume(self, sid, turn_no):
        self._seq += 1
        self.ledger.append({
            "seq": self._seq, "kind": "consume", "session_id": sid,
            "turn": turn_no, "entries": 1,
            "remaining_after": self.budget_remaining,
        })
