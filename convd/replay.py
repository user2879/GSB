"""Recovery and replay verification after forced termination.

Recovery rule: only fully committed rounds (cursor) are authoritative;
rounds beyond the cursor are UNKNOWN and must be re-run. Replay
re-orchestrates the committed rounds from the ledger's recorded scenario
/ budget versions and compares every per-executor conclusion, anomaly
set and trajectory field against the ledger; the first divergence is
reported as E_REPLAY with round, executor id and field.
"""

import os
import tempfile

from . import LEDGER_FORMAT_VERSION
from .errors import ReplayError
from .ledger import Ledger
from .orchestrator import Orchestrator


def check_meta(ledger, scenario, executor_set):
    format_version = ledger.get_meta("format_version")
    if format_version != LEDGER_FORMAT_VERSION:
        raise ReplayError(0, None, "format_version",
                          LEDGER_FORMAT_VERSION, format_version)
    if ledger.get_meta("scenario_version") != scenario.digest:
        raise ReplayError(0, None, "scenario_version",
                          scenario.digest, ledger.get_meta("scenario_version"))
    if ledger.get_meta("budget_version") != scenario.budget_version:
        raise ReplayError(0, None, "budget_version",
                          scenario.budget_version,
                          ledger.get_meta("budget_version"))
    if ledger.get_meta("executor_set_version") != \
            scenario.executor_set_version:
        raise ReplayError(0, None, "executor_set_version",
                          scenario.executor_set_version,
                          ledger.get_meta("executor_set_version"))


def unknown_rounds(ledger, scenario):
    """Rounds past the committed cursor: unknown, must be re-run."""
    cursor = ledger.cursor()
    return list(range(cursor + 1, scenario.max_rounds + 1))


def _compare_rounds(expected_entries, actual_entries):
    expected_map = {(r, e): entry for r, e, entry in expected_entries}
    actual_map = {(r, e): entry for r, e, entry in actual_entries}
    for key in sorted(expected_map):
        round_no, executor_id = key
        if key not in actual_map:
            raise ReplayError(round_no, executor_id, "<entry>",
                              expected_map[key], None)
        exp, act = expected_map[key], actual_map[key]
        for field in sorted(set(exp) | set(act)):
            if exp.get(field) != act.get(field):
                raise ReplayError(round_no, executor_id, field,
                                  exp.get(field), act.get(field))


def _compare_anomalies(expected, actual):
    exp_list = [{"round": r, "executor_id": e, "kind": k, "payload": p}
                for r, e, k, p in expected]
    act_list = [{"round": r, "executor_id": e, "kind": k, "payload": p}
                for r, e, k, p in actual]
    for exp, act in zip(exp_list, act_list):
        for field in ("round", "executor_id", "kind", "payload"):
            if exp[field] != act[field]:
                raise ReplayError(exp["round"], exp["executor_id"],
                                  "anomaly." + field, exp[field],
                                  act[field])
    if len(exp_list) != len(act_list):
        last = exp_list[min(len(exp_list), len(act_list))] \
            if exp_list and act_list else {"round": 0, "executor_id": None}
        raise ReplayError(last["round"], last["executor_id"],
                          "anomaly.count", len(exp_list), len(act_list))


def replay_verify(scenario, executor_set, ledger, base_ms=1, jitter_ms=4,
                  drop_map=None, up_to=None):
    """Re-orchestrate committed rounds and diff against the ledger."""
    check_meta(ledger, scenario, executor_set)
    cursor = ledger.cursor()
    target = min(up_to or cursor, cursor)
    if target == 0:
        return {"replayed_rounds": 0, "digest_match": True}
    expected_entries = [(r, e, entry) for r, e, entry
                        in ledger.round_entries() if r <= target]
    expected_anomalies = [(r, e, k, p) for r, e, k, p
                          in ledger.anomaly_entries() if r <= target]
    fd, tmp_path = tempfile.mkstemp(suffix=".replay.db")
    os.close(fd)
    try:
        replay_ledger = Ledger(tmp_path)
        replay_ledger.set_meta(
            format_version=LEDGER_FORMAT_VERSION,
            scenario_id=scenario.scenario_id,
            scenario_version=scenario.digest,
            budget_version=scenario.budget_version,
            executor_set_version=scenario.executor_set_version)
        with Orchestrator(scenario, executor_set, replay_ledger,
                          base_ms=base_ms, jitter_ms=jitter_ms,
                          drop_map=drop_map) as orch:
            orch.run(start_round=1, up_to=target)
        _compare_rounds(expected_entries, replay_ledger.round_entries())
        _compare_anomalies(expected_anomalies,
                           replay_ledger.anomaly_entries())
        digest_match = (ledger.content_digest()
                        == replay_ledger.content_digest()) \
            if target == cursor else True
        return {"replayed_rounds": target, "digest_match": digest_match}
    finally:
        replay_ledger.close()
        os.unlink(tmp_path)
