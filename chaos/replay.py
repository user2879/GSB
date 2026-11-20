"""Deterministic replay verification of a recovered ledger.

Recomputes the timeline from (spec, seed) and recomputes every observation
summary from the ledger's own per-request records (the ground truth), then
compares field-by-field with what the ledger claims. The first mismatch is
reported as E_REPLAY locating the injection seq and the field.
"""

import json

from .errors import ChaosError, E_REPLAY
from .fixedpoint import summarize

PHASES = ("pre", "active", "post")


def _fail(seq, field, expected, actual):
    raise ChaosError(
        E_REPLAY,
        "replay mismatch at injection seq=%s field=%s" % (seq, field),
        field="injections[%d].%s" % (seq, field),
        evidence={"seq": seq, "field": field,
                  "expected": expected, "actual": actual})


def replay(spec, timeline_entries, ledger):
    by_seq = {e.seq: e for e in timeline_entries}
    rows = ledger.conn.execute(
        "SELECT seq, stable_key, itype, target, params_json, planned_start_ms,"
        " state, committed FROM injections ORDER BY seq").fetchall()
    for (seq, stable_key, itype, target, params_json,
         planned_start, state, committed) in rows:
        entry = by_seq.get(seq)
        if entry is None:
            _fail(seq, "seq", "present in timeline", "absent")
        if entry.stable_key != stable_key:
            _fail(seq, "stable_key", entry.stable_key, stable_key)
        if entry.type != itype:
            _fail(seq, "type", entry.type, itype)
        if entry.target != target:
            _fail(seq, "target", entry.target, target)
        if json.dumps(entry.params, sort_keys=True) != json.dumps(
                json.loads(params_json), sort_keys=True):
            _fail(seq, "params", entry.params, json.loads(params_json))
        if entry.planned_start_ms != planned_start:
            _fail(seq, "planned_start_ms", entry.planned_start_ms, planned_start)
        if not committed:
            continue  # unknown rows carry no completed observation to replay
        for phase in PHASES:
            req = ledger.conn.execute(
                "SELECT ok, latency_ms FROM requests"
                " WHERE inj_seq=? AND phase=? ORDER BY req_idx",
                (seq, phase)).fetchall()
            obs = ledger.conn.execute(
                "SELECT samples, success, fail, rate_fixed, rate_err_bound,"
                " p50_fixed, p95_fixed, quant_err_bound, scale, truncation,"
                " decidable FROM observations WHERE inj_seq=? AND phase=?",
                (seq, phase)).fetchone()
            if obs is None:
                _fail(seq, phase + ".observation", "present", "absent")
            recomputed = summarize([r[1] for r in req], [r[0] for r in req],
                                   obs[8], spec.sampling.min_samples)
            actual = {
                "samples": obs[0], "success": obs[1], "fail": obs[2],
                "rate_fixed": obs[3], "rate_err_bound": obs[4],
                "p50_fixed": obs[5], "p95_fixed": obs[6],
                "quant_err_bound": obs[7], "scale": obs[8],
                "truncation": obs[9], "decidable": obs[10],
            }
            for field, want in recomputed.items():
                if actual[field] != want:
                    _fail(seq, phase + "." + field, want, actual[field])
    return len(rows)
