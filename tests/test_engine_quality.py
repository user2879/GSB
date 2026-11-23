"""Concurrency determinism, injection isolation, snapshots and budget."""

import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..")))

from escalation_arbiter.app import ArbiterApp
from escalation_arbiter.engine import (execute_command, restore_or_rebuild,
                                       write_snapshot)
from escalation_arbiter.samplegen import generate


def _view(app):
    return {
        tid: {
            "score": app.state.decisions[tid].decision_score,
            "boundary": app.state.decisions[tid].boundary,
            "priority": app.state.decisions[tid].priority_level,
            "reasons": app.state.decisions[tid].reason_codes,
        }
        for tid in app.state.ordered_tids()
        if tid in app.state.decisions
    }


def test_concurrency_levels_agree():
    root = tempfile.mkdtemp(prefix="arb_conc_")
    try:
        generate(os.path.join(root, "samples"))
        views = []
        for workers in (1, 4, 8):
            app = ArbiterApp(os.path.join(root, "samples"),
                             os.path.join(root, f"w{workers}"))
            object.__setattr__(app.state.config, 'concurrency', workers)
            app.run_full()
            views.append(_view(app))
        assert views[0] == views[1] == views[2]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_injection_isolation_and_codes():
    root = tempfile.mkdtemp(prefix="arb_inj_")
    try:
        generate(os.path.join(root, "samples"))
        app = ArbiterApp(os.path.join(root, "samples"),
                         os.path.join(root, "work"))
        app.run_full()
        codes = {hit["inj_id"]: hit["code"]
                 for hit in app.state.injection_hits}
        assert codes["INJ-001"] == "E_EVIDENCE_INVALID"
        assert codes["INJ-002"] == "E_RULE_CYCLE"
        assert codes["INJ-003"] == "E_RULE_UNKNOWN"
        assert codes["INJ-004"] == "E_FIXED_WIDTH"
        assert len(app.state.decisions) == 8
        app.report()
        report = json.load(open(
            os.path.join(app.work_dir, "diagnostic_report.json")))
        assert len(report["injection_hits"]) == 4
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_snapshot_restore_and_stale():
    root = tempfile.mkdtemp(prefix="arb_snap_")
    try:
        generate(os.path.join(root, "samples"))
        app = ArbiterApp(os.path.join(root, "samples"),
                         os.path.join(root, "work"))
        app.run_full()
        write_snapshot(app.state)
        restored = ArbiterApp(os.path.join(root, "samples"),
                              os.path.join(root, "work2"))
        # Point second state at the same snapshot directory tree.
        import shutil as _sh
        _sh.copytree(os.path.join(root, "work", "snapshot"),
                     os.path.join(root, "work2", "snapshot"),
                     dirs_exist_ok=True)
        result = restore_or_rebuild(restored.state)
        assert result["restored"] is True

        stale = ArbiterApp(os.path.join(root, "samples"),
                           os.path.join(root, "work3"))
        object.__setattr__(stale.state.config, "budget_ms", 1)
        _sh.copytree(os.path.join(root, "work", "snapshot"),
                     os.path.join(root, "work3", "snapshot"),
                     dirs_exist_ok=True)
        outcome = restore_or_rebuild(stale.state)
        assert outcome["code"] == "E_SNAPSHOT_STALE"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_budget_exceeded_returns_stale_summary():
    root = tempfile.mkdtemp(prefix="arb_budget_")
    try:
        generate(os.path.join(root, "samples"))
        app = ArbiterApp(os.path.join(root, "samples"),
                         os.path.join(root, "work"))
        app.run_full()
        object.__setattr__(app.state.config, "budget_ms", 0)
        summary = execute_command(
            app.state,
            {"op": "set_half_life", "half_life_ms": 12345},
            compute_delay_ms=5)
        assert summary["budget_exceeded"] is True
        assert summary["parameters_stale"] is True
        assert any(f["code"] == "E_INTERACT_BUDGET_EXCEEDED"
                   for f in summary["failures"])
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_validation_failure_codes():
    root = tempfile.mkdtemp(prefix="arb_load_")
    try:
        samples = os.path.join(root, "samples")
        generate(samples)
        rules_path = os.path.join(samples, "rules.json")
        rules = json.load(open(rules_path))
        rules["mutex"].append({"rid": "99", "kind": "mutex",
                               "left": "LOGIN", "right": "LOGIN"})
        rules["reason_codes"].append(dict(rules["reason_codes"][0]))
        rules["priority_tiers"][1]["limit"] = rules["priority_tiers"][0]["limit"]
        json.dump(rules, open(rules_path, "w"))
        app = ArbiterApp(samples, os.path.join(root, "work"))
        codes = {f.code for f in app.load_failures}
        assert "E_RULE_CYCLE" in codes
        assert "E_DICT_DUPLICATE" in codes
        assert "E_THRESHOLD_ORDER" in codes
    finally:
        shutil.rmtree(root, ignore_errors=True)
