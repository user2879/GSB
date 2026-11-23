"""Three-version serialization and lossy sidecar auditing."""

import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..")))

from escalation_arbiter.app import ArbiterApp
from escalation_arbiter.samplegen import generate
from escalation_arbiter.serialization import deserialize, serialize


def test_all_versions_preserve_logical_scores():
    root = tempfile.mkdtemp(prefix="arb_ver_")
    try:
        generate(os.path.join(root, "samples"))
        app = ArbiterApp(os.path.join(root, "samples"),
                         os.path.join(root, "work"))
        app.run_full()
        decisions = [app.state.decisions[tid]
                     for tid in app.state.ordered_tids()
                     if tid in app.state.decisions]
        exchange = os.path.join(root, "exchange")
        os.makedirs(exchange, exist_ok=True)
        for version in ("v1", "v2", "v3"):
            serialize(version, decisions, app.rules, app.config, exchange)
            restored = deserialize(version, exchange)
            baseline = {d.tid: d for d in decisions}
            for decision in restored:
                original = baseline[decision.tid]
                assert decision.decision_score == original.decision_score
                assert decision.reason_codes == original.reason_codes
                assert decision.boundary == original.boundary
                assert decision.priority_level == original.priority_level
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_v1_sidecar_records_reason_loss():
    root = tempfile.mkdtemp(prefix="arb_side_")
    try:
        generate(os.path.join(root, "samples"))
        app = ArbiterApp(os.path.join(root, "samples"),
                         os.path.join(root, "work"))
        app.run_full()
        app.state.rules.reason_dict["MX77"] = "synthetic"
        target = next(iter(app.state.decisions.values()))
        target.reason_codes = sorted(set(target.reason_codes + ["MX77"]))
        exchange = os.path.join(root, "exchange")
        os.makedirs(exchange, exist_ok=True)
        serialize("v1", list(app.state.decisions.values()), app.rules,
                  app.config, exchange)
        sidecar = json.load(open(os.path.join(exchange,
                                              "sidecar_v1.json")))
        assert any(loss["original"] == "MX77"
                   for loss in sidecar["losses"])
        restored = deserialize("v1", exchange)
        found = next(d for d in restored if d.tid == target.tid)
        assert "MX77" in found.reason_codes
    finally:
        shutil.rmtree(root, ignore_errors=True)
