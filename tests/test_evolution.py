from __future__ import annotations

import json
from pathlib import Path

from hopledger.ledger import Ledger
from hopledger.proc import ProcessManager
from hopledger.runner import Runner
from hopledger.sample import make_sample_configs


def _started_runner(tmp_path: Path):
    bundle = make_sample_configs(tmp_path / "config")
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    manager = ProcessManager(bundle, tmp_path / "logs")
    manager.start_all()
    runner = Runner(bundle, ledger, manager=manager)
    runner.run_all()
    return bundle, ledger, manager, runner


def test_config_version_evolution_recomputes_only_affected_receipts(tmp_path: Path):
    bundle, ledger, manager, runner = _started_runner(tmp_path)
    try:
        before = ledger.digest_index()
        chain_path = bundle.config_dir / "chain.json"
        data = json.loads(chain_path.read_text())
        data["hops"][0]["config_version"] = 2
        chain_path.write_text(json.dumps(data))

        runner.reload_bundle()
        manager.restart_node("fwd1")
        plan = runner.invalidation_plan()

        assert plan["affected_request_ids"] == bundle.request_ids
        assert sorted(plan["redispatch_nodes"]) == sorted(
            f"receipt:{request_id}:fwd1" for request_id in bundle.request_ids
        )
        assert any(
            step["reason"] == "downstream_chain_sum_depends_on_prior_accumulation"
            for step in plan["steps"]
        )

        result = runner.apply_invalidation(plan)
        assert result["unchanged_byte_check_passed"] is True
        assert result["digest_violations"] == []

        after = ledger.digest_index()
        for request_id in bundle.request_ids:
            assert after[f"receipt:{request_id}:fwd1"] != before[f"receipt:{request_id}:fwd1"]
            assert after[f"receipt:{request_id}:rev2"] == before[f"receipt:{request_id}:rev2"]
            assert after[f"receipt:{request_id}:origin"] == before[f"receipt:{request_id}:origin"]
            # A version-only change does not alter numeric error attribution.
            assert after[f"attribution:{request_id}"] == before[f"attribution:{request_id}"]
            attribution = ledger.get_attribution(request_id)
            assert attribution["dominant_hop"] == "fwd1"
        assert ledger.verify_content_digest()
    finally:
        manager.stop_all()
        ledger.close()


def test_scale_factor_evolution_targets_changed_hop(tmp_path: Path):
    bundle, ledger, manager, runner = _started_runner(tmp_path)
    try:
        before = ledger.digest_index()
        chain_path = bundle.config_dir / "chain.json"
        data = json.loads(chain_path.read_text())
        data["hops"][1]["scale_factor"] = 10_000
        chain_path.write_text(json.dumps(data))

        runner.reload_bundle()
        manager.restart_node("rev2")
        plan = runner.invalidation_plan()
        assert sorted(plan["redispatch_nodes"]) == sorted(
            f"receipt:{request_id}:rev2" for request_id in bundle.request_ids
        )

        result = runner.apply_invalidation(plan)
        assert result["digest_violations"] == []
        after = ledger.digest_index()
        for request_id in bundle.request_ids:
            assert after[f"receipt:{request_id}:rev2"] != before[f"receipt:{request_id}:rev2"]
            assert after[f"receipt:{request_id}:fwd1"] == before[f"receipt:{request_id}:fwd1"]
            conclusion = ledger.get_conclusion(request_id)
            # Millisecond totals remain exact; the finer scale changes error bounds.
            assert conclusion["e2e_total_units"] == 6000
            assert conclusion["used_scale_factors"]["rev2"] == 10_000
        assert ledger.verify_content_digest()
    finally:
        manager.stop_all()
        ledger.close()


def test_scale_caliber_evolution_recomputes_without_redispatch(tmp_path: Path):
    bundle, ledger, manager, runner = _started_runner(tmp_path)
    try:
        before = ledger.digest_index()
        diagnostic_path = bundle.config_dir / "diagnostic.json"
        data = json.loads(diagnostic_path.read_text())
        data["scale_caliber_version"] = 2
        diagnostic_path.write_text(json.dumps(data))

        runner.reload_bundle()
        plan = runner.invalidation_plan()
        assert plan["redispatch_nodes"] == []
        assert any(step["reason"] == "scale_caliber_version_changed"
                   for step in plan["steps"])

        result = runner.apply_invalidation(plan)
        assert result["digest_violations"] == []
        after = ledger.digest_index()
        for request_id in bundle.request_ids:
            for node_id in ("fwd1", "rev2", "origin"):
                key = f"receipt:{request_id}:{node_id}"
                assert after[key] == before[key]
            assert after[f"conclusion:{request_id}"] != before[f"conclusion:{request_id}"]
            assert ledger.get_conclusion(request_id)["scale_caliber_version"] == 2
    finally:
        manager.stop_all()
        ledger.close()
