from __future__ import annotations

from joinlab import load_workload
from joinlab.ledger import Ledger
from joinlab.session import ExperimentSession

from conftest import SAMPLE


def test_change_set_replaces_whole_view_and_marks_old_generation_stale(tmp_path):
    session = ExperimentSession(load_workload(SAMPLE))
    with Ledger(tmp_path / "invalidate.sqlite") as ledger:
        session.run(ledger)
        old_run_id = session.run_id
        old_generation = session.generation
        old_candidate_ids = list(session.current_view["candidate_ids"])
        old_hist_version = session.workload.table_map()["orders"].histograms["amount"].version
        report = session.apply_change_set(
            {
                "table_row_counts": {"orders": 200},
                "fixed_point_scale": 2000,
                "max_candidates": 12,
                "exact_scan_threshold_rows": 250,
            },
            ledger=ledger,
        )
        assert report["run_id"] != old_run_id
        assert report["generation"] == old_generation + 1
        assert report["view_diff"]["replacement_mode"] == "whole_view"
        assert len(session.estimates) == 12
        assert len(report["view_diff"]["removed"]) == 6
        assert report["view_diff"]["stale_estimate_ids"]
        assert session.stale_conclusions[0]["generation"] == old_generation
        assert old_candidate_ids != session.current_view["candidate_ids"]
        assert session.workload.table_map()["orders"].histograms["amount"].version == old_hist_version + 1
        assert all(
            item.selected.caliber == "exact" for item in session.stats.filter_stats
        )
        assert session.selection["generation"] == old_generation + 1
        assert session.selection["stale_generations_excluded"] == [old_generation]
        event = ledger.connection.execute(
            "SELECT COUNT(*) AS n FROM invalidation_events WHERE run_id=?",
            (session.run_id,),
        ).fetchone()
        assert event["n"] == 1


def test_histogram_override_affects_estimate_and_ledger_integrity(tmp_path):
    session = ExperimentSession(load_workload(SAMPLE))
    old_estimate_digest = session.best_estimate.digest
    with Ledger(tmp_path / "histchange.sqlite") as ledger:
        session.run(ledger)
        customers = session.workload.table_map()["customers"]
        hist = customers.histograms["id"]
        bins = [
            {"lo": int(bin.lo), "hi": int(bin.hi), "count": bin.count}
            for bin in hist.bins
        ]
        bins[0]["count"] += 1
        bins[1]["count"] -= 1
        report = session.apply_change_set(
            {"histograms": {"customers": {"id": {"version": 2, "bins": bins}}}},
            ledger=ledger,
        )
        assert report["view_diff"]["changed_tables"] == ["customers"]
        assert session.workload.table_map()["customers"].histograms["id"].version == 2
        assert session.best_estimate.digest != old_estimate_digest
        assert ledger.verify_integrity()["ok"] is True
