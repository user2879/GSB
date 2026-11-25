from __future__ import annotations

import sys
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from category_cross_diagnostic.comparison import deep_diff, normalize_report
from category_cross_diagnostic.config import load_bundle_from_dicts
from category_cross_diagnostic.engine import DiagnosticEngine
from category_cross_diagnostic.sample_data import generate_table


def small_dicts() -> tuple[dict, dict]:
    features = {
        "seed": 5150,
        "row_count": 160,
        "columns": [
            {"name": "alpha", "type": "categorical", "cardinality": 9, "missing_rate": 0.04},
            {"name": "beta", "type": "categorical", "cardinality": 11, "missing_rate": 0.03},
            {"name": "gamma", "type": "categorical", "cardinality": 13, "missing_rate": 0.02},
            {"name": "score", "type": "numeric", "low": 0, "high": 100},
        ],
    }
    strategy = {
        "global_version": 1,
        "expansion_mode": "clique",
        "candidate_pairs": [["alpha", "beta"], ["alpha", "gamma"], ["beta", "gamma"]],
        "order_limit": 3,
        "cardinality_budget": 120,
        "point_budget": 1_000_000,
        "encoding_width": 12,
        "fixed_point_width": 16,
        "quantization_levels": 16,
        "over_budget_action": "truncate",
        "max_workers": 3,
        "snapshot_commit_interval": 1,
    }
    return features, strategy


def engine_from_dicts(
    features: dict,
    strategy: dict,
    state: Path,
    *,
    serial: bool = False,
    stop_event: threading.Event | None = None,
) -> DiagnosticEngine:
    bundle = load_bundle_from_dicts(features, strategy)
    assert bundle.ok, bundle.issues
    return DiagnosticEngine(
        bundle,
        generate_table(bundle),
        state_dir=state,
        serial=serial,
        stop_event=stop_event,
    )


class DiagnosticTests(unittest.TestCase):
    def test_validation_collects_every_required_failure_code(self) -> None:
        features, strategy = small_dicts()
        features["columns"].extend(
            [
                {"name": "dup", "type": "categorical", "cardinality": 2},
                {"name": "dup", "type": "categorical", "cardinality": 2},
            ]
        )
        strategy["candidate_pairs"] = [
            ["missing", "alpha"],
            ["score", "alpha"],
            ["alpha", "alpha"],
            ["alpha", "beta"],
            ["beta", "alpha"],
        ]
        strategy.update(
            order_limit=-1,
            cardinality_budget=0,
            point_budget=-2,
            encoding_width=7,
            fixed_point_width=5,
            quantization_levels=6,
        )
        bundle = load_bundle_from_dicts(features, strategy)
        codes = {issue["code"] for issue in bundle.issues}
        for code in (
            "E_UNKNOWN_COLUMN",
            "E_TYPE_MISMATCH",
            "E_ORDER_LIMIT_INVALID",
            "E_BUDGET_INVALID",
            "E_WIDTH_INVALID",
            "E_DUPLICATE_PAIR",
        ):
            self.assertIn(code, codes)
        self.assertGreaterEqual(len(bundle.columns), 4)

    def test_enumeration_order_and_deterministic_fingerprints(self) -> None:
        features, strategy = small_dicts()
        with tempfile.TemporaryDirectory() as tmp:
            report = engine_from_dicts(features, strategy, Path(tmp)).run()
        self.assertEqual(
            [item["combo_id"] for item in report["combinations"]],
            [
                "cross__alpha",
                "cross__beta",
                "cross__gamma",
                "cross__alpha__beta",
                "cross__alpha__gamma",
                "cross__beta__gamma",
                "cross__alpha__beta__gamma",
            ],
        )
        fingerprints = [item["combo_fingerprint"] for item in report["combinations"]]
        self.assertEqual(len(fingerprints), len(set(fingerprints)))

    def test_serial_and_parallel_reports_match_field_by_field(self) -> None:
        features, strategy = small_dicts()
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            serial = engine_from_dicts(
                features, strategy, root_path / "serial", serial=True
            ).run()
            parallel = engine_from_dicts(
                features, strategy, root_path / "parallel", serial=False
            ).run()
            diffs = deep_diff(
                normalize_report(serial, cache_status=True),
                normalize_report(parallel, cache_status=True),
            )
            self.assertEqual(diffs, [])
            self.assertLessEqual(
                parallel["concurrency"]["observed_max_active_tasks"],
                strategy["max_workers"],
            )

    def test_budget_change_reuses_upstream_stages(self) -> None:
        features, strategy = small_dicts()
        with tempfile.TemporaryDirectory() as root:
            state = Path(root) / "state"
            engine_from_dicts(features, strategy, state).run()
            strategy["cardinality_budget"] = 12
            report = engine_from_dicts(features, strategy, state).run()
            statuses = {
                (entry["stage"], entry["status"]) for entry in report["cache_ledger"]
            }
            self.assertIn(("combination_enumeration", "reused"), statuses)
            self.assertIn(("encoding_dictionary_summary", "reused"), statuses)
            self.assertIn(("cardinality_diagnosis", "reused"), statuses)
            decision_statuses = {
                entry["status"]
                for entry in report["cache_ledger"]
                if entry["stage"] == "budget_decision"
            }
            self.assertEqual(decision_statuses, {"recomputed"})

    def test_column_replacement_invalidates_only_dependent_combos(self) -> None:
        features, strategy = small_dicts()
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            state = root_path / "incremental"
            engine_from_dicts(features, strategy, state).run()

            bundle = load_bundle_from_dicts(features, strategy)
            table = generate_table(bundle)
            replacement = [
                None if index % 11 == 0 else f"alpha__new_{index % 5}"
                for index in range(table.row_count)
            ]
            table.replace_column("alpha", replacement)
            incremental = DiagnosticEngine(bundle, table, state_dir=state).run()

            fresh_bundle = load_bundle_from_dicts(features, strategy)
            fresh_table = generate_table(fresh_bundle)
            fresh_table.replace_column("alpha", replacement)
            fresh = DiagnosticEngine(
                fresh_bundle,
                fresh_table,
                state_dir=root_path / "fresh",
            ).run()

            diffs = deep_diff(
                normalize_report(incremental), normalize_report(fresh)
            )
            self.assertEqual(diffs, [])
            recomputed = {
                entry["combo_id"]
                for entry in incremental["cache_ledger"]
                if entry["status"] == "recomputed"
            }
            self.assertEqual(
                recomputed,
                {
                    "cross__alpha",
                    "cross__alpha__beta",
                    "cross__alpha__gamma",
                    "cross__alpha__beta__gamma",
                },
            )

    def test_dual_budget_and_pending_order(self) -> None:
        features, strategy = small_dicts()
        strategy["point_budget"] = 2000
        strategy["encoding_width"] = 16
        with tempfile.TemporaryDirectory() as tmp:
            report = engine_from_dicts(features, strategy, Path(tmp)).run()
        self.assertTrue(report["pending"])
        exhausted = [
            item
            for item in report["combinations"]
            if item["budget_decision"]["point_exhausted"]
        ]
        self.assertTrue(exhausted)
        boundary = exhausted[-1]["budget_decision"]
        self.assertEqual(
            [step["constraint"] for step in boundary["decision_sequence"]],
            ["cardinality_budget", "point_budget"],
        )
        self.assertIn(
            "E_POINT_BUDGET_EXHAUSTED",
            [item["code"] for item in report["pending"]],
        )

    def test_snapshot_resume_matches_fresh(self) -> None:
        features, strategy = small_dicts()
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            state = root_path / "state"
            engine = engine_from_dicts(features, strategy, state, serial=True)
            original_commit = engine._commit_snapshot  # noqa: SLF001

            def stop_after_commit(*args, **kwargs):
                result = original_commit(*args, **kwargs)
                engine.request_stop()
                return result

            engine._commit_snapshot = stop_after_commit  # type: ignore[method-assign]  # noqa: SLF001
            interrupted = engine.run()
            self.assertEqual(interrupted["status"], "interrupted")

            recovered = engine_from_dicts(features, strategy, state, serial=True).run()
            fresh = engine_from_dicts(
                features, strategy, root_path / "fresh", serial=True
            ).run()
            diffs = deep_diff(normalize_report(recovered), normalize_report(fresh))
            self.assertEqual(diffs, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
