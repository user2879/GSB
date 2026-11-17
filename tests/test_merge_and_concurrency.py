import tempfile
import unittest

from feature_pipeline.concurrency import run_entity_tasks
from feature_pipeline.errors import E_PRECISION_INVALID, PrecisionInvalidError
from feature_pipeline.fingerprint import config_fingerprint
from feature_pipeline.merge import merge_tables
from feature_pipeline.config import parse_merge_config
from feature_pipeline.models import V1, V2, V3, Cell, EntityTable


def config(**overrides):
    base = {
        "feature_whitelist": ["f_a"],
        "scaling_method": "raw",
        "missing_fill": {"strategy": "zero"},
        "conflict_priority": "source_version",
        "source_order": [V1, V2, V3],
        "precision_rank": ["float64", "float32", "per_mille"],
        "numeric_tolerance": 0.02,
        "max_workers": 3,
        "batch_size": 2,
    }
    base.update(overrides)
    return parse_merge_config(base)


def make_tables():
    # Same entity/feature across sources: one tolerant match, one real conflict.
    tables = []
    data = {
        V1: {"e1": 1.00, "e2": 1.00},
        V2: {"e1": 1.01, "e2": 2.00},
        V3: {"e1": 1.015, "e2": 3.00},
    }
    for version in (V1, V2, V3):
        table = EntityTable(source_version=version, feature_order=["f_a"])
        for entity, value in data[version].items():
            table.cells.append(
                Cell(entity, "f_a", version, value, False,
                     "per_mille" if version == V1 else
                     "float32" if version == V3 else "float64")
            )
        tables.append(table)
    return tables


class MergeTests(unittest.TestCase):
    def test_tolerance_uses_highest_priority_source(self) -> None:
        cfg = config()
        tables = make_tables()
        out = merge_tables(tables, cfg, config_fingerprint(cfg))
        rows = {(row["entity_id"], row["feature"]): row for row in out.rows}
        self.assertEqual(rows[("e1", "f_a")]["winning_source"], V1)
        self.assertEqual(rows[("e1", "f_a")]["value"], 1.00)
        # e1 spread is exactly at the tolerance boundary and stays consistent;
        # e2 spread 2.0 exceeds tolerance and becomes a conflict.
        self.assertEqual(len(out.conflicts), 1)
        self.assertEqual(out.conflicts[0]["entity_id"], "e2")
        self.assertEqual(out.conflicts[0]["winning_reason"],
                         "conflict_source_version_priority")
        self.assertEqual(rows[("e2", "f_a")]["winning_source"], V1)

    def test_abs_value_priority(self) -> None:
        cfg = config(conflict_priority="abs_value")
        out = merge_tables(make_tables(), cfg, config_fingerprint(cfg))
        rows = {(row["entity_id"], row["feature"]): row for row in out.rows}
        self.assertEqual(rows[("e2", "f_a")]["winning_source"], V3)

    def test_precision_priority_and_provenance(self) -> None:
        cfg = config(conflict_priority="precision_label")
        out = merge_tables(make_tables(), cfg, config_fingerprint(cfg))
        rows = {(row["entity_id"], row["feature"]): row for row in out.rows}
        self.assertEqual(rows[("e2", "f_a")]["winning_source"], V2)
        prov = out.provenance[1]
        self.assertEqual(prov["candidate_sources"], [V1, V2, V3])
        self.assertIn("dependency_key", prov)

    def test_suppressed_list_covers_losers(self) -> None:
        cfg = config()
        out = merge_tables(make_tables(), cfg, config_fingerprint(cfg))
        suppressed_pairs = {
            (item["entity_id"], item["suppressed_source"])
            for item in out.suppressed
        }
        self.assertIn(("e2", V2), suppressed_pairs)
        self.assertIn(("e2", V3), suppressed_pairs)


class ConcurrencyTests(unittest.TestCase):
    def test_parallel_output_matches_single_thread_order(self) -> None:
        entities = [f"e{index:03d}" for index in range(20)]
        tasks = [
            (entity, "scale", lambda entity=entity: entity.upper())
            for entity in entities
        ]
        parallel_results, parallel_failures = run_entity_tasks(tasks, 4)
        serial_results, serial_failures = run_entity_tasks(tasks, 1)
        self.assertEqual(parallel_results, serial_results)
        self.assertEqual(parallel_results, [entity.upper() for entity in entities])
        self.assertEqual(parallel_failures, serial_failures)

    def test_entity_exception_is_isolated_and_coded(self) -> None:
        def good(entity):
            return entity

        def bad(entity):
            raise PrecisionInvalidError("label out of range", entity_id=entity,
                                        stage="precision_validate")

        tasks = [("e1", "scale", lambda: good("e1")),
                 ("e2", "precision_validate", lambda: bad("e2")),
                 ("e3", "scale", lambda: good("e3"))]
        results, failures = run_entity_tasks(tasks, 3)
        self.assertEqual(results, ["e1", "e3"])
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0].entity_id, "e2")
        self.assertEqual(failures[0].code, E_PRECISION_INVALID)
        self.assertEqual(failures[0].sort_key, ("e2", "precision_validate"))


if __name__ == "__main__":
    unittest.main()
