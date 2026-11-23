from __future__ import annotations

import unittest

from config_diag.canonical import canonical_json
from config_diag.clock import ManualClock
from config_diag.compute import record_bytes
from config_diag.engine import ConfigEngine
from config_diag.structured_log import NullLogger
from config_diag.validation import load_bundle

from .support import SAMPLES, local_temp


class IncrementalTests(unittest.TestCase):
    def test_invalidation_is_topological_and_unaffected_bytes_stay_identical(self) -> None:
        spec, clients, changes = load_bundle(SAMPLES)
        with local_temp("incremental-") as tmp:
            engine = ConfigEngine(tmp, logger=NullLogger(), clock=ManualClock(1_000_000_000))
            try:
                engine.initialize(spec, clients, now_ns=1_000_000_000)
                before = engine.ledger.records()
                expected_affected = [
                    "pricing.base_fee",
                    "pricing.net_fee",
                    "tax.amount",
                    "checkout.total",
                    "reporting.rounded",
                ]
                unaffected = sorted(set(before) - set(expected_affected))
                unaffected_bytes = {key: record_bytes(before[key]) for key in unaffected}
                unaffected_versions = {key: before[key]["version"] for key in unaffected}

                result = engine.apply_operation(changes.operations[0], now_ns=1_100_000_000)
                after = engine.ledger.records()

                self.assertEqual(result["affected"], expected_affected)
                self.assertEqual(
                    [item["edge"] for item in result["propagation"]],
                    [
                        None,
                        "pricing.base_fee->pricing.net_fee",
                        "pricing.net_fee->tax.amount",
                        "pricing.net_fee->checkout.total",
                        "checkout.total->reporting.rounded",
                    ],
                )
                for key in unaffected:
                    self.assertEqual(unaffected_bytes[key], record_bytes(after[key]), key)
                    self.assertEqual(unaffected_versions[key], after[key]["version"], key)
                self.assertEqual(after["pricing.net_fee"]["value"], 1250)
                self.assertEqual(after["tax.amount"]["value"], 100)
                self.assertEqual(after["checkout.total"]["value"], 1350)
                self.assertEqual(after["reporting.rounded"]["value"], 135)
                self.assertEqual(after["pricing.base_fee"]["version"], 2)
                self.assertEqual(after["pricing.net_fee"]["version"], 2)
            finally:
                engine.close()

    def test_group_edit_propagates_two_subtrees_and_formula_bump(self) -> None:
        spec, clients, changes = load_bundle(SAMPLES)
        with local_temp("incremental-group-") as tmp:
            engine = ConfigEngine(tmp, logger=NullLogger(), clock=ManualClock(1_000_000_000))
            try:
                engine.initialize(spec, clients, now_ns=1_000_000_000)
                engine.apply_operation(changes.operations[0], now_ns=1_100_000_000)
                result = engine.apply_operation(changes.operations[1], now_ns=1_200_000_000)
                self.assertEqual(
                    result["affected"],
                    [
                        "ops.batch_size",
                        "cache.shard_span",
                        "ops.batch_value",
                        "tax.rate_permille",
                        "tax.amount",
                        "checkout.total",
                        "reporting.rounded",
                    ],
                )
                propagation = {item["key"]: item for item in result["propagation"]}
                self.assertEqual(propagation["cache.shard_span"]["edge"], "ops.batch_size->cache.shard_span")
                self.assertEqual(propagation["tax.amount"]["edge"], "tax.rate_permille->tax.amount")
                after = engine.ledger.records()
                self.assertEqual(after["tax.amount"]["value"], 125)
                self.assertEqual(after["checkout.total"]["value"], 1375)
                self.assertEqual(after["ops.batch_value"]["value"], 90)
                self.assertEqual(after["cache.shard_span"]["value"], 120)
                self.assertEqual(after["cache.shard_span"]["version"], 2)

                formula_result = engine.apply_operation(changes.operations[2], now_ns=1_300_000_000)
                self.assertEqual(formula_result["affected"], ["reporting.rounded"])
                self.assertEqual(formula_result["derivation_version"], 2)
                self.assertEqual(engine.ledger.records()["reporting.rounded"]["value"], 142)
                self.assertEqual(engine.ledger.records()["reporting.rounded"]["version"], 4)
            finally:
                engine.close()

    def test_stable_topology_across_independent_fresh_engines(self) -> None:
        spec, clients, changes = load_bundle(SAMPLES)
        outputs = []
        for index in range(2):
            with local_temp(f"stable-topology-{index}-") as tmp:
                engine = ConfigEngine(tmp, logger=NullLogger(), clock=ManualClock(1_000_000_000))
                try:
                    engine.initialize(spec, clients, now_ns=1_000_000_000)
                    results = [
                        engine.apply_operation(operation, now_ns=1_000_000_000)
                        for operation in changes.operations[:2]
                    ]
                    outputs.append(
                        canonical_json(
                            [
                                {
                                    "affected": result["affected"],
                                    "propagation": result["propagation"],
                                }
                                for result in results
                            ]
                        )
                    )
                finally:
                    engine.close()
        self.assertEqual(outputs[0], outputs[1])


if __name__ == "__main__":
    unittest.main()
