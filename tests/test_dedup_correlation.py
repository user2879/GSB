from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from alert_pipeline.correlate import build_ledger, derive_candidates
from alert_pipeline.dedup import deduplicate_serial, process_group
from alert_pipeline.errors import PipelineError
from alert_pipeline.models import NormalizedAlert, Provenance
from alert_pipeline.spec import load_root_rules, load_service_graph

from .helpers import write_graph, write_rules


def alert(
    line: int,
    timestamp: int,
    service: str = "svc-a",
    fingerprint: str = "fp-a",
    window: int = 1000,
    level: str = "critical",
    strategy: str = "s1",
) -> NormalizedAlert:
    provenance = {
        name: Provenance("record", f"record.{name}")
        for name in (
            "timestamp_ms",
            "service_id",
            "level",
            "fingerprint",
            "silence_window_ms",
            "root_candidates",
            "strategy_id",
            "tags",
        )
    }
    return NormalizedAlert(
        line_number=line,
        timestamp_ms=timestamp,
        service_id=service,
        level=level,
        fingerprint=fingerprint,
        tags={},
        silence_window_ms=window,
        root_candidate_refs=tuple(),
        strategy_id=strategy,
        format_version=3,
        declared_version=1,
        provenance=provenance,
        raw_digest=f"raw-{line}",
    )


class DedupTests(unittest.TestCase):
    def test_window_boundary_and_participant_metadata(self) -> None:
        alerts = [
            alert(1, 100),
            alert(2, 500),
            alert(3, 1100),
            alert(4, 2099),
            alert(5, 2100),
        ]
        aggregates = deduplicate_serial(alerts)
        self.assertEqual(len(aggregates), 3)
        self.assertEqual(aggregates[0].first_timestamp_ms, 100)
        self.assertEqual(aggregates[0].last_timestamp_ms, 500)
        self.assertEqual(aggregates[0].suppressed_count, 1)
        self.assertEqual(aggregates[0].participant_lines, (1, 2))
        self.assertEqual(aggregates[1].first_timestamp_ms, 1100)
        self.assertEqual(aggregates[1].last_timestamp_ms, 2099)
        self.assertEqual(aggregates[1].suppressed_count, 1)
        self.assertEqual(aggregates[2].first_timestamp_ms, 2100)
        self.assertEqual(aggregates[2].suppressed_count, 0)
        self.assertEqual(aggregates[0].window_version, "v3:1000:s1@decl1")

    def test_groups_are_scoped_by_service_and_fingerprint(self) -> None:
        aggregates = deduplicate_serial(
            [
                alert(1, 100, service="a", fingerprint="f"),
                alert(2, 100, service="b", fingerprint="f"),
                alert(3, 100, service="a", fingerprint="g"),
            ]
        )
        self.assertEqual(len(aggregates), 3)

    def test_arrival_order_does_not_change_conclusion(self) -> None:
        alerts = [alert(1, 100), alert(2, 500), alert(3, 1100), alert(4, 1500)]
        forward = deduplicate_serial(alerts)
        backward = deduplicate_serial(list(reversed(alerts)))
        self.assertEqual(
            [item.to_public_dict() for item in forward],
            [item.to_public_dict() for item in backward],
        )

    def test_state_carries_across_batches(self) -> None:
        batch_one = process_group([alert(1, 100), alert(2, 800)])
        batch_two = process_group([alert(3, 1500)], batch_one)
        self.assertEqual(len(batch_two), 2)
        self.assertEqual(batch_two[0].suppressed_count, 1)
        self.assertEqual(batch_two[0].participant_lines, (1, 2))
        self.assertEqual(batch_two[1].first_timestamp_ms, 1500)


class CorrelationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.graph_path = self.root / "graph.json"
        write_graph(
            self.graph_path,
            ["alert", "caller", "dep1", "dep2", "base"],
            [
                ("alert", "dep1"),
                ("dep1", "base"),
                ("alert", "dep2"),
                ("caller", "alert"),
            ],
            version=4,
        )
        self.rules_path = self.root / "rules.json"
        write_rules(
            self.rules_path,
            version=7,
            rules=[
                {
                    "rule_id": "alert-rule",
                    "services": ["alert"],
                    "directions": ["upstream", "downstream"],
                    "max_depth": 2,
                    "include_self": True,
                }
            ],
        )
        self.graph = load_service_graph(self.graph_path)
        self.rules = load_root_rules(self.rules_path, self.graph)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_upstream_downstream_paths_and_stable_order(self) -> None:
        candidates, rule_id = derive_candidates("alert", self.graph, self.rules)
        self.assertEqual(rule_id, "alert-rule")
        services = [candidate.service_id for candidate in candidates]
        self.assertEqual(set(services), {"alert", "caller", "dep1", "dep2", "base"})
        self.assertEqual(services[0], "alert")
        by_service = {candidate.service_id: candidate for candidate in candidates}
        self.assertEqual(by_service["dep1"].path, ("alert", "dep1"))
        self.assertEqual(by_service["base"].path, ("alert", "dep1", "base"))
        self.assertEqual(by_service["caller"].path, ("alert", "caller"))
        self.assertEqual(by_service["dep1"].direction, "upstream")
        self.assertEqual(by_service["caller"].direction, "downstream")
        again, _ = derive_candidates("alert", self.graph, self.rules)
        self.assertEqual(
            [item.to_public_dict() for item in candidates],
            [item.to_public_dict() for item in again],
        )

    def test_reversed_dependency_input_has_same_candidates(self) -> None:
        reversed_path = self.root / "graph-reversed.json"
        write_graph(
            reversed_path,
            ["base", "dep2", "dep1", "caller", "alert"],
            [
                ("caller", "alert"),
                ("alert", "dep2"),
                ("dep1", "base"),
                ("alert", "dep1"),
            ],
            version=4,
        )
        reversed_graph = load_service_graph(reversed_path)
        reversed_rules = load_root_rules(self.rules_path, reversed_graph)
        first, _ = derive_candidates("alert", self.graph, self.rules)
        second, _ = derive_candidates("alert", reversed_graph, reversed_rules)
        self.assertEqual(
            [item.to_public_dict() for item in first],
            [item.to_public_dict() for item in second],
        )

    def test_ledger_contains_declared_refs_and_digest(self) -> None:
        aggregate = deduplicate_serial([alert(1, 100, service="alert", fingerprint="f")])[0]
        entries = build_ledger(
            [aggregate],
            self.graph,
            self.rules,
            {aggregate.alert_id: ["dep1", "base"]},
        )
        self.assertEqual(entries[0].declared_root_refs, ("base", "dep1"))
        self.assertEqual(entries[0].rule_version, 7)
        self.assertEqual(entries[0].graph_version, 4)
        self.assertEqual(len(entries[0].content_digest), 64)

    def test_cycle_rejected(self) -> None:
        cycle_path = self.root / "cycle.json"
        write_graph(cycle_path, ["a", "b"], [("a", "b"), ("b", "a")])
        with self.assertRaises(PipelineError) as caught:
            load_service_graph(cycle_path)
        self.assertEqual(caught.exception.code, "E_SPEC")


if __name__ == "__main__":
    unittest.main()
