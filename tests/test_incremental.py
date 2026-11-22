from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from search_intent_workbench.loader import load_commands, load_dataset
from search_intent_workbench.pipeline import Workbench, semantic_projection
from search_intent_workbench.state import WorkbenchState, apply_command


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "base"


class IncrementalAndConcurrencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        dataset = load_dataset(
            BASE / "queries.jsonl",
            BASE / "intent_config.json",
            BASE / "normalization_rules.json",
        )
        commands, _ = load_commands(BASE / "commands.json")
        cls.dataset = dataset
        cls.commands = commands
        cls.initial_state = WorkbenchState(
            queries=dataset.queries,
            config=dataset.config,
            rules=dataset.rules,
            failures=list(dataset.failures),
        )
        cls.initial_run = Workbench(
            dataset.config, dataset.rules, max_workers=4
        ).run(cls.initial_state.sorted_queries(), input_failures=cls.initial_state.failures)

    def test_dictionary_replacement_invalidates_only_matching_queries(self) -> None:
        state = self.initial_state.clone()
        command = next(
            item for item in self.commands if item["command_id"] == "cmd005"
        )
        state, applied, _ = apply_command(state, command)
        self.assertTrue(applied)
        cache = Workbench.stage_cache_from_run(self.initial_run)
        run = Workbench(
            state.config,
            state.rules,
            max_workers=4,
            previous_stage_cache=cache,
        ).run(state.sorted_queries(), input_failures=state.failures)
        self.assertEqual(run["reuse_report"]["normalization"]["recomputed"], [])
        self.assertEqual(
            set(run["reuse_report"]["slots"]["recomputed"]),
            {"q001", "q009", "q016"},
        )
        self.assertEqual(
            set(run["reuse_report"]["intent"]["recomputed"]),
            {"q001", "q009", "q016"},
        )
        self.assertEqual(
            set(run["reuse_report"]["rewrite"]["recomputed"]),
            {"q001", "q009", "q016"},
        )
        self.assertIn("q002", run["reuse_report"]["slots"]["reused"])

    def test_weight_change_reuses_normalization_and_slots(self) -> None:
        state = self.initial_state.clone()
        command = next(
            item for item in self.commands if item["command_id"] == "cmd004"
        )
        state, applied, _ = apply_command(state, command)
        self.assertTrue(applied)
        cache = Workbench.stage_cache_from_run(self.initial_run)
        run = Workbench(
            state.config,
            state.rules,
            max_workers=4,
            previous_stage_cache=cache,
        ).run(state.sorted_queries(), input_failures=state.failures)
        self.assertEqual(run["reuse_report"]["normalization"]["recomputed"], [])
        self.assertEqual(run["reuse_report"]["slots"]["recomputed"], [])
        self.assertGreater(len(run["reuse_report"]["intent"]["recomputed"]), 0)
        self.assertEqual(
            run["reuse_report"]["intent"]["recomputed"],
            run["reuse_report"]["rewrite"]["recomputed"],
        )
        self.assertEqual(
            set(run["reuse_report"]["intent"]["reused"]), {"q009", "q016"}
        )

    def test_case_fold_change_only_touches_case_bearing_queries(self) -> None:
        state = self.initial_state.clone()
        command = next(
            item for item in self.commands if item["command_id"] == "cmd003"
        )
        state, applied, _ = apply_command(state, command)
        self.assertTrue(applied)
        cache = Workbench.stage_cache_from_run(self.initial_run)
        run = Workbench(
            state.config,
            state.rules,
            max_workers=4,
            previous_stage_cache=cache,
        ).run(state.sorted_queries(), input_failures=state.failures)
        self.assertEqual(
            set(run["reuse_report"]["normalization"]["recomputed"]),
            {"q001", "q002", "q009", "q010", "q013", "q016"},
        )
        self.assertIn("q004", run["reuse_report"]["normalization"]["reused"])

    def test_every_command_matches_full_rebuild(self) -> None:
        state = self.initial_state.clone()
        workbench = Workbench(state.config, state.rules, max_workers=4)
        incremental = workbench.run(
            state.sorted_queries(),
            input_failures=state.failures,
            global_version=1,
        )
        previous = Workbench.stage_cache_from_run(incremental)
        for step, command in enumerate(self.commands, start=1):
            state, _, _ = apply_command(state, command)
            incremental = Workbench(
                state.config,
                state.rules,
                max_workers=4,
                previous_stage_cache=previous,
            ).run(
                state.sorted_queries(),
                input_failures=state.failures,
                global_version=step + 1,
                command_index=step,
            )
            full = Workbench(state.config, state.rules, max_workers=4).run(
                state.sorted_queries(),
                input_failures=state.failures,
                global_version=step + 1,
                command_index=step,
            )
            self.assertEqual(
                semantic_projection(incremental),
                semantic_projection(full),
                msg=f"mismatch after {command['command_id']}",
            )
            for key in semantic_projection(full)["queries"]:
                self.assertTrue(key["query_id"])
            previous = Workbench.stage_cache_from_run(incremental)

    def test_serial_concurrent_and_capped_runs_are_identical(self) -> None:
        queries = self.initial_state.sorted_queries()
        failures = self.initial_state.failures
        serial = Workbench(self.dataset.config, self.dataset.rules, max_workers=1).run(
            queries, input_failures=failures
        )
        concurrent_wb = Workbench(self.dataset.config, self.dataset.rules, max_workers=4)
        concurrent = concurrent_wb.run(queries, input_failures=failures)
        capped_wb = Workbench(self.dataset.config, self.dataset.rules, max_workers=99)
        capped = capped_wb.run(queries, input_failures=failures)
        self.assertEqual(semantic_projection(serial), semantic_projection(concurrent))
        self.assertEqual(semantic_projection(serial), semantic_projection(capped))
        self.assertEqual(capped_wb.max_workers, 4)
        self.assertLessEqual(concurrent["metadata"]["peak_active_tasks"], 4)
        self.assertGreater(concurrent["metadata"]["peak_active_tasks"], 1)
        self.assertEqual(serial["metadata"]["peak_active_tasks"], 1)
        self.assertEqual(
            [item["query_id"] for item in serial["queries"]],
            [item["query_id"] for item in concurrent["queries"]],
        )

    def test_single_query_exception_is_isolated(self) -> None:
        original = Workbench._run_query

        def flaky(self, query):  # type: ignore[no-untyped-def]
            if query["query_id"] == "q005":
                raise RuntimeError("injected query failure")
            return original(self, query)

        with patch.object(Workbench, "_run_query", new=flaky):
            run = Workbench(self.dataset.config, self.dataset.rules, max_workers=4).run(
                self.dataset.queries
            )
        self.assertEqual(len(run["queries"]), 15)
        self.assertEqual(len(run["failures"]), 1)
        self.assertEqual(run["failures"][0]["code"], "E_QUERY_EXEC")
        self.assertEqual(run["failures"][0]["query_id"], "q005")
        self.assertNotIn("q005", {item["query_id"] for item in run["queries"]})

    def test_failed_command_is_atomic_and_recorded(self) -> None:
        state = self.initial_state.clone()
        before_queries = state.sorted_queries()
        before_config = dict(state.config)
        state, applied, change = apply_command(
            state,
            {
                "command_id": "bad-weight",
                "type": "update_intent_weights",
                "weights": {"product_search": 0},
            },
        )
        self.assertFalse(applied)
        self.assertEqual(change["reason"], "weight_invalid")
        self.assertEqual(state.sorted_queries(), before_queries)
        self.assertEqual(state.config, before_config)
        self.assertEqual(len(state.failures), 1)
        self.assertEqual(state.failures[0]["command_id"], "bad-weight")
        self.assertEqual(state.failures[0]["code"], "E_CONFIG_INVALID")

    def test_stage_sort_keys_are_stable(self) -> None:
        expected = [
            f"{query_id}::{stage}"
            for query_id in sorted(query["query_id"] for query in self.dataset.queries)
            for stage in ("normalization", "slots", "intent", "rewrite")
        ]
        self.assertEqual(self.initial_run["metadata"]["task_schedule"], expected)
        for record in self.initial_run["queries"]:
            for stage in ("normalization", "slots", "intent", "rewrite"):
                self.assertEqual(
                    record["stages"][stage]["sort_key"],
                    f"{record['query_id']}::{stage}",
                )


if __name__ == "__main__":
    unittest.main()
