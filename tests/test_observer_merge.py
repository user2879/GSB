from __future__ import annotations

import time
import unittest

from config_ledger.merge import canonical_json, merge_observations
from config_ledger.observer import ObservationFetcher
from config_ledger.simulator import reserve_free_port
from config_ledger.structured_log import StructuredLogger

from .helpers import (
    log_records,
    make_spec_data,
    parse_test_spec,
    running_servers,
    temp_workspace,
)


def _scenario(delay_ms: int = 35) -> dict:
    return {
        "node-a": {
            "default_response_delay_ms": delay_ms,
            "observations": {
                "app.mode": [{"value": "stable", "observation_seq": 10, "timestamp_ms": 1000}],
                "duplicate.value": [{"value": 42, "observation_seq": 20, "timestamp_ms": 2000}],
                "missing.quorum": [{"value": 1, "observation_seq": 30, "timestamp_ms": 3000}],
                "tie.rule": [{"value": "alpha", "observation_seq": 5, "timestamp_ms": 1000}],
            },
        },
        "node-b": {
            "default_response_delay_ms": delay_ms,
            "duplicate_counts": {"duplicate.value": 2},
            "observations": {
                "app.mode": [{"value": "stable", "observation_seq": 9, "timestamp_ms": 900}],
                "duplicate.value": [{"value": 42, "observation_seq": 21, "timestamp_ms": 2100}],
                "tie.rule": [{"value": "beta", "observation_seq": 9, "timestamp_ms": 1100}],
            },
            "missing_keys": ["missing.quorum"],
        },
        "node-c": {
            "default_response_delay_ms": delay_ms,
            "observations": {
                "app.mode": [{"value": "canary", "observation_seq": 11, "timestamp_ms": 1100}],
                "duplicate.value": [{"value": 42, "observation_seq": 22, "timestamp_ms": 2200}],
                "missing.quorum": [{"value": 1, "observation_seq": 31, "timestamp_ms": 3100}],
            },
            "missing_keys": ["tie.rule"],
        },
    }


KEYS = {
    "app.mode": {"expected": "stable", "quorum_bp": 6000},
    "duplicate.value": {"expected": 42, "quorum_bp": 6667},
    "missing.quorum": {"expected": 1, "quorum_bp": 10000},
    "tie.rule": {"expected": "alpha", "quorum_bp": 6000},
}


class ObserverAndMergeTests(unittest.TestCase):
    def test_double_budget_deterministic_merge_majority_and_tie_break(self) -> None:
        with temp_workspace() as temp, running_servers(_scenario()) as ports:
            spec = parse_test_spec(
                make_spec_data(
                    ports,
                    KEYS,
                    global_concurrency=2,
                    per_node_quota=1,
                    timeout_ms=600,
                )
            )
            logger = StructuredLogger(temp / "observe.jsonl")
            fetcher = ObservationFetcher(logger)
            first_batch = fetcher.fetch_all(spec)
            second_batch = fetcher.fetch_all(spec)
            first = merge_observations(spec, first_batch)
            second = merge_observations(spec, second_batch)

            self.assertEqual(first.deterministic_bytes(), second.deterministic_bytes())
            self.assertEqual(first.divergence_keys, ("app.mode", "tie.rule"))
            self.assertLessEqual(first_batch.peak_global, 2)
            self.assertGreaterEqual(first_batch.peak_global, 2)
            for node_id, peak in first_batch.peak_by_node.items():
                self.assertLessEqual(peak, 1, node_id)
            self.assertTrue(any(outcome.queued for outcome in first_batch.outcomes))
            for outcome in first_batch.outcomes:
                self.assertGreaterEqual(outcome.queue_wait_end_ms, outcome.queue_wait_start_ms)

            decisions = first.decisions_by_key()
            app_mode = decisions["app.mode"]
            self.assertEqual(app_mode["status"], "DIVERGED")
            self.assertEqual(app_mode["candidate"]["value"], "canary")
            self.assertEqual(app_mode["effective_value"], "stable")
            self.assertEqual(app_mode["winner_basis"], "MAJORITY")
            self.assertEqual(app_mode["majority_count"], 2)

            tie = decisions["tie.rule"]
            self.assertTrue(tie["quorum_met"])
            self.assertEqual(tie["winner_basis"], "LATEST_TIE_BREAK")
            self.assertEqual(tie["effective_value"], "beta")
            self.assertEqual(
                tie["attributions"], ["OBSERVATION_MISSING", "VALUE_DIVERGENCE"]
            )

            missing = decisions["missing.quorum"]
            self.assertFalse(missing["quorum_met"])
            self.assertEqual(missing["status"], "INSUFFICIENT_OBSERVATIONS")
            self.assertIsNone(missing["effective_value"])
            self.assertEqual(missing["attributions"], ["OBSERVATION_MISSING"])

            duplicate_row = first.comparisons_by_key_node()[("duplicate.value", "node-b")]
            self.assertEqual(duplicate_row["duplicate_count"], 2)
            self.assertTrue(duplicate_row["adopted"])
            # Canonical JSON is stable and contains no hash-order-dependent keys.
            self.assertIn('"divergence_keys":["app.mode","tie.rule"]', canonical_json(first.deterministic_payload()))

    def test_invalid_json_is_e_observe_missing_in_the_round(self) -> None:
        scenario = {
            "node-a": {
                "observations": {
                    "key": [{"value": 1, "observation_seq": 1, "timestamp_ms": 1}]
                }
            }
        }
        with temp_workspace() as temp, running_servers(
            scenario, invalid_json_nodes={"node-a"}
        ) as ports:
            spec = parse_test_spec(
                make_spec_data(ports, {"key": {"expected": 1, "quorum_bp": 10000}})
            )
            log_path = temp / "invalid.jsonl"
            batch = ObservationFetcher(StructuredLogger(log_path)).fetch_all(spec)
            self.assertEqual(batch.outcomes[0].status, "INVALID_JSON")
            self.assertEqual(batch.outcomes[0].error_reason, "invalid_json_payload")
            events = log_records(log_path)
            self.assertTrue(any(event["code"] == "E_OBSERVE" for event in events))

    def test_real_socket_timeout_is_not_a_synthetic_delay(self) -> None:
        scenario = {
            "node-a": {
                "response_delays_ms": {"slow.key": 400},
                "observations": {
                    "slow.key": [{"value": 1, "observation_seq": 1, "timestamp_ms": 1}]
                },
            }
        }
        with temp_workspace() as temp, running_servers(scenario) as ports:
            spec = parse_test_spec(
                make_spec_data(
                    ports,
                    {"slow.key": {"expected": 1, "quorum_bp": 10000}},
                    timeout_ms=60,
                )
            )
            log_path = temp / "timeout.jsonl"
            started = time.perf_counter()
            batch = ObservationFetcher(StructuredLogger(log_path)).fetch_all(spec)
            elapsed = time.perf_counter() - started
            self.assertLess(elapsed, 1.5)
            outcome = batch.outcomes[0]
            self.assertEqual(outcome.status, "TIMEOUT")
            self.assertIn("socket_timeout_60ms", outcome.error_reason)
            events = log_records(log_path)
            self.assertTrue(
                any(
                    event["code"] == "E_OBSERVE" and event.get("status") == "TIMEOUT"
                    for event in events
                )
            )

    def test_unreachable_port_logs_e_port_and_e_observe(self) -> None:
        with temp_workspace() as temp:
            closed_port = reserve_free_port()
            spec = parse_test_spec(
                make_spec_data(
                    {"node-a": closed_port},
                    {"key": {"expected": 1, "quorum_bp": 10000}},
                    timeout_ms=100,
                )
            )
            log_path = temp / "port.jsonl"
            batch = ObservationFetcher(StructuredLogger(log_path)).fetch_all(spec)
            outcome = batch.outcomes[0]
            self.assertEqual(outcome.status, "CONNECTION_FAILED")
            self.assertEqual(outcome.diagnostic_code, "E_PORT")
            records = log_records(log_path)
            codes = {record["code"] for record in records}
            self.assertIn("E_OBSERVE", codes)
            self.assertIn("E_PORT", codes)


if __name__ == "__main__":
    unittest.main()
