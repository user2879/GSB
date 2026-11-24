import json
import os
import tempfile
import unittest

from receipt_ledger.errors import (
    E_CONFIG_INVALID,
    E_DUPLICATE_REQUEST_ID,
    E_REQUEST_INVALID,
    E_WORKER_UNKNOWN,
)
from receipt_ledger.loader import (
    load_commands,
    load_merge_rules,
    load_request_stream,
    load_worker_config,
)


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, name, payload):
        path = os.path.join(self.dir, name)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        return path

    def test_worker_identity_port_and_dimensions(self):
        path = self._write(
            "workers.json",
            {
                "batch_size": 2,
                "workers": [
                    {"worker_id": "w1", "port": 9000, "model_dim": 4, "hidden_dim": 8,
                     "param_version": 1},
                    {"worker_id": "w1", "port": 9001, "model_dim": 4, "hidden_dim": 8,
                     "param_version": 1},
                    {"worker_id": "w2", "port": 80, "model_dim": 4, "hidden_dim": 8,
                     "param_version": 1},
                    {"worker_id": "w3", "port": 9002, "model_dim": 0, "hidden_dim": 8,
                     "param_version": 1},
                ],
            },
        )
        config, failures = load_worker_config(path)
        codes = [item.code for item in failures]
        self.assertEqual(codes.count(E_CONFIG_INVALID), 3)
        self.assertEqual([worker.worker_id for worker in config["workers"]], ["w1"])

    def test_request_stream_ordering_arrival_and_duplicates(self):
        path = os.path.join(self.dir, "requests.jsonl")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write('{"seq":1,"request_id":"q1","arrival_ms":0,"token_count":3}\n')
            handle.write('{"seq":1,"request_id":"q2","arrival_ms":1,"token_count":3}\n')
            handle.write('{"seq":2,"request_id":"q1","arrival_ms":-1,"token_count":3}\n')
            handle.write('{"seq":3,"request_id":"q3","arrival_ms":2,"token_count":3}\n')
        requests, failures = load_request_stream(path)
        self.assertEqual([item.request_id for item in requests], ["q1", "q3"])
        self.assertIn(E_REQUEST_INVALID, [item.code for item in failures])
        self.assertIn(E_DUPLICATE_REQUEST_ID, [item.code for item in failures])

    def test_merge_rule_unknown_worker_and_batch_size(self):
        worker_path = self._write(
            "workers.json",
            {"batch_size": 1, "workers": [
                {"worker_id": "w1", "port": 9000, "model_dim": 2, "hidden_dim": 2,
                 "param_version": 0}]},
        )
        config, _ = load_worker_config(worker_path)
        rule_path = self._write(
            "rules.json",
            {"rules": [
                {"rule_id": "r1", "worker_ids": ["w1", "ghost"], "batch_size": 2},
                {"rule_id": "r2", "worker_ids": ["w1"], "batch_size": 0},
            ]},
        )
        rules, failures = load_merge_rules(rule_path, config["workers"])
        self.assertEqual(rules, ())
        self.assertEqual(failures[0].code, E_WORKER_UNKNOWN)
        self.assertEqual(failures[1].code, E_CONFIG_INVALID)

    def test_command_validation(self):
        path = self._write("commands.json", {"commands": [{"type": "nope"}]})
        commands, failures = load_commands(path)
        self.assertEqual(commands, ())
        self.assertEqual(failures[0].code, "E_COMMAND_INVALID")


if __name__ == "__main__":
    unittest.main()
