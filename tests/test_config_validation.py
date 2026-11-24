from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from mmledger.config import load_config
from mmledger.errors import (
    E_LIMIT_INVALID,
    E_MERGE_WORKER_SET_UNKNOWN,
    E_MERGE_WORKER_UNKNOWN,
    E_MODEL_DIMENSION,
    E_REQUEST_ARRIVAL,
    E_REQUEST_SEQUENCE,
    E_REQUEST_WORKER_UNKNOWN,
    E_WORKER_DUPLICATE_ID,
    E_WORKER_PORT_RANGE,
)

from tests.helpers import generated_workspace, read_json, write_json


class ConfigValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = generated_workspace(Path(self.temp.name))
        self.config_path = self.root / "samples" / "config.json"
        self.config = read_json(self.config_path)
        self.requests_path = self.root / "samples" / "requests.json"
        self.requests = read_json(self.requests_path)
        self.rules_path = self.root / "samples" / "merge_rules.json"
        self.rules = read_json(self.rules_path)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _load_with(self, *, config=None, requests=None, rules=None):
        write_json(self.config_path, config if config is not None else self.config)
        write_json(self.requests_path, requests if requests is not None else self.requests)
        write_json(self.rules_path, rules if rules is not None else self.rules)
        return load_config(self.config_path)

    def test_valid_config_loads(self) -> None:
        result = self._load_with()
        self.assertIsNotNone(result.bundle)
        self.assertEqual(result.failures, ())

    def test_duplicate_worker_id(self) -> None:
        config = copy.deepcopy(self.config)
        config["workers"].append(copy.deepcopy(config["workers"][0]))
        codes = {item.code for item in self._load_with(config=config).failures}
        self.assertIn(E_WORKER_DUPLICATE_ID, codes)

    def test_port_out_of_range(self) -> None:
        config = copy.deepcopy(self.config)
        config["workers"][0]["port"] = 80
        codes = {item.code for item in self._load_with(config=config).failures}
        self.assertIn(E_WORKER_PORT_RANGE, codes)

    def test_model_dimension_must_be_positive_integer(self) -> None:
        config = copy.deepcopy(self.config)
        config["model"]["input_dim"] = 0
        codes = {item.code for item in self._load_with(config=config).failures}
        self.assertIn(E_MODEL_DIMENSION, codes)

    def test_batch_size_must_be_positive(self) -> None:
        config = copy.deepcopy(self.config)
        config["limits"]["batch_size"] = 0
        codes = {item.code for item in self._load_with(config=config).failures}
        self.assertIn(E_LIMIT_INVALID, codes)

    def test_request_sequence_must_strictly_increase(self) -> None:
        requests = copy.deepcopy(self.requests)
        requests["requests"][1]["sequence"] = requests["requests"][0]["sequence"]
        codes = {item.code for item in self._load_with(requests=requests).failures}
        self.assertIn(E_REQUEST_SEQUENCE, codes)

    def test_arrival_must_be_non_negative(self) -> None:
        requests = copy.deepcopy(self.requests)
        requests["requests"][0]["arrival_ms"] = -1
        codes = {item.code for item in self._load_with(requests=requests).failures}
        self.assertIn(E_REQUEST_ARRIVAL, codes)

    def test_request_worker_must_exist(self) -> None:
        requests = copy.deepcopy(self.requests)
        requests["requests"][0]["worker_id"] = "w-missing"
        codes = {item.code for item in self._load_with(requests=requests).failures}
        self.assertIn(E_REQUEST_WORKER_UNKNOWN, codes)

    def test_merge_rule_references_must_exist(self) -> None:
        rules = copy.deepcopy(self.rules)
        rules["worker_sets"]["active"].append("w-missing")
        codes = {item.code for item in self._load_with(rules=rules).failures}
        self.assertIn(E_MERGE_WORKER_UNKNOWN, codes)

    def test_active_worker_set_must_exist(self) -> None:
        rules = copy.deepcopy(self.rules)
        rules["active_worker_set"] = "missing-set"
        codes = {item.code for item in self._load_with(rules=rules).failures}
        self.assertIn(E_MERGE_WORKER_SET_UNKNOWN, codes)


if __name__ == "__main__":
    unittest.main()
