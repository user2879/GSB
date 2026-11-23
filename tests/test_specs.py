from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from gray_merge_lab.errors import LabError
from gray_merge_lab.specs import (
    load_spec,
    validate_backends,
    validate_rules,
    validate_samples,
)


def backend(**changes):
    value = {
        "backend_id": "b1",
        "host": "127.0.0.1",
        "port": 20001,
        "model_dim": 4,
        "hidden_dim": 8,
        "seed": 7,
        "quant_profile": "fp32-v1",
    }
    value.update(changes)
    return value


def rule(**changes):
    value = {
        "rule_id": "r1",
        "request_prefix": "/api",
        "stable_backend": "b1",
        "canary_backend": "b2",
        "canary_weight_bp": 1000,
        "batch_size": 2,
    }
    value.update(changes)
    return value


def sample(**changes):
    value = {
        "sample_id": "s1",
        "sample_seq": 1,
        "client_key": "client",
        "rule_id": "r1",
        "request_path": "/api/1",
        "input_seed": 99,
    }
    value.update(changes)
    return value


class SpecValidationTests(unittest.TestCase):
    def assertSpecError(self, code, field, operation):
        with self.assertRaises(LabError) as caught:
            operation()
        self.assertEqual(caught.exception.code, code)
        self.assertEqual(caught.exception.field, field)

    def test_valid_backends_and_rules(self):
        backends = validate_backends(
            [backend(), backend(backend_id="b2", port=20002)]
        )
        rules = validate_rules([rule()], backends)
        self.assertEqual(len(backends), 2)
        self.assertEqual(rules[0].batch_size, 2)

    def test_first_backend_violation_is_host(self):
        self.assertSpecError(
            "E_SPEC",
            "backends[0].host",
            lambda: validate_backends([backend(host="localhost")]),
        )

    def test_duplicate_backend_id(self):
        self.assertSpecError(
            "E_SPEC",
            "backends[1].backend_id",
            lambda: validate_backends([backend(), backend(port=20002)]),
        )

    def test_duplicate_port(self):
        self.assertSpecError(
            "E_SPEC",
            "backends[1].port",
            lambda: validate_backends([backend(), backend(backend_id="b2")]),
        )

    def test_positive_integer_dimensions_reject_bool_and_zero(self):
        self.assertSpecError(
            "E_SPEC",
            "backends[0].model_dim",
            lambda: validate_backends([backend(model_dim=True)]),
        )
        self.assertSpecError(
            "E_SPEC",
            "backends[0].hidden_dim",
            lambda: validate_backends([backend(hidden_dim=0)]),
        )

    def test_quantization_profile_supported(self):
        self.assertSpecError(
            "E_SPEC",
            "backends[0].quant_profile",
            lambda: validate_backends([backend(quant_profile="int3-magic")]),
        )

    def test_unknown_backend_field_rejected(self):
        self.assertSpecError(
            "E_SPEC",
            "backends[0].unexpected",
            lambda: validate_backends([backend(unexpected=1)]),
        )

    def test_rule_weight_range_and_duplicate_id(self):
        backends = validate_backends([backend(), backend(backend_id="b2", port=20002)])
        self.assertSpecError(
            "E_SPEC",
            "rules[0].canary_weight_bp",
            lambda: validate_rules([rule(canary_weight_bp=10001)], backends),
        )
        self.assertSpecError(
            "E_SPEC",
            "rules[1].rule_id",
            lambda: validate_rules([rule(), rule()], backends),
        )

    def test_rule_backend_must_exist_and_differ(self):
        backends = validate_backends([backend(), backend(backend_id="b2", port=20002)])
        self.assertSpecError(
            "E_SPEC",
            "rules[0].stable_backend",
            lambda: validate_rules([rule(stable_backend="missing")], backends),
        )
        self.assertSpecError(
            "E_SPEC",
            "rules[0].canary_backend",
            lambda: validate_rules([rule(canary_backend="b1")], backends),
        )

    def test_sample_prefix_duplicate_id_and_sequence(self):
        backends = validate_backends([backend(), backend(backend_id="b2", port=20002)])
        rules = validate_rules([rule()], backends)
        self.assertSpecError(
            "E_SPEC",
            "samples[0].request_path",
            lambda: validate_samples([sample(request_path="/other")], rules),
        )
        self.assertSpecError(
            "E_SPEC",
            "samples[1].sample_id",
            lambda: validate_samples([sample(), sample(sample_seq=2)], rules),
        )
        self.assertSpecError(
            "E_SPEC",
            "samples[1].sample_seq",
            lambda: validate_samples([sample(), sample(sample_id="s2")], rules),
        )

    def test_load_spec_from_files_and_invalid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "backends.json").write_text(
                json.dumps({"format_version": 1, "backends": [backend(), backend(backend_id="b2", port=20002)]}),
                encoding="utf-8",
            )
            (root / "rules.json").write_text(
                json.dumps({"format_version": 1, "rules": [rule()]}), encoding="utf-8"
            )
            (root / "samples.ndjson").write_text(json.dumps(sample()) + "\n", encoding="utf-8")
            loaded = load_spec(root / "backends.json", root / "rules.json", root / "samples.ndjson")
            self.assertEqual(loaded.backend_version().split(":", 1)[0], "sha256")
            (root / "rules.json").write_text("{bad", encoding="utf-8")
            with self.assertRaises(LabError) as caught:
                load_spec(root / "backends.json", root / "rules.json", root / "samples.ndjson")
            self.assertEqual(caught.exception.code, "E_SPEC")
            self.assertEqual(caught.exception.field, "rules")


if __name__ == "__main__":
    unittest.main()

