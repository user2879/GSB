import copy
import json
import os
import tempfile
import unittest

from snapshot_migration.chain import load_chain, parse_chain
from snapshot_migration.errors import ChainError

SAMPLE = os.path.join(os.path.dirname(__file__), "..", "samples",
                      "snapshot_chain.json")


class ChainValidationTests(unittest.TestCase):
    def setUp(self):
        with open(SAMPLE, encoding="utf-8") as handle:
            self.raw = json.load(handle)

    def test_valid_sample_loads(self):
        chain = parse_chain(copy.deepcopy(self.raw))
        self.assertEqual([g.generation_id for g in chain.generations],
                         ["G0", "G1", "G2", "G3"])

    def _expect_chain_error(self, mutate):
        raw = copy.deepcopy(self.raw)
        mutate(raw)
        with self.assertRaises(ChainError) as ctx:
            parse_chain(raw)
        self.assertEqual(ctx.exception.code, "E_CHAIN")

    def test_duplicate_generation_id(self):
        self._expect_chain_error(
            lambda raw: raw["generations"][1].__setitem__(
                "generation_id", "G0"))

    def test_negative_seq_rejected(self):
        self._expect_chain_error(
            lambda raw: raw["generations"][1].__setitem__("seq", -1))

    def test_missing_parent_rejected(self):
        self._expect_chain_error(
            lambda raw: raw["generations"][1].__setitem__(
                "parent_id", "NOPE"))

    def test_cycle_rejected(self):
        def mutate(raw):
            raw["generations"][0]["parent_id"] = "G2"
        self._expect_chain_error(mutate)

    def test_limits_must_have_lower_below_upper(self):
        self._expect_chain_error(
            lambda raw: raw["limits"].__setitem__("lower", 9))

    def test_non_contiguous_ranges_rejected(self):
        def mutate(raw):
            raw["generations"][0]["ranges"][1]["lower"] = "k0150"
        self._expect_chain_error(mutate)

    def test_unsorted_keys_rejected(self):
        def mutate(raw):
            keys = raw["generations"][0]["ranges"][0]["keys"]
            keys[0], keys[1] = keys[1], keys[0]
        self._expect_chain_error(mutate)

    def test_duplicate_key_rejected(self):
        def mutate(raw):
            keys = raw["generations"][0]["ranges"][0]["keys"]
            keys[1] = keys[0]
        self._expect_chain_error(mutate)

    def test_rule_references_unknown_range(self):
        def mutate(raw):
            raw["generations"][1]["rules"][0]["range_id"] = "RX"
        self._expect_chain_error(mutate)

    def test_discard_with_keys_rejected(self):
        def mutate(raw):
            r2 = raw["generations"][1]["ranges"][1]
            r2["keys"] = ["k0100"]
        self._expect_chain_error(mutate)

    def test_invalid_action_rejected(self):
        self._expect_chain_error(
            lambda raw: raw["generations"][1]["rules"][0].__setitem__(
                "action", "explode"))

    def test_load_missing_file(self):
        with self.assertRaises(ChainError):
            load_chain(os.path.join(tempfile.gettempdir(),
                                    "definitely_missing_chain.json"))


if __name__ == "__main__":
    unittest.main()
