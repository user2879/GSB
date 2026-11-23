import json
import os
import struct
import unittest

from hotblock_diag.config import (GEAR_FRAC, load_commands, load_fixedpoint,
                                  load_prefilter, validate_candidates,
                                  validate_layout)
from hotblock_diag.errors import (E_COMMAND_REF, E_CONFIG_INVALID,
                                  E_TENSOR_LAYOUT, FailureList)
from hotblock_diag.tensorpack import read_pack, write_pack
from tests._harness import (blocks_for, layout_for, make_fixedpoint,
                            make_pack, make_prefilter, temp_workspace,
                            write_json)


class TensorPackTests(unittest.TestCase):
    def test_roundtrip_and_fingerprint(self):
        ws = temp_workspace()
        path, tensors = make_pack(ws)
        failures = FailureList()
        pack = read_pack(path, failures)
        self.assertIsNotNone(pack)
        self.assertEqual(pack.order, ["query", "key", "value"])
        self.assertEqual(pack.tensors["query"].shape, (16, 8))
        self.assertEqual(failures.codes(), [])

    def test_truncated_pack_raises_pack_code(self):
        ws = temp_workspace()
        path, _ = make_pack(ws)
        with open(path, "rb") as handle:
            blob = handle.read()
        cut = os.path.join(ws, "cut.bin")
        with open(cut, "wb") as handle:
            handle.write(blob[:-12])
        failures = FailureList()
        self.assertIsNone(read_pack(cut, failures))
        self.assertIn("E_PACK_TRUNCATED", failures.codes())

    def test_corrupted_crc_layout_code(self):
        ws = temp_workspace()
        path, _ = make_pack(ws)
        with open(path, "rb") as handle:
            blob = bytearray(handle.read())
        blob[60] ^= 0xFF
        corrupt = os.path.join(ws, "corrupt.bin")
        with open(corrupt, "wb") as handle:
            handle.write(bytes(blob))
        failures = FailureList()
        self.assertIsNone(read_pack(corrupt, failures))
        self.assertIn(E_TENSOR_LAYOUT, failures.codes())


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.ws = temp_workspace()

    def _write(self, name, obj):
        path = os.path.join(self.ws, name)
        write_json(path, obj)
        return path

    def test_invalid_acc_bits_and_gear(self):
        pre_path = self._write("p.json", dict(q_block=4, k_block=4, candidates=2,
                                               score_floor_ratio=0.5, temperature=1.0))
        fp_path = self._write("f.json", dict(
            frac_bits=8, quant_gear="q8", acc_bits=8, saturation="saturate",
            rounding="nearest_even", hot_capacity=4, hot_keep=3, max_rounds=3,
            threads=2, batch_budget=10, qblock_quota=5, snapshot_interval=1,
            tolerance=0.0,
        ))
        failures = FailureList()
        self.assertIsNone(load_fixedpoint(fp_path, failures))
        self.assertIn(E_CONFIG_INVALID, failures.codes())

    def test_temperature_must_be_positive_finite(self):
        path = self._write("p.json", dict(q_block=4, k_block=4, candidates=2,
                                          score_floor_ratio=0.5, temperature=0))
        failures = FailureList()
        self.assertIsNone(load_prefilter(path, failures))
        self.assertIn(E_CONFIG_INVALID, failures.codes())

    def test_candidates_cannot_exceed_key_blocks(self):
        pre = make_prefilter(candidates=99)
        failures = FailureList()
        _, tensors = make_pack(self.ws)
        layout = layout_for(tensors, pre, failures)
        validate_candidates(pre, layout, failures)
        self.assertIn(E_CONFIG_INVALID, failures.codes())

    def test_padding_reject_and_last_zero(self):
        _, tensors = make_pack(self.ws, seq_q=18, seq_k=17)
        failures = FailureList()
        pre_reject = make_prefilter(padding="reject")
        self.assertIsNone(layout_for(tensors, pre_reject, failures))
        self.assertIn(E_TENSOR_LAYOUT, failures.codes())

        failures = FailureList()
        pre_pad = make_prefilter()
        layout = layout_for(tensors, pre_pad, failures)
        self.assertEqual(layout.padded_q, 20)
        self.assertEqual(layout.padded_k, 20)
        blocks = blocks_for(tensors, layout, pre_pad)
        self.assertTrue((blocks["query"][-1][-2:] == 0).all())

    def test_command_reference_validation(self):
        pre = make_prefilter()
        fp = make_fixedpoint()
        _, tensors = make_pack(self.ws)
        layout = layout_for(tensors, pre)
        bad_commands = [
            {"cmd": "replace_qblock", "index": 99, "data": [0.0] * 32},
            {"cmd": "replace_kblock", "index": 0, "data": [1.0] * 31},
            {"cmd": "set_quant", "quant_gear": "q32"},
            {"cmd": "update_prefilter",
             "params": {"temperature": -1.0}},
        ]
        path = self._write("c.json", bad_commands)
        failures = FailureList()
        commands = load_commands(path, layout, pre, fp, failures)
        self.assertEqual(commands, [])
        self.assertTrue(all(code == E_COMMAND_REF for code in failures.codes()))

    def test_valid_commands_parse(self):
        pre = make_prefilter()
        fp = make_fixedpoint()
        _, tensors = make_pack(self.ws)
        layout = layout_for(tensors, pre)
        good = [
            {"cmd": "replace_qblock", "index": 0, "data": [0.1] * 32},
            {"cmd": "set_quant", "quant_gear": "q12"},
            {"cmd": "set_hot_capacity", "capacity": 6},
        ]
        path = self._write("c.json", good)
        failures = FailureList()
        commands = load_commands(path, layout, pre, fp, failures)
        self.assertEqual(len(commands), 3)
        self.assertEqual(failures.codes(), [])


if __name__ == "__main__":
    unittest.main()
