import json
import shutil
import tempfile
import unittest
from pathlib import Path

from quantcalib.data.samples import generate_workspace
from quantcalib.ledger.ledger import adjudicate
from quantcalib.models import UnifiedReceipt
from quantcalib.runner import run_incremental
from quantcalib.schedule.commands import write_command_stream


class LedgerTests(unittest.TestCase):
    def test_authority_prefers_higher_param_version_then_refs(self):
        receipts = [
            UnifiedReceipt(layer="l", coverage_permille=800,
                           source_version="q1", precision_labels={}),
            UnifiedReceipt(layer="l", coverage_permille=700,
                           source_version="q2", param_version=1, time_ms=5,
                           precision_labels={}),
            UnifiedReceipt(layer="l", coverage_permille=900,
                           source_version="q3", param_version=2, time_ms=6,
                           group_scale_digest="d", duplicate_refs=["x"],
                           weight_snapshot="s", precision_labels={}),
        ]
        entry = adjudicate(receipts)[0]
        self.assertEqual(entry.authority_source, "q3")
        self.assertEqual(entry.param_version, 2)
        self.assertEqual(entry.rationale, "higher param_version")

    def test_reference_count_breaks_param_tie(self):
        receipts = [
            UnifiedReceipt(layer="l", coverage_permille=900,
                           source_version="q3", param_version=2, time_ms=6,
                           group_scale_digest="d", duplicate_refs=[],
                           weight_snapshot="s", precision_labels={}),
            UnifiedReceipt(layer="l", coverage_permille=900,
                           source_version="q3", param_version=2, time_ms=6,
                           group_scale_digest="d2", duplicate_refs=["a", "b"],
                           weight_snapshot="s2", precision_labels={}),
        ]
        entry = adjudicate(receipts)[0]
        self.assertEqual(entry.rationale,
                         "more covered duplicate calibration references")
        self.assertEqual(entry.group_scale_digest, "d2")


class InvalidationTests(unittest.TestCase):
    def test_weight_replace_only_recomputes_affected_layer_groups(self):
        tmp = Path(tempfile.mkdtemp(prefix="qc_inv_"))
        try:
            generate_workspace(tmp)
            commands = [
                {"command": "replace_layer_weights",
                 "payload": {"layer": "layer.fc2", "seed": 4242,
                             "scale": 1.2}},
            ]
            write_command_stream(tmp / "commands.json", commands)
            results = run_incremental(tmp, tmp / "commands.json")
            final = results[-1]
            rows = final["reuse"]["records"]
            fc1 = [r for r in rows if r["layer"] == "layer.fc1"]
            fc2 = [r for r in rows if r["layer"] == "layer.fc2"]
            self.assertTrue(fc1)
            self.assertTrue(all(r["hit"] for r in fc1))
            self.assertTrue(all(not r["hit"] for r in fc2))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_bit_change_only_invalidates_quantization_of_target_layer(self):
        tmp = Path(tempfile.mkdtemp(prefix="qc_bit_"))
        try:
            generate_workspace(tmp)
            commands = [
                {"command": "set_layer_bits",
                 "payload": {"layer": "layer.fc3", "bits": 4}},
            ]
            write_command_stream(tmp / "commands.json", commands)
            results = run_incremental(tmp, tmp / "commands.json")
            final = results[-1]
            rows = final["reuse"]["records"]
            target_quant = [r for r in rows
                            if r["layer"] == "layer.fc3"
                            and r["stage"] == "quantization"]
            target_calib = [r for r in rows
                            if r["layer"] == "layer.fc3"
                            and r["stage"] == "calibration"]
            other = [r for r in rows if r["layer"] != "layer.fc3"]
            self.assertTrue(all(not r["hit"] for r in target_quant))
            self.assertTrue(all(r["hit"] for r in target_calib))
            self.assertTrue(all(r["hit"] for r in other))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
