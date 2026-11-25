import json
import tempfile
import unittest
from pathlib import Path

from fxpipe.config import (
    Chain,
    OpSpec,
    PipelineConfig,
    QuantSpec,
    validate_chain,
    validate_tokens_segments,
)
from fxpipe.errors import Code, IssueLog
from fxpipe.exchange import parse_e1_line, QuantRecord
from fxpipe.loader import load_edits, load_pipeline
from fxpipe.samplegen import generate


def qspec(permille=1000, width=32):
    return QuantSpec(permille=permille, width=width, scheme=f"qint{width}",
                     precision_label="t", completeness="complete")


def op(chain_id, seq, inputs=(), op_type="embed"):
    return OpSpec(chain_id=chain_id, seq=seq, op_type=op_type,
                  inputs=tuple(inputs), params={}, quant=qspec(),
                  source_version="e2")


class ValidationTests(unittest.TestCase):
    def test_config_head_dim_divisibility_and_enums(self):
        log = IssueLog()
        bad = PipelineConfig(heads=3, dim=10, frac_bits=12, gear="gx",
                             point_budget=1, operator_quota=1, max_workers=1, vocab_size=4)
        self.assertFalse(bad.validate(log))
        codes = log.count_by_code()
        self.assertIn("E_SCHEMA", codes)

        good = PipelineConfig(heads=2, dim=8, frac_bits=16, gear="g1",
                              point_budget=10, operator_quota=10, max_workers=2, vocab_size=4)
        self.assertTrue(good.validate(IssueLog()))

    def test_token_segment_coverage_and_strict_increase(self):
        log = IssueLog()
        self.assertTrue(validate_tokens_segments([1, 2, 3], [(0, 2), (2, 3)], 5, log))
        log = IssueLog()
        self.assertFalse(validate_tokens_segments([1, 2, 3], [(0, 1), (1, 2)], 5, log))
        self.assertIn("E_SCHEMA", log.count_by_code())
        log = IssueLog()
        self.assertFalse(validate_tokens_segments([1, 99], [(0, 2)], 5, log))

    def test_chain_contiguous_refs_and_cycle(self):
        log = IssueLog()
        good = Chain(1, [op(1, 0), op(1, 1, (0,), "final")])
        self.assertTrue(validate_chain(1, good.ops, log))

        log = IssueLog()
        missing_ref = Chain(1, [op(1, 0, (7,), "final")])
        self.assertFalse(validate_chain(1, missing_ref.ops, log))
        self.assertIn("E_REF", log.count_by_code())

        log = IssueLog()
        cyclic = Chain(1, [
            op(1, 0, (1,), "layer_norm"),
            op(1, 1, (0,), "final"),
        ])
        self.assertFalse(validate_chain(1, cyclic.ops, log))
        self.assertIn("E_CYCLE", log.count_by_code())

        log = IssueLog()
        gaps = Chain(1, [op(1, 0), op(1, 2, (0,), "final")])
        self.assertFalse(validate_chain(1, gaps.ops, log))

    def test_e1_permille_and_width_range(self):
        log = IssueLog()
        over = ("000001" + "0000" + str(10_000_000).rjust(10, "0") + "00000032")
        record = parse_e1_line(over, 1, log)
        self.assertIsNone(record)
        self.assertIn("E_SCHEMA", log.count_by_code())

        log = IssueLog()
        bad_width = "000001" + "0000" + "0000001000" + "00000064"
        self.assertIsNone(parse_e1_line(bad_width, 1, log))
        self.assertIn("E_SCHEMA", log.count_by_code())

    def test_bad_edit_commands_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "edits.json"
            path.write_text(json.dumps({"edits": [
                {"cmd": "replace_token", "index": 0, "token": 1},
                {"cmd": "explode"},
                "not-an-object",
            ]}))
            log = IssueLog()
            edits = load_edits(path, log)
            self.assertEqual(len(edits), 1)
            self.assertIn("E_SCHEMA", log.count_by_code())

    def test_full_pipeline_loader_skips_invalid_chain_but_keeps_valid(self):
        paths = generate(Path(tempfile.mkdtemp()))
        log = IssueLog()
        model = load_pipeline(
            paths["config"], paths["tokens"], paths["segments"],
            paths["invalid_graph"],
            e2_path=paths["invalid_quant"], bypass_path=paths["bypass"],
        )
        log.extend(model.log)
        self.assertIn("E_REF", log.count_by_code())
        # Invalid chain 9 is rejected rather than raising.
        self.assertNotIn(9, model.chains)


if __name__ == "__main__":
    unittest.main()
