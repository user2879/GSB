import unittest
from pathlib import Path
import tempfile

from fxpipe.config import QuantSpec
from fxpipe.errors import Code, IssueLog
from fxpipe.exchange import (
    E1_LINE_LEN,
    QuantRecord,
    e1_to_e2,
    e2_to_e1,
    encode_e1_line,
    parse_e1_file,
    parse_e1_line,
    parse_e2_file,
    read_bypass,
    restore_e2_from_bypass,
    write_bypass,
    write_e1,
    write_e2,
)


def e1_record(chain_id=1, seq=2, permille=1250, width=16):
    return QuantRecord(
        chain_id=chain_id, seq=seq,
        quant=QuantSpec(permille=permille, width=width, scheme=f"qint{width}",
                        precision_label="e1-exact", completeness="complete"),
        source_version="e1",
    )


class ExchangeTests(unittest.TestCase):
    def test_e1_round_trip_fixed_width(self):
        record = e1_record()
        line = encode_e1_line(record)
        self.assertEqual(len(line), E1_LINE_LEN)
        log = IssueLog()
        parsed = parse_e1_line(line, 1, log)
        self.assertEqual(log.issues, [])
        self.assertEqual(parsed.quant.permille, 1250)
        self.assertEqual(parsed.quant.width, 16)
        self.assertEqual(parsed.source_version, "e1")

    def test_e1_truncated_and_illegal_chars(self):
        log = IssueLog()
        self.assertIsNone(parse_e1_line("00000100000000000100000000032", 1, log))
        self.assertEqual(log.issues[0].code, Code.E_FIELD_TRUNCATED)
        log = IssueLog()
        bad = "000001" + "0000" + "0000abc000" + "00000016"
        self.assertEqual(len(bad), E1_LINE_LEN)
        self.assertIsNone(parse_e1_line(bad, 2, log))
        self.assertTrue(any(issue.code == Code.E_FIELD_TRUNCATED for issue in log.issues))

    def test_e1_e2_equivalent_internal_representation(self):
        e1 = e1_record()
        e2 = e1_to_e2(e1)
        self.assertEqual(e2.source_version, "e2")
        self.assertAlmostEqual(e2.quant.permille / 1000.0, 1.25)
        self.assertEqual(e2.quant.scheme, "qint16")
        self.assertTrue(e1.quant.equivalent(e2.quant))

    def test_downgrade_bypass_restore_residual_and_label(self):
        original = QuantRecord(
            chain_id=3, seq=1,
            quant=QuantSpec(permille=1234, width=32, scheme="qint32",
                            precision_label="lab-origin", completeness="complete",
                            residual=0.5),
            source_version="e2",
        )
        downgraded, bypass = e2_to_e1(original)
        self.assertEqual(downgraded.source_version, "e1")
        self.assertEqual(bypass["precision_label"], "lab-origin")
        self.assertEqual(bypass["overwritten_refs"], ["c3op1"])
        restored = restore_e2_from_bypass(downgraded, bypass)
        self.assertEqual(restored.quant.precision_label, "lab-origin")
        self.assertEqual(restored.quant.residual, 0.5)
        self.assertEqual(restored.quant.permille, 1234)

    def test_file_round_trip_and_bypass_map(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            records = [e1_record(1, 0, 1000, 32), e1_record(1, 1, 875, 16)]
            e2_records = [e1_to_e2(r) for r in records]
            bypass = {}
            downgraded = []
            for record in e2_records:
                e1r, meta = e2_to_e1(record)
                bypass[record.node_id()] = meta
                downgraded.append(e1r)
            write_e1(tmp / "c.e1", downgraded)
            write_e2(tmp / "c.e2", e2_records)
            write_bypass(tmp / "bypass.json", bypass)

            log = IssueLog()
            e1_loaded = parse_e1_file(tmp / "c.e1", log)
            e2_loaded = parse_e2_file(tmp / "c.e2", log, read_bypass(tmp / "bypass.json"))
            self.assertEqual(log.issues, [])
            for a, b in zip(e1_loaded, e2_loaded):
                self.assertTrue(a.quant.equivalent(b.quant))
                self.assertEqual(a.node_id(), b.node_id())

    def test_missing_field_marker_and_completeness(self):
        line = "000001" + "0001" + "NA".rjust(10) + "00000016"
        log = IssueLog()
        record = parse_e1_line(line, 1, log)
        self.assertIn("permille", record.quant.missing)
        self.assertTrue(record.quant.completeness.startswith("missing:"))


if __name__ == "__main__":
    unittest.main()
