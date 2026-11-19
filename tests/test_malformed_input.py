import tempfile
import unittest
from pathlib import Path

from cache_invalidation.errors import EGraph
from cache_invalidation.loader import load_declarations


ROOT = Path(__file__).resolve().parent.parent


class MalformedInputTest(unittest.TestCase):
    def test_malformed_declaration_json_becomes_e_graph_with_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            bad = work / "namespaces.json"
            bad.write_text(
                '{\n  "namespaces": [\n    {"name": "x",\n}',
                encoding="utf-8")
            good = ROOT / "samples" / "entries.json"
            with self.assertRaises(EGraph) as caught:
                load_declarations(bad, good)
            self.assertEqual(caught.exception.code, "E_GRAPH")
            self.assertEqual(caught.exception.field_path, "<document>")
            self.assertIsNotNone(caught.exception.line)


if __name__ == "__main__":
    unittest.main()
