import copy
import json
import unittest

from arbworkbench.canonical import canonical_json
from arbworkbench.ledger import Ledger
from arbworkbench.session import Session
from arbworkbench.spec import load_json, parse_specification
import tempfile
from pathlib import Path

from tests import SAMPLE_ENV, SAMPLE_VIEWS


class DeterminismTests(unittest.TestCase):
    def test_shuffled_declaration_order_byte_identical(self):
        env_doc = load_json(SAMPLE_ENV)
        view_doc = load_json(SAMPLE_VIEWS)

        def run(doc_env, doc_views):
            spec = parse_specification(doc_env, doc_views)
            with tempfile.TemporaryDirectory() as tmp:
                session = Session(spec, Ledger(Path(tmp) / "l.sqlite"))
                session.run_all()
                return canonical_json(session.build_snapshot())

        baseline = run(env_doc, view_doc)

        shuffled_env = copy.deepcopy(env_doc)
        shuffled_env["environments"].reverse()
        for env in shuffled_env["environments"]:
            for key in env["observations"]:
                env["observations"][key].reverse()
        shuffled_views = copy.deepcopy(view_doc)
        shuffled_views["views"].reverse()
        for view in shuffled_views["views"]:
            view["env_ids"].reverse()
            view["policies"] = {k: view["policies"][k]
                                for k in reversed(list(view["policies"]))}

        self.assertEqual(baseline, run(shuffled_env, shuffled_views))

    def test_repeated_runs_keep_ledger_stable(self):
        spec = parse_specification(load_json(SAMPLE_ENV), load_json(SAMPLE_VIEWS))
        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "l.sqlite"
            first = Session(spec, Ledger(ledger_path))
            first.run_all()
            digest_a = canonical_json(first.build_snapshot())
            first.ledger.close()
            second = Session(spec, Ledger(ledger_path))
            digest_b = canonical_json(second.build_snapshot())
            self.assertEqual(digest_a, digest_b)


if __name__ == "__main__":
    unittest.main()
