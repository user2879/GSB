"""Requirements 3 & 8: deterministic encoding/init and concurrency parity."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest

import numpy as np

from dbench.config import load_config
from dbench.injection import load_injections
from dbench.orchestrator import Orchestrator
from dbench.stages import encode_prompt, initialize_latent, load_vocab
from dbench.store import SnapshotStore

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CFG = os.path.join(ROOT, "configs", "valid_base.json")


def _load_cfg():
    cfg = load_config(CFG)
    base = os.path.dirname(CFG)
    import dataclasses

    return dataclasses.replace(
        cfg,
        vocab_file=os.path.normpath(os.path.join(base, cfg.vocab_file)),
    )


class DeterminismTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="dbench-det-")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_embedding_and_latent_elementwise_stable(self):
        cfg = _load_cfg()
        vocab = load_vocab(cfg.vocab_file)
        emb_a, _ = encode_prompt(cfg.prompt, vocab)
        emb_b, _ = encode_prompt(cfg.prompt, vocab)
        np.testing.assert_array_equal(emb_a, emb_b)
        self.assertAlmostEqual(float(emb_a.sum()), 1.0, places=12)

        from dbench.hashing import fingerprint_array

        lat_a, _ = initialize_latent(cfg, fingerprint_array(emb_a),
                                     tuple(cfg.latent_shape))
        lat_b, _ = initialize_latent(cfg, fingerprint_array(emb_b),
                                     tuple(cfg.latent_shape))
        np.testing.assert_array_equal(lat_a, lat_b)

    def test_concurrency_matches_serial_with_and_without_jitter(self):
        import dataclasses

        reports = {}
        arrays = {}
        for tag, workers, jitter in (("serial", 1, False),
                                     ("parallel", 4, False),
                                     ("jitter", 4, True)):
            cfg = dataclasses.replace(_load_cfg(), max_workers=workers)
            store = SnapshotStore(os.path.join(self.tmp, tag))
            orch = Orchestrator(cfg, store, jitter=jitter,
                                output_png=os.path.join(self.tmp, tag + ".png"))
            reports[tag] = orch.run()
            arrays[tag] = {
                item["stage"]: store.load_array(item["stage"], item["version"])
                for item in reports[tag]["stages"]
            }
        for stage in arrays["serial"]:
            np.testing.assert_array_equal(arrays["serial"][stage],
                                          arrays["parallel"][stage])
            np.testing.assert_array_equal(arrays["serial"][stage],
                                          arrays["jitter"][stage])
        # Failure list / stage list ordering is identical (same sort keys).
        order = [s["stage"] for s in reports["serial"]["stages"]]
        for tag in ("parallel", "jitter"):
            self.assertEqual(
                order, [s["stage"] for s in reports[tag]["stages"]]
            )
            self.assertEqual(reports["serial"]["failures"],
                             reports[tag]["failures"])


if __name__ == "__main__":
    unittest.main()
