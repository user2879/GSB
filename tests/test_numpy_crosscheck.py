import os
import shutil
import tempfile
import unittest

import numpy as np

from rdiq.engine import Engine
from rdiq.fixedpoint import WEIGHT_BASE, clamp, div_trunc
from rdiq.models import load_dataset
from rdiq.synth import generate_sample_set


class NumpyCrossCheckTest(unittest.TestCase):
    def test_engine_matches_independent_numpy_integer_recomputation(self):
        temp_dir = tempfile.mkdtemp(prefix="rdiq-numpy-")
        try:
            sample_dir = os.path.join(temp_dir, "samples")
            generate_sample_set(sample_dir)
            dataset = load_dataset(sample_dir)
            engine = Engine(
                dataset,
                os.path.join(temp_dir, "ledger.sqlite"),
                os.path.join(temp_dir, "backups"),
            )
            result = engine.run()
            width = dataset.config.precision["width"]
            scale = np.int64(dataset.config.precision["scale"])

            for session_payload in result["sessions"]:
                for record in session_payload["rounds"]:
                    weights = np.array(
                        [chain["quantized_weight"] for chain in record["adopted"]],
                        dtype=np.int64,
                    )
                    hits = np.array(
                        [chain["relevance_hits"] for chain in record["adopted"]],
                        dtype=np.int64,
                    )
                    if weights.size:
                        products = weights * hits
                        weighted_sum = int(products.sum(dtype=np.int64))
                        confidence, _ = clamp(
                            div_trunc(weighted_sum, int(weights.size))[0], width
                        )
                    else:
                        weighted_sum, confidence = 0, 0
                    self.assertEqual(weighted_sum, record["weighted_sum"])
                    self.assertEqual(confidence, record["confidence"])

                    # Independently quantize one raw channel weight with NumPy.
                    if record["adopted"]:
                        chain = record["adopted"][0]
                        numpy_quant = int(
                            np.floor_divide(
                                np.int64(chain["channel_raw"]) * scale,
                                np.int64(WEIGHT_BASE),
                            )
                        )
                        self.assertEqual(
                            numpy_quant, chain["channel_quantized"]
                        )
            engine.close()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
