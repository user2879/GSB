from __future__ import annotations

import copy
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mlpipe_workbench.ledger import Ledger
from mlpipe_workbench.loader import load_workspace, prepare_from_objects

from tests.helpers import DATA, PIPELINE, base_objects, prepared_base


class FingerprintTests(unittest.TestCase):
    def test_stage_array_order_does_not_change_fingerprint(self) -> None:
        data, pipeline = base_objects()
        pipeline["stages"] = list(reversed(pipeline["stages"]))
        first = prepare_from_objects(data, pipeline)
        second = prepared_base()
        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertEqual(
            [stage.id for stage in first.pipeline.stages],
            ["canonicalize", "fill_missing", "normalize_amount", "clip_amount", "pack"],
        )

    def test_parameter_object_order_does_not_change_fingerprint(self) -> None:
        data, pipeline = base_objects()
        params = pipeline["stages"][3]["params"]
        pipeline["stages"][3]["params"] = {
            "max_outlier_permille": params["max_outlier_permille"],
            "column": params["column"],
            "z_limit_permille": params["z_limit_permille"],
        }
        self.assertEqual(prepare_from_objects(data, pipeline).fingerprint, prepared_base().fingerprint)

    def test_content_changes_change_fingerprint(self) -> None:
        base = prepared_base()
        data, pipeline = base_objects()
        data["sources"][0]["columns"][1]["values"][50] = "999.000"
        sample_changed = prepare_from_objects(data, pipeline)

        data2, pipeline2 = base_objects()
        pipeline2["stages"][1]["version"] = 2
        stage_changed = prepare_from_objects(data2, pipeline2)
        self.assertNotEqual(base.fingerprint, sample_changed.fingerprint)
        self.assertNotEqual(base.fingerprint, stage_changed.fingerprint)

    def test_upstream_reference_changes_change_fingerprint(self) -> None:
        data, pipeline = base_objects()
        fill = copy.deepcopy(pipeline["stages"][1])
        fill["id"] = "fill_missing_b"
        stages = pipeline["stages"]
        # Insert a second fill stage, then swap the two fill stages in the chain.
        stages.insert(1, copy.deepcopy(fill))
        stages[1]["id"] = "fill_missing_a"
        stages[1]["upstream"] = "canonicalize"
        stages[2]["upstream"] = "fill_missing_a"
        stages[2]["id"] = "fill_missing_b"
        stages[3]["upstream"] = "fill_missing_b"
        first = prepare_from_objects(data, pipeline)

        data2, pipeline2 = base_objects()
        fill2 = copy.deepcopy(pipeline2["stages"][1])
        fill2["id"] = "fill_missing_b"
        stages2 = pipeline2["stages"]
        stages2.insert(1, copy.deepcopy(fill2))
        stages2[1]["id"] = "fill_missing_a"
        # Reversed chain: canonicalize -> b -> a -> normalize.
        pipeline2["stages"][1]["upstream"] = "fill_missing_b"
        pipeline2["stages"][2]["id"] = "fill_missing_b"
        pipeline2["stages"][2]["upstream"] = "canonicalize"
        pipeline2["stages"][3]["upstream"] = "fill_missing_a"
        second = prepare_from_objects(data2, pipeline2)
        self.assertNotEqual(first.fingerprint, second.fingerprint)

    def test_concurrent_loads_produce_identical_fingerprint(self) -> None:
        with ThreadPoolExecutor(max_workers=8) as pool:
            fingerprints = list(pool.map(lambda _: load_workspace(DATA, PIPELINE).fingerprint, range(16)))
        self.assertEqual(len(set(fingerprints)), 1)

    def test_spec_versions_are_monotonic_for_new_fingerprints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.db"
            base = prepared_base()
            data, pipeline = base_objects()
            data["sources"][0]["columns"][1]["values"][50] = "999.000"
            changed = prepare_from_objects(data, pipeline)
            with Ledger(path) as ledger:
                v1, new1, _ = ledger.allocate_spec(base)
                v1_again, new_again, _ = ledger.allocate_spec(base)
                v2, new2, _ = ledger.allocate_spec(changed)
            self.assertEqual((v1, new1), (1, True))
            self.assertEqual((v1_again, new_again), (1, False))
            self.assertEqual((v2, new2), (2, True))


if __name__ == "__main__":
    unittest.main()
