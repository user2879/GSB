from __future__ import annotations

import unittest

import numpy as np

from batch_audit.canonical import (
    assignment_identifier,
    digest_inputs,
    digest_model,
    digest_sample_output,
)
from batch_audit.constants import MODEL_ARRAY_ORDER
from batch_audit.dataset import generate_assets
from batch_audit.forward import forward_batch, forward_one


class DeterminismTests(unittest.TestCase):
    def test_assets_are_bit_identical_for_same_arguments(self) -> None:
        first = generate_assets(seed=77, sample_count=9, input_dim=5, hidden_dim=7)
        second = generate_assets(seed=77, sample_count=9, input_dim=5, hidden_dim=7)
        self.assertEqual(first.sample_ids, second.sample_ids)
        self.assertEqual(first.inputs.tobytes(), second.inputs.tobytes())
        for name in MODEL_ARRAY_ORDER:
            self.assertEqual(first.model[name].tobytes(), second.model[name].tobytes())
        self.assertEqual(
            digest_inputs(first.sample_ids, first.inputs),
            digest_inputs(second.sample_ids, second.inputs),
        )
        self.assertEqual(
            digest_model(first.model, MODEL_ARRAY_ORDER),
            digest_model(second.model, MODEL_ARRAY_ORDER),
        )

    def test_different_seed_changes_assets(self) -> None:
        first = generate_assets(seed=1, sample_count=4, input_dim=3, hidden_dim=4)
        second = generate_assets(seed=2, sample_count=4, input_dim=3, hidden_dim=4)
        self.assertNotEqual(first.inputs.tobytes(), second.inputs.tobytes())
        self.assertNotEqual(first.model["W1"].tobytes(), second.model["W1"].tobytes())

    def test_input_digest_is_independent_of_row_order(self) -> None:
        assets = generate_assets(seed=9, sample_count=6, input_dim=4, hidden_dim=5)
        order = [4, 0, 5, 2, 1, 3]
        shuffled_ids = [assets.sample_ids[index] for index in order]
        shuffled_inputs = assets.inputs[order]
        self.assertEqual(
            digest_inputs(assets.sample_ids, assets.inputs),
            digest_inputs(shuffled_ids, shuffled_inputs),
        )

    def test_forward_and_identifiers_are_stable(self) -> None:
        assets = generate_assets(seed=11, sample_count=5, input_dim=4, hidden_dim=6)
        inputs = {sample_id: assets.inputs[index] for index, sample_id in enumerate(assets.sample_ids)}
        first = forward_batch(list(assets.sample_ids), inputs, assets.model)
        second = forward_batch(list(reversed(assets.sample_ids)), inputs, assets.model)
        self.assertEqual(first.batch_output_digest, second.batch_output_digest)
        for sample_id in assets.sample_ids:
            self.assertEqual(first.sample_digests[sample_id], second.sample_digests[sample_id])
            one = forward_one(inputs[sample_id], assets.model)
            again = forward_one(inputs[sample_id], assets.model)
            self.assertEqual(one.tobytes(), again.tobytes())
            self.assertEqual(
                digest_sample_output(sample_id, one),
                digest_sample_output(sample_id, again),
            )
        param_digest = digest_model(assets.model, MODEL_ARRAY_ORDER)
        id1 = assignment_identifier(
            numbering_version="BA-V1",
            batch_slot=2,
            rank=3,
            sample_id="S0001",
            param_digest=param_digest,
        )
        id2 = assignment_identifier(
            numbering_version="BA-V1",
            batch_slot=2,
            rank=3,
            sample_id="S0001",
            param_digest=param_digest,
        )
        self.assertEqual(id1, id2)
        self.assertNotEqual(
            id1,
            assignment_identifier(
                numbering_version="BA-V1",
                batch_slot=2,
                rank=4,
                sample_id="S0001",
                param_digest=param_digest,
            ),
        )


if __name__ == "__main__":
    unittest.main()
