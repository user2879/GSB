import unittest

import numpy as np

from fxpipe.synthetic import (
    integer_permille_table,
    synthetic_bias_vector,
    synthetic_embedding_matrix,
)


class SyntheticTests(unittest.TestCase):
    def test_embedding_matrix_is_deterministic_and_integer(self):
        first = synthetic_embedding_matrix(12, 8, seed=20261126)
        second = synthetic_embedding_matrix(12, 8, seed=20261126)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(first.shape, (12, 8))
        self.assertTrue(np.issubdtype(first.dtype, np.integer))
        # A different seed must produce different data.
        self.assertFalse(np.array_equal(first, synthetic_embedding_matrix(12, 8, seed=1)))

    def test_bias_vector_shape_and_dtype(self):
        bias = synthetic_bias_vector(8, seed=42)
        self.assertEqual(bias.shape, (8,))
        self.assertTrue(np.issubdtype(bias.dtype, np.integer))

    def test_permille_table_rows_sum_to_1000(self):
        table = integer_permille_table(rows=6, cols=4, seed=7)
        self.assertEqual(table.shape, (6, 4))
        self.assertTrue(np.all(table >= 0))
        self.assertTrue(np.all(table.sum(axis=1) == 1000))


if __name__ == "__main__":
    unittest.main()
