from __future__ import annotations

import unittest

from spectral_diag.errors import ErrorCode


class ErrorCodeTests(unittest.TestCase):
    def test_required_error_codes_exist(self) -> None:
        expected = {
            "E_MATRIX_SINGULAR",
            "E_QUANT_RANGE",
            "E_WHITE_POINT",
            "E_OBSERVER_INVALID",
            "E_ATTRIB_MISMATCH",
            "E_SNAPSHOT_VERSION",
        }
        self.assertTrue(expected <= {str(code) for code in ErrorCode})


if __name__ == "__main__":
    unittest.main()
