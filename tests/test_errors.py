from __future__ import annotations

import unittest

from volumetric_editor.errors import BusinessError, ErrorCode
from volumetric_editor.scene import generate_scene
from volumetric_editor.session import IncrementalSession


class ErrorCodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scene = generate_scene(seed=5150, width=32, height=24)
        cls.session = IncrementalSession(cls.scene, tile_size=8)
        cls.session.render_full(threads=2)

    def assert_code(self, expected: ErrorCode, action) -> None:
        with self.assertRaises(BusinessError) as context:
            action()
        self.assertEqual(context.exception.code, expected)

    def test_business_errors_do_not_crash_or_silently_skip(self) -> None:
        self.assert_code(
            ErrorCode.TILE_SIZE,
            lambda: self.session.render_full(threads=2, tile_size=0),
        )
        self.assert_code(
            ErrorCode.THREAD_BOUNDS,
            lambda: self.session.render_full(threads=0),
        )
        self.assert_code(
            ErrorCode.IOR_RANGE,
            lambda: self.session.edit_ior("lens_left", 9.0),
        )
        self.assert_code(
            ErrorCode.MEDIUM_COEFF,
            lambda: self.session.edit_scatter(-0.1),
        )
        self.assert_code(
            ErrorCode.BUDGET_DENIED,
            lambda: IncrementalSession(self.scene, 8).render_full(
                threads=2, queue_memory_budget_bytes=0
            ),
        )
        self.assert_code(
            ErrorCode.BUDGET_DENIED,
            lambda: IncrementalSession(self.scene, 8).render_full(
                threads=2, compute_budget_units=0
            ),
        )


if __name__ == "__main__":
    unittest.main()
