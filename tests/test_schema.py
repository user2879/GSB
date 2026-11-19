from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from mesh_audit.cli import main
from mesh_audit.errors import BusinessError, ErrorCode
from mesh_audit.schema import validate_mesh_payload, validate_repair_script
from tests.helpers import make_script


class SchemaValidationTests(unittest.TestCase):
    def test_non_finite_coordinate_is_schema_error(self) -> None:
        with self.assertRaises(BusinessError) as caught:
            validate_mesh_payload(
                {"vertices": [[0, 0, 0], [1, 0, 0], [0, 0, float("nan")]], "faces": [[0, 1, 2]]}
            )
        self.assertEqual(caught.exception.code, ErrorCode.E_SCHEMA)

    def test_out_of_bounds_index_is_index_error(self) -> None:
        with self.assertRaises(BusinessError) as caught:
            validate_mesh_payload(
                {"vertices": [[0, 0, 0], [1, 0, 0], [0, 0, 1]], "faces": [[0, 1, 9]]}
            )
        self.assertEqual(caught.exception.code, ErrorCode.E_INDEX)

    def test_unknown_operator_is_schema_error(self) -> None:
        payload = {
            "coordinate_tolerance": 1e-7,
            "operators": [{"type": "delete_everything"}],
        }
        with self.assertRaises(BusinessError) as caught:
            validate_repair_script(payload)
        self.assertEqual(caught.exception.code, ErrorCode.E_SCHEMA)

    def test_non_positive_tolerance_is_schema_error(self) -> None:
        with self.assertRaises(BusinessError) as caught:
            validate_repair_script(
                {"coordinate_tolerance": -1.0, "operators": [{"type": "reorient_normals"}]}
            )
        self.assertEqual(caught.exception.code, ErrorCode.E_SCHEMA)

    def test_cli_invalid_input_writes_no_state_or_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bad_mesh = root / "bad.mesh.json"
            bad_mesh.write_text(
                json.dumps(
                    {
                        "vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                        "faces": [[0, 1, 9]],
                    }
                ),
                encoding="utf-8",
            )
            script_path = root / "script.json"
            script_path.write_text(
                json.dumps(make_script().normalized), encoding="utf-8"
            )
            state = root / "state-must-not-exist"
            output = root / "output-must-not-exist"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = main(
                    [
                        "run",
                        "--mesh",
                        str(bad_mesh),
                        "--script",
                        str(script_path),
                        "--state",
                        str(state),
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(code, 2)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["error"]["code"], ErrorCode.E_INDEX)
            self.assertFalse(state.exists())
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
