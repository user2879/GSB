"""Exercise every required business error code against local fixtures."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feature_pipeline.config import parse_merge_config
from feature_pipeline.errors import (
    E_CONFIG_INVALID,
    E_LINE_OVERFLOW,
    E_NONFINITE_INPUT,
    E_OFFSET_OUT_OF_RANGE,
    E_PRECISION_INVALID,
)
from feature_pipeline.conversion import validate_precision
from feature_pipeline.models import Cell, FeatureMeta
from feature_pipeline.parsers import v1_fixedwidth, v2_json, v3_binary


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        results = []

        # E_CONFIG_INVALID
        try:
            parse_merge_config({"feature_whitelist": []})
        except Exception as exc:
            results.append(("config_missing_field", exc.code))

        # E_PRECISION_INVALID
        meta = {"f_a": FeatureMeta("f_a", "continuous", ("float32",))}
        try:
            validate_precision(
                Cell("e1", "f_a", "v2", 1.0, False, "float64"), meta
            )
        except Exception as exc:
            results.append(("precision_out_of_range", exc.code))

        # E_LINE_OVERFLOW
        bad_v1 = work / "overflow.txt"
        bad_v1.write_text("x" * (v1_fixedwidth.LINE_WIDTH + 3), encoding="utf-8")
        try:
            v1_fixedwidth.parse_file(bad_v1)
        except Exception as exc:
            results.append(("fixed_width_overflow", exc.code))

        # E_NONFINITE_INPUT
        bad_v2 = work / "nonfinite.json"
        bad_v2.write_text(
            json.dumps(
                {
                    "format_version": "v2",
                    "entities": {
                        "e1": {"f_a": {"value": float("inf"),
                                       "precision": "float64"}}
                    },
                }
            ),
            encoding="utf-8",
        )
        try:
            v2_json.parse_file(bad_v2)
        except Exception as exc:
            results.append(("non_finite_value", exc.code))

        # E_OFFSET_OUT_OF_RANGE: truncate a valid v3 file.
        from tests.test_parsers import sample_v1_table
        from feature_pipeline.models import V3, EntityTable

        table = EntityTable(source_version=V3, feature_order=["f_a"])
        table.partition_boundaries[0] = ["e1", "e2"]
        table.cells = [
            Cell(entity, "f_a", V3, 1.0, False, "float32", partition_id=0)
            for entity in ("e1", "e2")
        ]
        good_v3 = work / "good.bin"
        v3_binary.write_file(good_v3, table)
        truncated = work / "truncated.bin"
        truncated.write_bytes(good_v3.read_bytes()[:-6])
        try:
            v3_binary.parse_file(truncated)
        except Exception as exc:
            results.append(("partition_offset", exc.code))

        expected = {
            E_CONFIG_INVALID,
            E_PRECISION_INVALID,
            E_LINE_OVERFLOW,
            E_NONFINITE_INPUT,
            E_OFFSET_OUT_OF_RANGE,
        }
        observed = {code for _, code in results}
        for name, code in results:
            print(f"{name:26s} -> {code}")
        missing = expected - observed
        if missing:
            print(f"MISSING CODES: {sorted(missing)}", file=sys.stderr)
            return 1
        print("all five business codes observed")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
