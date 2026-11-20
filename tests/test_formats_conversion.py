from __future__ import annotations

import json

from pref_reward_lab.config import LabConfig
from pref_reward_lab.conversion import (
    convert_v1_to_v2,
    convert_v2_to_v1,
    float_to_milli_half_up,
)
from pref_reward_lab.formats import (
    V1_LINE_WIDTH,
    V1Record,
    V2Record,
    encode_v1_record,
    parse_v1_text,
    parse_v2_text,
)
from pref_reward_lab.sample import default_config


def config() -> LabConfig:
    return LabConfig.from_dict(default_config())


def test_v1_fixed_width_roundtrip_and_invalid_rows() -> None:
    record = V1Record("pair-000001", 612, 123)
    line = encode_v1_record(record)
    assert len(line) == V1_LINE_WIDTH
    parsed = parse_v1_text(line + "\nBROKEN\n")
    assert parsed.records["pair-000001"] == record
    assert len(parsed.failures) == 1
    assert parsed.failures[0]["code"] == "E_FORMAT_INVALID"


def test_v2_array_jsonl_alias_and_enum_failures() -> None:
    cfg = config()
    array_text = json.dumps(
        [
            {
                "pair_id": "pair-000001",
                "reward_score": 0.5,
                "reward_delta": 0.1,
                "reward_label": "preference",
                "precision_label": "float64",
            },
            {
                "pair_id": "pair-000002",
                "score": 0.25,
                "delta": 0.05,
                "label": "neutral",
                "precision": "float32",
            },
            {
                "pair_id": "pair-000003",
                "reward_score": 0.3,
                "reward_delta": 0.1,
                "reward_label": "bad-label",
                "precision_label": "float64",
            },
        ]
    )
    parsed = parse_v2_text(array_text, cfg)
    assert set(parsed.records) == {"pair-000001", "pair-000002"}
    assert len(parsed.failures) == 1
    assert parsed.failures[0]["pair_id"] == "pair-000003"

    jsonl = "\n".join(
        [
            json.dumps(
                {
                    "pair_id": "pair-000004",
                    "reward_score": 0.4,
                    "reward_delta": 0.2,
                    "reward_label": "preference",
                    "precision_label": "float64",
                }
            ),
            "{bad json}",
        ]
    )
    parsed_jsonl = parse_v2_text(jsonl, cfg)
    assert "pair-000004" in parsed_jsonl.records
    assert parsed_jsonl.failures[0]["code"] == "E_FORMAT_INVALID"


def test_v1_to_v2_exact_milli_and_downside_sidecar_restore() -> None:
    cfg = config()
    v1 = V1Record("pair-000001", 612, -123)
    up = convert_v1_to_v2([v1], cfg)
    assert up.records[0].reward_score == 0.612
    assert up.records[0].reward_delta == -0.123
    assert up.records[0].precision_label == "milli_exact"

    v2 = V2Record("pair-000001", 0.6124, -0.1226, "preference", "float64")
    down = convert_v2_to_v1([v2], cfg)
    assert down.records[0] == V1Record("pair-000001", 612, -123)
    entry = down.sidecar["records"]["pair-000001"]
    assert entry["reward_label"] == "preference"
    assert entry["precision_label"] == "float64"
    assert entry["score_residual_milli"] == "0.4"
    assert entry["delta_residual_milli"] == "0.4"
    assert entry["v1_content_fingerprint"]

    restored = convert_v1_to_v2(down.records, cfg, sidecar=down.sidecar)
    assert restored.restored_pair_ids == ["pair-000001"]
    assert restored.records[0] == v2
    assert restored.losses["max_abs_score_loss"] == 0.0


def test_half_up_rounding_is_symmetric() -> None:
    assert float_to_milli_half_up(1.2345) == 1235
    assert float_to_milli_half_up(-1.2345) == -1235
    assert float_to_milli_half_up(1.2344) == 1234


def test_sidecar_missing_injection_is_pair_local() -> None:
    cfg = config()
    records = [
        V1Record("pair-000001", 500, 100),
        V1Record("pair-000002", 600, 200),
    ]
    outcome = convert_v1_to_v2(
        records,
        cfg,
        force_missing_sidecar_pair="pair-000001",
        force_missing_sidecar_injection_id="INJ-X",
    )
    assert [record.pair_id for record in outcome.records] == ["pair-000002"]
    assert outcome.failures[0]["code"] == "E_SIDECAR_MISSING"
    assert outcome.failures[0]["injection_id"] == "INJ-X"

