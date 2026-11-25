from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from .config import Config, validate_config
from .formats import make_v1_line, parse_v1_lines
from .loader import load_requests_jsonl


def run_injection_plan(plan_path: str | Path, config: Config) -> dict[str, Any]:
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    results: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="candidate-injections-") as tmp:
        tmp_path = Path(tmp)
        for injection in plan["injections"]:
            kind = injection["kind"]
            injection_id = injection["id"]
            observed: list[str] = []
            business_codes: list[str | None] = []
            detail: dict[str, Any] = {}

            if kind == "non_finite_score":
                value = float("inf") if injection.get("value") == "Infinity" else float("nan")
                row = {
                    "request_id": injection["target_request_id"],
                    "candidates": [
                        {
                            "candidate_id": 9001,
                            "raw_score": value,
                            "aux_feature": 1,
                            "channel": "search",
                            "_injection_id": injection_id,
                        }
                    ],
                }
                path = tmp_path / f"{injection_id}.jsonl"
                path.write_text(json.dumps(row, allow_nan=True) + "\n", encoding="utf-8")
                _, failures = load_requests_jsonl(path, config)
                observed = [f.code for f in failures]
                business_codes = [f.business_code for f in failures]
                detail["failures"] = [f.to_dict() for f in failures]

            elif kind == "duplicate_candidate_id":
                row = {
                    "request_id": injection["target_request_id"],
                    "candidates": [
                        {"candidate_id": 9002, "raw_score": 10.0, "channel": "search",
                         "_injection_id": injection_id},
                        {"candidate_id": 9002, "raw_score": 20.0, "channel": "feed",
                         "_injection_id": injection_id},
                    ],
                }
                path = tmp_path / f"{injection_id}.jsonl"
                path.write_text(json.dumps(row) + "\n", encoding="utf-8")
                _, failures = load_requests_jsonl(path, config)
                observed = [f.code for f in failures]
                detail["failures"] = [f.to_dict() for f in failures]

            elif kind == "candidate_count_over_limit":
                count = int(injection["count"])
                candidates = [
                    {
                        "candidate_id": 9100 + index,
                        "raw_score": 10.0 + index,
                        "aux_feature": index,
                        "channel": "search",
                    }
                    for index in range(count)
                ]
                row = {"request_id": "inj-count", "candidates": candidates}
                path = tmp_path / f"{injection_id}.jsonl"
                path.write_text(json.dumps(row) + "\n", encoding="utf-8")
                _, failures = load_requests_jsonl(path, config)
                observed = [f.code for f in failures]
                detail["failures"] = [f.to_dict() for f in failures]

            elif kind == "sampling_ratio_out_of_range":
                raw = dict(config.raw)
                raw["sampling"] = dict(config.raw["sampling"])
                raw["sampling"]["ratio"] = injection["ratio"]
                raw["sampling"]["ratio_permille"] = injection["ratio_permille"]
                failures = validate_config(raw)
                observed = [f.code for f in failures]
                business_codes = [f.business_code for f in failures]
                detail["failures"] = [f.to_dict() for f in failures]

            elif kind == "unknown_selection_code":
                records, failures = parse_v1_lines([injection["v1_line"]], config, source=injection_id)
                observed = [f.code for f in failures]
                business_codes = [f.business_code for f in failures]
                detail = {"parsed_records": records, "failures": [f.to_dict() for f in failures]}

            elif kind == "score_field_width_insufficient":
                narrow_raw = dict(config.raw)
                narrow_raw["v1_fields"] = dict(config.raw["v1_fields"])
                narrow_raw["v1_fields"]["score_permille"] = int(injection["score_width"])
                narrow_config = Config(narrow_raw)
                record = {
                    "request_id": "inj-width",
                    "seq": 1,
                    "candidate_id": 9003,
                    "score_permille": int(injection["score_permille"]),
                    "selection_code": 1,
                }
                line, failure = make_v1_line(record, narrow_config)
                if failure:
                    observed = [failure.code]
                    business_codes = [failure.business_code]
                    detail["failure"] = failure.to_dict()
                detail["line"] = line

            expected = injection["expected_code"]
            expected_business = injection.get("business_code")
            hit = expected in observed or (expected_business and expected_business in business_codes)
            results.append(
                {
                    "injection_id": injection_id,
                    "kind": kind,
                    "expected_code": expected,
                    "expected_business_code": expected_business,
                    "observed_codes": observed,
                    "observed_business_codes": business_codes,
                    "hit": hit,
                    "detail": detail,
                }
            )

    return {
        "schema": "injection-report-v1",
        "total": len(results),
        "hit_count": sum(1 for item in results if item["hit"]),
        "all_hit": all(item["hit"] for item in results),
        "results": results,
    }
