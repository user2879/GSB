from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from candidate_diagnostics.config import load_config
from candidate_diagnostics.formats import result_to_records, serialize_v1
from candidate_diagnostics.loader import load_requests_jsonl
from candidate_diagnostics.pipeline import analyze_many


def main() -> None:
    config = load_config(ROOT / "samples" / "config.json")
    requests, failures = load_requests_jsonl(ROOT / "samples" / "candidates.jsonl", config)
    if failures:
        raise SystemExit([failure.to_dict() for failure in failures])
    results = analyze_many(requests, config)
    records = [
        record
        for request_id in sorted(results)
        for record in result_to_records(results[request_id], config)
    ]
    lines, v1_failures = serialize_v1(records, config)
    if v1_failures:
        raise SystemExit([failure.to_dict() for failure in v1_failures])
    (ROOT / "samples" / "conclusions.generated.v1.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (ROOT / "samples" / "conclusions.generated.v2.json").write_text(
        json.dumps(
            {"schema": "candidate-conclusion-v2-batch", "records": records},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(lines)} v1 lines and {len(records)} v2 records")


if __name__ == "__main__":
    main()
