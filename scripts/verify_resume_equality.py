"""比对全量报告与中断恢复报告的逐字段可比较载荷。"""

from __future__ import annotations

import json
import sys

from regbench.report import compare_reports


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print("用法: verify_resume_equality.py <full.json> <resume1.json> [resume2.json ...]")
        return 2
    reports: dict[str, dict] = {}
    for index, path in enumerate(argv):
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        label = "full" if index == 0 else f"resume_{index}"
        reports[label] = payload
    result = compare_reports(reports)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "identical" else 5


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
