"""按注入配置执行异常注入样例，输出业务码映射报告。"""

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spmm import datasets
from spmm.config import EngineConfig
from spmm.inject import load_injection_config, run_injections

ROOT = Path(__file__).resolve().parents[1]


def main():
    cfg = EngineConfig.from_json(ROOT / "configs" / "default.json")
    data_dir = ROOT / cfg.data_dir
    a, b, c, _ = datasets.load_sample(data_dir, "uniform")
    injections = load_injection_config(ROOT / "configs" / "injections.json")
    work_root = ROOT / "state" / "injections"
    if work_root.exists():
        shutil.rmtree(work_root)
    report = run_injections(injections, a, b, c, cfg.to_dict(), work_root)
    report_dir = ROOT / cfg.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    out = report_dir / "injections_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({"hits": report["hits"], "total": report["total"],
                      "report": str(out)}, ensure_ascii=False))
    return 0 if report["hits"] == report["total"] else 1


if __name__ == "__main__":
    sys.exit(main())
