from __future__ import annotations

import copy
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gsb.generation import corrupt_sample_file, ensure_scenario_samples
from gsb.model import load_scenario


EXAMPLES = ROOT / "examples"
ANOMALIES = EXAMPLES / "anomalies"


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def main() -> int:
    ANOMALIES.mkdir(parents=True, exist_ok=True)
    base_payload = json.loads((EXAMPLES / "demo_skewed.json").read_text(encoding="utf-8"))

    missing_version = copy.deepcopy(base_payload)
    missing_version["name"] = "anomaly-missing-parameter-version"
    missing_version["join"].pop("parameter_version", None)
    write_json(ANOMALIES / "missing_parameter_version.json", missing_version)

    zero_buckets = copy.deepcopy(base_payload)
    zero_buckets["name"] = "anomaly-zero-buckets"
    zero_buckets["join"]["buckets"] = 0
    write_json(ANOMALIES / "zero_buckets.json", zero_buckets)

    sample_exceeded = copy.deepcopy(base_payload)
    sample_exceeded["name"] = "anomaly-sample-size-exceeded"
    sample_exceeded["join"]["sample_size"] = 999999
    write_json(ANOMALIES / "sample_size_exceeded.json", sample_exceeded)

    invalid_spec = copy.deepcopy(base_payload)
    invalid_spec["name"] = "anomaly-invalid-spec"
    invalid_spec["tables"]["orders"]["columns"] = [
        column for column in invalid_spec["tables"]["orders"]["columns"] if column["name"] != "customer_id"
    ]
    invalid_spec["join"]["left_key"] = "does_not_exist"
    invalid_spec["join"]["left_predicates"].append({"column": "missing_filter_column", "op": "eq", "value": 1})
    write_json(ANOMALIES / "invalid_spec.json", invalid_spec)

    workdir = ROOT / "var" / "demo_generation"
    if workdir.exists():
        shutil.rmtree(workdir)
    scenario = load_scenario(EXAMPLES / "demo_skewed.json")
    samples = ensure_scenario_samples(scenario, workdir)
    first_name = sorted(samples)[0]
    source = Path(samples[first_name].path)
    target = ANOMALIES / "corrupted_sample.npz"
    shutil.copyfile(source, target)
    details = corrupt_sample_file(target)
    write_json(ANOMALIES / "corrupted_sample_manifest.json", details)

    readme = [
        "这些异常文件由 scripts/make_demo.py 生成。",
        "",
        "- missing_parameter_version.json: 直接解析触发 E_VERSION",
        "- zero_buckets.json: 直接解析触发 E_BUCKET",
        "- sample_size_exceeded.json: 完成过滤后触发 E_SAMPLE；上限为过滤后较小表的可用行数",
        "- invalid_spec.json: 多个声明错误聚合为 E_SPEC",
        "- corrupted_sample.npz: 截断后的 NPZ；配合样本清单摘要触发 E_SAMPLE",
        "",
        "推荐命令：",
        "python -m gsb run --spec examples/anomalies/missing_parameter_version.json --workdir var/anomalies/version",
        "python -m gsb run --spec examples/anomalies/zero_buckets.json --workdir var/anomalies/bucket",
        "python -m gsb run --spec examples/anomalies/sample_size_exceeded.json --workdir var/anomalies/sample",
        "python -m gsb run --spec examples/anomalies/invalid_spec.json --workdir var/anomalies/spec",
        "python -m gsb run --spec examples/demo_skewed.json --workdir var/anomalies/corrupt --inject-sample-corruption",
    ]
    (ANOMALIES / "README.txt").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(f"generated anomaly fixtures in {ANOMALIES}")
    print(details)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
