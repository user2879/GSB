from pathlib import Path

from spmm import datasets
from spmm.config import EngineConfig
from spmm.inject import load_injection_config, run_injections

ROOT = Path(__file__).resolve().parents[1]


def test_all_injections_hit_expected_codes(tmp_path):
    datasets.generate_all(tmp_path / "data")
    a, b, c, _ = datasets.load_sample(tmp_path / "data", "uniform")
    cfg = EngineConfig.from_json(ROOT / "configs" / "default.json")
    injections = load_injection_config(ROOT / "configs" / "injections.json")
    report = run_injections(injections, a, b, c, cfg.to_dict(),
                            tmp_path / "injections")
    assert report["hits"] == report["total"] == len(injections)
    for item in report["injections"]:
        assert item["hit"], item
