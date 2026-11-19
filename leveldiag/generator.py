"""生成入口：求解 -> 量化 -> 网格 -> 约束 -> 账本，并落盘全部工件。

输出目录结构：
  out/<name>/
    spec.canonical.json     规范化规格（复算 spec_hash 的依据）
    spec.source.yaml        原始 YAML 文本
    continuous/
      index.json            连续域索引（含哈希、seed_tree、归一化）
      region_<rid>.json     逐区域高程/密度场分片
      terrain.json          地形类别网格与抽样 uniforms
      resources.json
    mapping/
      index.json            量化规则、误差上界、分片哈希
      region_<rid>.json     逐区域映射分片
    grid.json
    ledger.json
    boundary_report.json
    manifest.json           全部文件内容哈希清单
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .constraints import evaluate
from .continuous import solve_continuous
from .errors import ErrorCode, LevelDiagError
from .grid import build_grid
from .ledger import boundary_report, build_ledger
from .quantize import quantize
from .spec import Spec, parse_spec, spec_from_dict
from .util import canonical_dumps, file_hash


def _write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_dumps(obj) + b"\n")


def run_pipeline(spec: Spec) -> dict:
    """纯内存流水线，返回各阶段对象（测试与增量模块复用）。"""
    continuous = solve_continuous(spec)
    mapping = quantize(spec, continuous)
    grid = build_grid(spec, continuous, mapping)
    entries = evaluate(spec, continuous, mapping, grid)
    ledger = build_ledger(spec, continuous, mapping, grid, entries)
    report = boundary_report(ledger)
    return {"continuous": continuous, "mapping": mapping,
            "grid": grid, "ledger": ledger, "boundary_report": report}


def write_artifacts(spec: Spec, result: dict, out_dir: str | Path,
                    source_yaml: str | None = None) -> dict:
    out = Path(out_dir)
    cont_dir = out / "continuous"
    map_dir = out / "mapping"
    continuous = result["continuous"]
    mapping = result["mapping"]
    grid = result["grid"]
    ledger = result["ledger"]
    report = result["boundary_report"]

    _write(out / "spec.canonical.json", spec.raw)
    if source_yaml is not None:
        (out / "spec.source.yaml").write_text(source_yaml, encoding="utf-8",
                                               newline="\n")
    shard_hashes = {}
    for rid, shard in continuous["region_shards"].items():
        p = cont_dir / f"region_{rid}.json"
        _write(p, shard)
        shard_hashes[f"continuous/region_{rid}.json"] = file_hash(p)
    _write(cont_dir / "terrain.json", {
        "terrain_grid": continuous["terrain_grid"],
        "terrain_uniforms": continuous["terrain_uniforms"]})
    _write(cont_dir / "resources.json", {"resources": continuous["resources"]})
    cont_index = {
        "spec_hash": continuous["spec_hash"],
        "continuous_hash": continuous["continuous_hash"],
        "seed_tree": continuous["seed_tree"],
        "weight_normalization": continuous["weight_normalization"],
        "region_shard_files": sorted(shard_hashes),
    }
    _write(cont_dir / "index.json", cont_index)

    map_shard_hashes = {}
    for rid, shard in mapping["region_shards"].items():
        p = map_dir / f"region_{rid}.json"
        _write(p, shard)
        map_shard_hashes[f"mapping/region_{rid}.json"] = file_hash(p)
    _write(map_dir / "index.json", {
        "spec_hash": mapping["spec_hash"],
        "continuous_hash": mapping["continuous_hash"],
        "mapping_hash": mapping["mapping_hash"],
        "rules": mapping["rules"],
        "observed_max_error": mapping["observed_max_error"],
        "region_shard_files": sorted(map_shard_hashes),
    })
    _write(out / "grid.json", grid)
    _write(out / "ledger.json", ledger)
    _write(out / "boundary_report.json", report)

    files = sorted(p for p in out.rglob("*") if p.is_file()
                   and p.name != "manifest.json")
    manifest = {
        "spec_hash": spec.spec_hash,
        "continuous_hash": continuous["continuous_hash"],
        "mapping_hash": mapping["mapping_hash"],
        "grid_hash": grid["grid_hash"],
        "ledger_hash": ledger["ledger_hash"],
        "files": {str(p.relative_to(out)).replace("\\", "/"): file_hash(p)
                  for p in files},
    }
    _write(out / "manifest.json", manifest)
    return manifest


def generate(spec_path: str | Path, out_dir: str | Path,
             strict_boundary: bool = False) -> dict:
    """从 YAML 文件生成完整工件。strict_boundary 下存在 UNCERTAIN 即报错。"""
    text = Path(spec_path).read_text(encoding="utf-8")
    spec = parse_spec(text, source_path=str(spec_path))
    result = run_pipeline(spec)
    manifest = write_artifacts(spec, result, out_dir, source_yaml=text)
    if strict_boundary and result["ledger"]["summary"]["uncertain"]:
        raise LevelDiagError(
            ErrorCode.E_BOUNDARY_AMBIGUOUS,
            "存在落在误差区间内的不确定判定（工件仍已完整写出）",
            {"uncertain_ids": result["ledger"]["summary"]["uncertain_ids"],
             "out_dir": str(out_dir)})
    return manifest


def generate_from_dict(data: dict, out_dir: str | Path,
                       strict_boundary: bool = False) -> dict:
    spec = spec_from_dict(data)
    result = run_pipeline(spec)
    text = yaml.safe_dump(data, sort_keys=True, allow_unicode=True,
                          default_flow_style=False)
    manifest = write_artifacts(spec, result, out_dir, source_yaml=text)
    if strict_boundary and result["ledger"]["summary"]["uncertain"]:
        raise LevelDiagError(
            ErrorCode.E_BOUNDARY_AMBIGUOUS,
            "存在落在误差区间内的不确定判定（工件仍已完整写出）",
            {"uncertain_ids": result["ledger"]["summary"]["uncertain_ids"],
             "out_dir": str(out_dir)})
    return manifest
