"""YAML 关卡规格解析、校验与双精度数值模型。

规格结构（字段路径在错误信息中按点号给出，如 constraints.density_cap）：

grid: {width, height}
quantization: {elevation_step, density_step, rounding}   # rounding: nearest/down/up
terrain:
  types: [grass, rock, water, ...]
  weights: {grass: 0.6, ...}        # 和与 1 的偏差须 <= weight_normalization_tolerance
  weight_normalization_tolerance: 1.0e-9
regions:
  - id: r1
    cells: [[x, y], ...]            # 网格坐标，不得越界/重叠
connectivity:
  - id: c1
    region_ids: [r1, r2]
    min_channel_width_ratio: 0.25   # 双精度阈值
    tolerance: 1.0e-12
resources:
  count: {min: 3, max: 6}
  density_cap: 0.08                 # 资源数 / 区域单元数 上限
  tolerance: 1.0e-12
terrain_rules:
  passable: [grass, rock]
  impassable: [water]
  forbidden_adjacent: [["water", "spawn"]]
points:
  spawn: [x, y]
  target: [x, y]
thresholds:                          # 额外的双精度判定阈值
  elevation_pass: 0.5
  cell_density_max: 0.75
  cell_density_tolerance: 1.0e-12
seed: 20241119
density_anchor:                      # 可选，用于确定性边界样本
  region_id: r2
  cell: [x, y]
  continuous_density: 0.375
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .errors import ErrorCode, LevelDiagError
from .util import canonical_dumps, content_hash


REQUIRED_TOP_LEVEL = [
    "grid",
    "quantization",
    "terrain",
    "regions",
    "connectivity",
    "resources",
    "terrain_rules",
    "points",
    "thresholds",
    "seed",
]


def _invalid(path: str, reason: str) -> LevelDiagError:
    return LevelDiagError(
        ErrorCode.E_SPEC_INVALID,
        f"规格非法: {path}: {reason}",
        {"field_path": path, "reason": reason},
    )


def _require(mapping: dict, key: str, path: str):
    if not isinstance(mapping, dict) or key not in mapping:
        raise _invalid(f"{path}.{key}" if path else key, "缺失字段")
    return mapping[key]


def _as_float(value, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _invalid(path, f"期望双精度小数，实际为 {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise _invalid(path, "数值必须有限")
    return result


def _as_int(value, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _invalid(path, f"期望整数，实际为 {type(value).__name__}")
    return int(value)


@dataclass
class Spec:
    """校验后的规格。raw 保持 YAML 原生结构（已排序键），哈希基于规范化序列化。"""

    raw: dict
    spec_hash: str
    source_path: str | None = None

    # 常用访问器
    @property
    def width(self) -> int:
        return int(self.raw["grid"]["width"])

    @property
    def height(self) -> int:
        return int(self.raw["grid"]["height"])

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])


def load_spec(path: str | Path) -> Spec:
    text = Path(path).read_text(encoding="utf-8")
    return parse_spec(text, source_path=str(path))


def parse_spec(text: str, source_path: str | None = None) -> Spec:
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise _invalid("<yaml>", f"YAML 解析失败: {exc}") from exc
    if not isinstance(data, dict):
        raise _invalid("<root>", "顶层必须是映射")
    validate(data)
    spec_hash = content_hash(data)
    return Spec(raw=data, spec_hash=spec_hash, source_path=source_path)


def spec_from_dict(data: dict) -> Spec:
    validate(data)
    return Spec(raw=data, spec_hash=content_hash(data))


def validate(data: dict) -> None:
    for key in REQUIRED_TOP_LEVEL:
        if key not in data:
            raise _invalid(key, "缺失字段")

    grid = data["grid"]
    width = _as_int(_require(grid, "width", "grid"), "grid.width")
    height = _as_int(_require(grid, "height", "grid"), "grid.height")
    if width <= 0 or height <= 0:
        raise _invalid("grid", "width/height 必须为正整数")

    quant = data["quantization"]
    elev_step = _as_float(_require(quant, "elevation_step", "quantization"),
                          "quantization.elevation_step")
    dens_step = _as_float(_require(quant, "density_step", "quantization"),
                          "quantization.density_step")
    rounding = _require(quant, "rounding", "quantization")
    if elev_step <= 0.0 or dens_step <= 0.0:
        raise _invalid("quantization", "量化步长必须为正数")
    if rounding not in ("nearest", "down", "up"):
        raise _invalid("quantization.rounding", "只支持 nearest/down/up")

    terrain = data["terrain"]
    types = _require(terrain, "types", "terrain")
    weights = _require(terrain, "weights", "terrain")
    tol_w = _as_float(
        _require(terrain, "weight_normalization_tolerance", "terrain"),
        "terrain.weight_normalization_tolerance",
    )
    if not isinstance(types, list) or not types or len(set(types)) != len(types):
        raise _invalid("terrain.types", "必须为非空且无重复的列表")
    if not isinstance(weights, dict):
        raise _invalid("terrain.weights", "必须为映射")
    for name in types:
        if name not in weights:
            raise _invalid(f"terrain.weights.{name}", "缺少该地形权重")
        w = _as_float(weights[name], f"terrain.weights.{name}")
        if w < 0.0:
            raise _invalid(f"terrain.weights.{name}", "权重不得为负")
    extra = set(weights) - set(types)
    if extra:
        raise _invalid("terrain.weights", f"存在未声明地形的权重: {sorted(extra)}")
    # 双精度权重归一化偏差（显式求和，不使用 fsum，保留可复算的真实舍入）
    weight_sum = 0.0
    for name in sorted(weights):  # 固定求和顺序，结果与字典遍历顺序无关
        weight_sum += float(weights[name])
    deviation = abs(weight_sum - 1.0)
    if deviation > tol_w:
        raise _invalid(
            "terrain.weights",
            f"权重和={weight_sum!r} 与 1 的偏差 {deviation!r} 超过容忍 {tol_w!r}",
        )

    regions = data["regions"]
    if not isinstance(regions, list) or not regions:
        raise _invalid("regions", "必须为非空列表")
    seen_cells: set[tuple[int, int]] = set()
    region_ids: set[str] = set()
    for idx, region in enumerate(regions):
        rpath = f"regions[{idx}]"
        rid = _require(region, "id", rpath)
        if not isinstance(rid, str) or not rid:
            raise _invalid(f"{rpath}.id", "必须为非空字符串")
        if rid in region_ids:
            raise _invalid(f"{rpath}.id", f"区域 id 重复: {rid}")
        region_ids.add(rid)
        cells = _require(region, "cells", rpath)
        if not isinstance(cells, list) or not cells:
            raise _invalid(f"{rpath}.cells", "必须为非空列表")
        for cell in cells:
            if not isinstance(cell, (list, tuple)) or len(cell) != 2:
                raise _invalid(f"{rpath}.cells", "每个坐标必须是 [x, y]")
            x = _as_int(cell[0], f"{rpath}.cells")
            y = _as_int(cell[1], f"{rpath}.cells")
            if not (0 <= x < width and 0 <= y < height):
                raise _invalid(f"{rpath}.cells", f"坐标 [{x}, {y}] 越界")
            if (x, y) in seen_cells:
                raise _invalid(f"{rpath}.cells", f"坐标 [{x}, {y}] 跨区域重叠")
            seen_cells.add((x, y))

    connectivity = data["connectivity"]
    if not isinstance(connectivity, list):
        raise _invalid("connectivity", "必须为列表")
    for idx, conn in enumerate(connectivity):
        cpath = f"connectivity[{idx}]"
        cid = _require(conn, "id", cpath)
        if not isinstance(cid, str):
            raise _invalid(f"{cpath}.id", "必须为字符串")
        rids = _require(conn, "region_ids", cpath)
        if not isinstance(rids, list) or len(rids) < 2:
            raise _invalid(f"{cpath}.region_ids", "至少包含两个区域")
        for rid in rids:
            if rid not in region_ids:
                raise _invalid(f"{cpath}.region_ids", f"未知区域 {rid}")
        ratio = _as_float(_require(conn, "min_channel_width_ratio", cpath),
                          f"{cpath}.min_channel_width_ratio")
        if not (0.0 <= ratio <= 1.0):
            raise _invalid(f"{cpath}.min_channel_width_ratio", "必须落在 [0,1]")
        _as_float(conn.get("tolerance", 0.0), f"{cpath}.tolerance")

    resources = data["resources"]
    count = _require(resources, "count", "resources")
    rmin = _as_int(_require(count, "min", "resources.count"), "resources.count.min")
    rmax = _as_int(_require(count, "max", "resources.count"), "resources.count.max")
    if rmin < 0 or rmax < rmin:
        raise _invalid("resources.count", "要求 0 <= min <= max")
    cap = _as_float(_require(resources, "density_cap", "resources"),
                    "resources.density_cap")
    if cap <= 0.0:
        raise _invalid("resources.density_cap", "必须为正数")
    _as_float(resources.get("tolerance", 0.0), "resources.tolerance")

    rules = data["terrain_rules"]
    passable = _require(rules, "passable", "terrain_rules")
    impassable = _require(rules, "impassable", "terrain_rules")
    if not isinstance(passable, list) or not isinstance(impassable, list):
        raise _invalid("terrain_rules", "passable/impassable 必须为列表")
    for name in passable + impassable:
        if name not in types:
            raise _invalid("terrain_rules", f"未知地形类型 {name}")
    if set(passable) & set(impassable):
        raise _invalid("terrain_rules", "同一地形不能同时可通行与不可通行")
    forbidden = rules.get("forbidden_adjacent", [])
    if not isinstance(forbidden, list):
        raise _invalid("terrain_rules.forbidden_adjacent", "必须为列表")
    for pair in forbidden:
        if not isinstance(pair, list) or len(pair) != 2:
            raise _invalid("terrain_rules.forbidden_adjacent", "每项必须是两个名称")

    points = data["points"]
    spawn = _require(points, "spawn", "points")
    target = _require(points, "target", "points")
    spawn_xy = _validate_point(spawn, "points.spawn", width, height)
    target_xy = _validate_point(target, "points.target", width, height)
    if spawn_xy == target_xy:
        raise _invalid("points.target", "出生点与目标点重合")

    thresholds = data["thresholds"]
    for tname in ("elevation_pass", "cell_density_max", "cell_density_tolerance"):
        if tname not in thresholds:
            raise _invalid(f"thresholds.{tname}", "缺失字段")
        _as_float(thresholds[tname], f"thresholds.{tname}")

    seed = data["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise _invalid("seed", "必须为非负整数")

    anchors = data.get("density_anchors")
    if anchors is not None:
        if not isinstance(anchors, list) or not anchors:
            raise _invalid("density_anchors", "必须为非空列表")
        seen_anchor_cells: set[tuple[int, int]] = set()
        for k, anchor in enumerate(anchors):
            apath = f"density_anchors[{k}]"
            rid = _require(anchor, "region_id", apath)
            if rid not in region_ids:
                raise _invalid(f"{apath}.region_id", f"未知区域 {rid}")
            cell = _require(anchor, "cell", apath)
            xy = _validate_point(cell, f"{apath}.cell", width, height)
            if xy in seen_anchor_cells:
                raise _invalid(f"{apath}.cell", "锚点单元重复")
            seen_anchor_cells.add(xy)
            value = _as_float(_require(anchor, "continuous_density", apath),
                              f"{apath}.continuous_density")
            if not (0.0 <= value <= 1.0):
                raise _invalid(f"{apath}.continuous_density", "必须落在 [0,1]")

    elev_anchors = data.get("elevation_anchors")
    if elev_anchors is not None:
        if not isinstance(elev_anchors, list) or not elev_anchors:
            raise _invalid("elevation_anchors", "必须为非空列表")
        seen_elev: set[tuple[int, int]] = set()
        for k, anchor in enumerate(elev_anchors):
            apath = f"elevation_anchors[{k}]"
            rid = _require(anchor, "region_id", apath)
            if rid not in region_ids:
                raise _invalid(f"{apath}.region_id", f"未知区域 {rid}")
            xy = _validate_point(_require(anchor, "cell", apath),
                                 f"{apath}.cell", width, height)
            if xy in seen_elev:
                raise _invalid(f"{apath}.cell", "锚点单元重复")
            seen_elev.add(xy)
            value = _as_float(_require(anchor, "continuous_elevation", apath),
                              f"{apath}.continuous_elevation")
            if not (0.0 <= value <= 1.0):
                raise _invalid(f"{apath}.continuous_elevation", "必须落在 [0,1]")


def _validate_point(value, path: str, width: int, height: int) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise _invalid(path, "必须是 [x, y]")
    x = _as_int(value[0], path)
    y = _as_int(value[1], path)
    if not (0 <= x < width and 0 <= y < height):
        raise _invalid(path, f"坐标 [{x}, {y}] 越界")
    return x, y


def dump_canonical_spec(spec: Spec) -> bytes:
    """规格的规范化字节形式（账本引用 spec_hash 的复算依据）。"""
    return canonical_dumps(spec.raw)
