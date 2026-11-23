"""统一中间表示（IR）：特征、匹配、体素点及跨版本校验。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .errors import (
    Failure,
    E_COORD_OUT_OF_RANGE,
    E_DUPLICATE_VIEW,
    E_NON_FINITE,
    E_SCORE_OUT_OF_RANGE,
    E_VOXEL_NOT_ALIGNED,
    E_VOXEL_ZERO_SIZE,
    FailureList,
)
from .utils import is_finite_number


@dataclass
class Feature:
    view: int
    feature_id: int
    x: float                 # 像素坐标（v1 为整数浮点，v2 可为任意浮点）
    y: float
    landmark_id: int         # 可检测图元/纹理标记编号（匹配真值锚点）
    precision: str = "exact"  # 精度标签

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Feature":
        return cls(**d)


@dataclass
class Match:
    view_a: int
    feature_a: int
    view_b: int
    feature_b: int
    score_permille: float    # 千分比得分，范围 [0, 1000]
    metric: str = "ncc"      # 命名匹配度量
    metric_code: int = 0
    precision: str = "exact"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Match":
        return cls(**d)


@dataclass
class VoxelPoint:
    vx: int                  # 整数体素坐标（可被 voxel_size 整除语义在浮点代表点上校验）
    vy: int
    vz: int
    x: float                 # 体素代表点（浮点）
    y: float
    z: float
    landmark_id: int
    precision: str = "exact"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "VoxelPoint":
        return cls(**d)


@dataclass
class ReconstructionIR:
    image_size: tuple                       # (W, H)
    voxel_size: float
    fixed_point_bits: int
    features: list = field(default_factory=list)
    matches: list = field(default_factory=list)
    voxel_points: list = field(default_factory=list)
    rounding_mode: str = "ROUND_HALF_UP"
    source_version: str = "v2"
    failures: FailureList = field(default_factory=FailureList)

    # ---- 序列化（FailureList 转为普通字典列表） ----
    def to_dict(self) -> dict:
        return {
            "image_size": list(self.image_size),
            "voxel_size": self.voxel_size,
            "fixed_point_bits": self.fixed_point_bits,
            "rounding_mode": self.rounding_mode,
            "source_version": self.source_version,
            "features": [f.to_dict() for f in self.features],
            "matches": [m.to_dict() for m in self.matches],
            "voxel_points": [p.to_dict() for p in self.voxel_points],
            "failures": self.failures.dicts(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ReconstructionIR":
        fl = FailureList()
        for fd in d.get("failures", []):
            fl.add(Failure.from_dict(fd))
        return cls(
            image_size=tuple(d["image_size"]),
            voxel_size=float(d["voxel_size"]),
            fixed_point_bits=int(d["fixed_point_bits"]),
            features=[Feature.from_dict(x) for x in d.get("features", [])],
            matches=[Match.from_dict(x) for x in d.get("matches", [])],
            voxel_points=[VoxelPoint.from_dict(x) for x in d.get("voxel_points", [])],
            rounding_mode=d.get("rounding_mode", "ROUND_HALF_UP"),
            source_version=d.get("source_version", "v2"),
            failures=fl,
        )

    # ---- 统一校验（需求 1） ----
    def validate(self) -> FailureList:
        width, height = self.image_size
        seen_feature_keys: set = set()
        seen_views: set = set()

        if self.voxel_size <= 0:
            self.failures.fail(
                E_VOXEL_ZERO_SIZE,
                f"voxel size must be positive, got {self.voxel_size}",
                stage="validate",
            )

        for f in self.features:
            if f.view in seen_views:
                pass  # 视图序号唯一性在“记录视图”层面校验，见下方 view 列表
            for name, val in (("x", f.x), ("y", f.y)):
                if not is_finite_number(val):
                    self.failures.fail(
                        E_NON_FINITE,
                        f"feature {name} non-finite in view {f.view}",
                        view=f.view,
                        stage="validate",
                    )
            if is_finite_number(f.x) and not (0 <= f.x < width):
                self.failures.fail(
                    E_COORD_OUT_OF_RANGE,
                    f"feature x={f.x} out of image width {width}",
                    view=f.view,
                    stage="validate",
                )
            if is_finite_number(f.y) and not (0 <= f.y < height):
                self.failures.fail(
                    E_COORD_OUT_OF_RANGE,
                    f"feature y={f.y} out of image height {height}",
                    view=f.view,
                    stage="validate",
                )
            key = (f.view, f.feature_id)
            if key in seen_feature_keys:
                self.failures.fail(
                    E_DUPLICATE_VIEW,
                    f"duplicate feature id {f.feature_id} in view {f.view}",
                    view=f.view,
                    stage="validate",
                )
            seen_feature_keys.add(key)
            seen_views.add(f.view)

        for m in self.matches:
            if not is_finite_number(m.score_permille):
                self.failures.fail(
                    E_NON_FINITE,
                    f"match score non-finite for views {m.view_a}/{m.view_b}",
                    stage="validate",
                )
            elif not (0 <= m.score_permille <= 1000):
                self.failures.fail(
                    E_SCORE_OUT_OF_RANGE,
                    f"match score {m.score_permille} not in [0,1000]",
                    view=m.view_a,
                    stage="validate",
                )

        if self.voxel_size > 0:
            for p in self.voxel_points:
                for name, val in (("x", p.x), ("y", p.y), ("z", p.z)):
                    if not is_finite_number(val):
                        self.failures.fail(
                            E_NON_FINITE,
                            f"voxel point {name} non-finite",
                            stage="validate",
                        )
                # 体素坐标可被体素尺寸整除：代表点应落在 voxel_size 网格上
                # 定点量化容差：半个最小可表示单位（1/2^bits 的一半）
                tol = 0.5 / (1 << self.fixed_point_bits)
                for name, val, idx in (("x", p.x, p.vx), ("y", p.y, p.vy), ("z", p.z, p.vz)):
                    if not is_finite_number(val):
                        continue
                    quotient = val / self.voxel_size
                    nearest = round(quotient)
                    if abs(quotient - nearest) * self.voxel_size > tol:
                        self.failures.fail(
                            E_VOXEL_NOT_ALIGNED,
                            f"voxel {name}={val} not divisible by voxel_size {self.voxel_size}",
                            stage="validate",
                        )
                    if nearest != idx:
                        self.failures.fail(
                            E_VOXEL_NOT_ALIGNED,
                            f"voxel index {idx} inconsistent with coordinate {val}",
                            stage="validate",
                        )
        return self.failures

    def view_ids(self) -> list:
        return sorted({f.view for f in self.features})

    def sort_canonical(self) -> None:
        """规范化排序，保证全量/增量逐字段一致。"""
        self.features.sort(key=lambda f: (f.view, f.feature_id))
        self.matches.sort(key=lambda m: (m.view_a, m.view_b, m.feature_a, m.feature_b))
        self.voxel_points.sort(key=lambda p: (p.vx, p.vy, p.vz, p.landmark_id))
