"""重建流水线：特征 -> 匹配 -> 三角化 -> 深度图 -> 点云。

同一套函数同时服务于 float32 主路径与 float64 参考路径，
仅通过 ``precision`` 参数切换 NumPy dtype，保证可对比性。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .config import RebuildConfig
from .errors import BizCode, FailureRecord, WorkbenchError
from .features import Feature, MatchRecord, extract_features, match_views
from .geometry import (
    CloudPoint,
    DepthMap,
    TriPoint,
    build_depth_map,
    merge_point_cloud,
    triangulate_pair,
)
from .loader import Camera, ViewData


@dataclass
class PairResult:
    view_a: int
    view_b: int
    matches: List[MatchRecord]
    tri_points: List[TriPoint]
    depth_a: DepthMap
    depth_b: DepthMap
    match_key: str = ""
    depth_key_a: str = ""
    depth_key_b: str = ""
    boundary_match: bool = False
    below_inliers: bool = False


@dataclass
class PipelineResult:
    precision: str
    features: Dict[int, List[Feature]] = field(default_factory=dict)
    pairs: List[PairResult] = field(default_factory=list)
    cloud: List[CloudPoint] = field(default_factory=list)
    failures: List[FailureRecord] = field(default_factory=list)
    budget_deductions: List[Dict[str, Any]] = field(default_factory=list)
    pending: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def features_flat(self) -> List[Feature]:
        out: List[Feature] = []
        for view in sorted(self.features):
            out.extend(self.features[view])
        return out

    @property
    def matches_flat(self) -> List[MatchRecord]:
        out: List[MatchRecord] = []
        for pair in self.pairs:
            out.extend(pair.matches)
        return out

    @property
    def tripoints_flat(self) -> List[TriPoint]:
        out: List[TriPoint] = []
        for pair in self.pairs:
            out.extend(pair.tri_points)
        return out

    @property
    def depth_maps(self) -> List[DepthMap]:
        out: List[DepthMap] = []
        for pair in self.pairs:
            out.extend([pair.depth_a, pair.depth_b])
        return out


def _injection_marker(camera: Camera, sample_dir_hint: Optional[str] = None
                      ) -> Optional[Dict[str, Any]]:
    # 注入标记通过 camera 对象外的原始 JSON 读取，这里由引擎在调用前注入；
    # pipeline 只识别 ViewData 上挂载的标记字典。
    marker = getattr(camera, "injection_marker", None)
    return marker


def extract_view_features(view_data: ViewData, config: RebuildConfig,
                          precision: str) -> List[Feature]:
    marker = getattr(view_data.camera, "injection_marker", None)
    nonfinite = bool(marker and marker.get("type") == "nonfinite_descriptor")
    return extract_features(
        view_data.gray, view_data.view, config,
        precision=precision, nonfinite_injection=nonfinite,
    )


def run_pair(view_a: ViewData, view_b: ViewData,
             features_a: Sequence[Feature], features_b: Sequence[Feature],
             config: RebuildConfig, precision: str
             ) -> PairResult:
    """单个视图对：匹配 -> 最小内点门限 -> 三角化 -> 双视图深度图。"""
    records, boundary_match = match_views(
        features_a, features_b, view_a.gray, view_b.gray,
        view_a.view, view_b.view, config, precision=precision,
    )
    below = len(records) < config.min_inliers
    if below:
        raise WorkbenchError(
            f"视图对 ({view_a.view},{view_b.view}) 匹配内点 "
            f"{len(records)} 少于门限 {config.min_inliers}",
            view=view_b.view, stage="match", code=BizCode.E_MIN_INLIERS,
            detail={"inliers": len(records),
                    "min_inliers": config.min_inliers},
        )

    marker_b = getattr(view_b.camera, "injection_marker", None)
    override = None
    if marker_b and marker_b.get("type") == "disparity_out_of_range":
        override = float(marker_b.get("forced_disparity", 0.0))
    tri_points = triangulate_pair(
        records, features_a, features_b, view_a.camera, view_b.camera,
        config, precision=precision, disparity_override=override,
    )
    # 视差越界注入：全部点被门限丢弃时给出可追溯业务码
    if override is not None and not tri_points:
        raise WorkbenchError(
            f"视图 {view_b.view} 视差越界，无点通过最小视差门限",
            view=view_b.view, stage="triangulation",
            code=BizCode.E_MIN_DISPARITY,
            detail={"forced_disparity": override,
                    "min_disparity": config.min_disparity},
        )

    depth_a = build_depth_map(
        tri_points, view_a.camera, config, precision=precision
    )
    depth_b = build_depth_map(
        tri_points, view_b.camera, config, precision=precision
    )
    return PairResult(
        view_a=view_a.view, view_b=view_b.view, matches=records,
        tri_points=tri_points, depth_a=depth_a, depth_b=depth_b,
        boundary_match=boundary_match,
    )


def assemble_cloud(pairs: Sequence[PairResult], config: RebuildConfig,
                   precision: str) -> List[CloudPoint]:
    all_points: List[TriPoint] = []
    for pair in sorted(pairs, key=lambda p: (p.view_a, p.view_b)):
        all_points.extend(pair.tri_points)
    return merge_point_cloud(all_points, config, precision=precision)


def stable_task_key(view: int, stage: str) -> str:
    """由视图序号与阶段名派生的稳定排序键。"""
    return f"{view:04d}:{stage}"
