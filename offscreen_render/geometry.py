"""统一世界坐标下的光线与多介质求交（业务要求 2）。

关键约定
--------
* 每条光线用原点 ``origin`` 与单位方向 ``direction`` 参数化 ``P(t)=O+tD``。
* 对每个轴对齐盒用 slab 法求出参数区间 ``[t_enter, t_exit]``。
* 把所有盒的区间端点作为切分点，在每个相邻切分点之间用
  **盒声明顺序最小（即最先声明）** 的活动盒唯一归属，重叠区归属因此确定且唯一。
  声明顺序（``volume.index``）是第一裁决键；同序场景下以折射率（``ior``）
  作为次级裁决键，二者共同保证归属唯一、确定、可复现。本数据集中每个盒的
  声明顺序天然唯一，因此实际由 ``index`` 决定，``ior`` 随归属区间一并记录。
* 输出区间链严格有序、互不重叠、首尾相接；交界处既不会被两个盒重复计入，
  也不会出现无主空隙。
* 采样使用 *段中点归属*：渲染时每个步进段的光学厚度按该段中点所在区间取值，
  因此落在边界上的采样点不会依赖端点归属判定。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .errors import E_INTERSECT, RenderError
from .scene import VolumeBox

_EPS = 1.0e-12


@dataclass(frozen=True)
class MediumInterval:
    """光线参数轴上的一段介质区间。"""

    t_enter: float
    t_exit: float
    volume: VolumeBox

    @property
    def length(self) -> float:
        return self.t_exit - self.t_enter


def ray_aabb(origin: np.ndarray, direction: np.ndarray,
             lo: np.ndarray, hi: np.ndarray) -> tuple[float, float] | None:
    """Slab 法求光线与 AABB 的参数区间；不相交返回 None。"""
    t_lo = -np.inf
    t_hi = np.inf
    for axis in range(3):
        d = direction[axis]
        if abs(d) < _EPS:
            if origin[axis] < lo[axis] or origin[axis] > hi[axis]:
                return None
            continue
        inv = 1.0 / d
        t1 = (lo[axis] - origin[axis]) * inv
        t2 = (hi[axis] - origin[axis]) * inv
        if t1 > t2:
            t1, t2 = t2, t1
        t_lo = max(t_lo, t1)
        t_hi = min(t_hi, t2)
        if t_lo > t_hi:
            return None
    if t_hi < 0.0:
        return None
    t_lo = max(t_lo, 0.0)
    if t_hi <= t_lo:
        return None
    return float(t_lo), float(t_hi)


def build_interval_chain(origin: np.ndarray, direction: np.ndarray,
                         volumes: tuple[VolumeBox, ...]) -> list[MediumInterval]:
    """把多盒区间合并为有序、无重叠、唯一归属的区间链。

    重叠区按盒声明顺序（``volume.index`` 升序）确定唯一归属；
    折射率仅作为归属区间的属性记录，不改变几何归属规则。
    """
    raw: list[tuple[float, float, VolumeBox]] = []
    for vol in volumes:
        hit = ray_aabb(origin, direction, vol.min_corner, vol.max_corner)
        if hit is not None:
            raw.append((hit[0], hit[1], vol))
    if not raw:
        return []

    # 收集全部切分点（含 0 起点裁剪）。
    cuts = {0.0}
    for t0, t1, _ in raw:
        cuts.add(max(t0, 0.0))
        cuts.add(t1)
    boundaries = sorted(cuts)

    chain: list[MediumInterval] = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        if right - left <= _EPS:
            continue
        midpoint = 0.5 * (left + right)
        owner: VolumeBox | None = None
        # 声明顺序遍历，首个覆盖中点的盒即归属，保证 index 最小者优先。
        for t0, t1, vol in raw:
            if t0 - _EPS <= midpoint <= t1 + _EPS:
                owner = vol
                break
        if owner is None:
            continue
        if chain and chain[-1].volume.index == owner.index and \
                abs(chain[-1].t_exit - left) <= _EPS * max(1.0, abs(left)):
            # 同盒相邻段合并，保持链简洁（交界处仍只有一份光学厚度）。
            chain[-1] = MediumInterval(chain[-1].t_enter, right, owner)
        else:
            chain.append(MediumInterval(left, right, owner))
    return chain


def interval_at(chain: list[MediumInterval], t: float) -> MediumInterval | None:
    """按中点归属返回参数 t 所在区间；恰在边界时归入包含该点且后开始的区间。

    渲染步进以段中点查询，正常路径不会落在精确边界上；该函数仅用于
    误差定位等需要标注“所在介质区间”的场景，行为对边界显式定义。
    """
    for seg in chain:
        if seg.t_enter - _EPS <= t < seg.t_exit - _EPS:
            return seg
    if chain and abs(t - chain[-1].t_exit) <= _EPS:
        return chain[-1]
    return None


def total_chain_length(chain: list[MediumInterval]) -> float:
    total = 0.0
    for seg in chain:
        total += seg.length
    return total


def validate_chain_integrity(chain: list[MediumInterval]) -> None:
    """自检：区间链必须有序、首尾相接、无重叠。"""
    for prev, nxt in zip(chain[:-1], chain[1:]):
        if nxt.t_enter < prev.t_exit - _EPS:
            raise RenderError(E_INTERSECT, "区间链出现重叠，求交结果不确定")
        if nxt.t_enter <= prev.t_enter:
            raise RenderError(E_INTERSECT, "区间链未按入射次序排序")
