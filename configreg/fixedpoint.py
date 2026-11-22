"""统一精度口径下的整数定点量化、派生求值与误差界（需求 3）。

口径声明
--------
* 缩放因子 ``scale = 10 ** precision``，``precision`` 必须是非负整数。
* 叶子实数 ``x`` 量化为 ``q = trunc_toward_zero(x * scale)``，即向零截断。
* 加法为饱和整数加；乘法先做整数乘再向零截断除以 ``scale``，模拟
  定点 Q0.p 乘法，全部乘加以 Python 任意精度整数完成。
* 饱和上下限为显式常量 :data:`SAT_MIN` / :data:`SAT_MAX`，任何中间结果
  越过界限即钳制到界限并在求值结果中置 ``saturated=True``。
* 单值量化误差满足 ``|x - q/scale| <= 1/scale``（记为 1 ulp）。
"""
from __future__ import annotations

from dataclasses import dataclass

from .manifest import ItemSpec, Manifest

SAT_MAX = 2**63 - 1
SAT_MIN = -(2**63)


@dataclass(frozen=True)
class FixedPointSpec:
    precision: int

    @property
    def scale(self) -> int:
        return 10 ** self.precision

    @property
    def ulp(self) -> float:
        return 1.0 / self.scale


def make_spec(precision: int) -> FixedPointSpec:
    if not isinstance(precision, int) or precision < 0 or precision > 12:
        raise ValueError("定点精度 precision 须为 0..12 的非负整数")
    return FixedPointSpec(precision)


def quantize(value: float, spec: FixedPointSpec) -> int:
    """向零截断的定点量化，结果做饱和钳制。"""
    scaled = value * spec.scale
    truncated = int(scaled)  # Python int() 对浮点即向零截断
    return _saturate(truncated)


def dequantize(q: int, spec: FixedPointSpec) -> float:
    return q / spec.scale


def _saturate(q: int) -> int:
    if q > SAT_MAX:
        return SAT_MAX
    if q < SAT_MIN:
        return SAT_MIN
    return q


def sat_add(left: int, right: int) -> int:
    return _saturate(left + right)


def sat_mul_q(left: int, right: int, spec: FixedPointSpec) -> int:
    """定点乘：先整数乘，再向零截断除以缩放因子。"""
    return _saturate((left * right) // spec.scale)


@dataclass(frozen=True)
class QuantizedManifest:
    spec: FixedPointSpec
    item_q: dict[str, int]
    coeff_q: dict[str, dict[str, int]]
    saturated: bool


def quantize_manifest(manifest: Manifest, spec: FixedPointSpec) -> QuantizedManifest:
    """按稳定拓扑序以纯整数乘加求值全部配置项的定点值。"""
    item_q: dict[str, int] = {}
    coeff_q: dict[str, dict[str, int]] = {}
    saturated = False
    for key in manifest.topological_order():
        item = manifest.items[key]
        if not item.derived:
            q = quantize(item.value or 0.0, spec)
            item_q[key] = q
            saturated |= q in (SAT_MIN, SAT_MAX)
            continue
        local_coeff: dict[str, int] = {}
        accum = quantize(item.const, spec)
        for source in sorted(item.coeffs):
            coefficient_q = quantize(item.coeffs[source], spec)
            local_coeff[source] = coefficient_q
            term = sat_mul_q(coefficient_q, item_q[source], spec)
            accum = sat_add(accum, term)
        item_q[key] = accum
        coeff_q[key] = local_coeff
        saturated |= accum in (SAT_MIN, SAT_MAX)
    return QuantizedManifest(spec, item_q, coeff_q, saturated)


def real_values(manifest: Manifest) -> dict[str, float]:
    """以实数口径求值配置项，用于误差界合成与节点下发。"""
    values: dict[str, float] = {}
    for key in manifest.topological_order():
        item = manifest.items[key]
        if not item.derived:
            values[key] = float(item.value or 0.0)
            continue
        total = item.const
        for source, coefficient in item.coeffs.items():
            total += coefficient * values[source]
        values[key] = total
    return values


def leaf_error_bound(spec: FixedPointSpec) -> float:
    """叶子值量化误差上界（向零截断，单值 <= 1 ulp）。"""
    return spec.ulp


def propagated_error(item: ItemSpec, upstream: dict[str, float], spec: FixedPointSpec) -> float:
    r"""派生项 ``y = const + sum a_i x_i`` 的定点误差上界。

    截断误差沿派生链按系数绝对值合成：
    ``E(y) <= ulp(1 + sum|a_i|) + sum |a_i| E(x_i)``，
    饱和发生时误差界退化为无界（``inf``），对应结论标记受饱和影响。
    """
    if not item.derived:
        return leaf_error_bound(spec)
    bound = spec.ulp * (1.0 + sum(abs(coef) for coef in item.coeffs.values()))
    for source, coefficient in item.coeffs.items():
        bound += abs(coefficient) * upstream[source]
    return bound


def error_bounds(manifest: Manifest, spec: FixedPointSpec, quantized: QuantizedManifest) -> dict[str, float]:
    bounds: dict[str, float] = {}
    for key in manifest.topological_order():
        item = manifest.items[key]
        if not item.derived:
            bounds[key] = leaf_error_bound(spec)
        elif quantized.saturated:
            bounds[key] = float("inf")
        else:
            bounds[key] = propagated_error(item, bounds, spec)
    return bounds


def difference_error_bound(
    item: ItemSpec,
    base_manifest: Manifest,
    cand_manifest: Manifest,
    base_bounds: dict[str, float],
    cand_bounds: dict[str, float],
) -> float:
    r"""两版同一配置项取值差异的不可区分阈值。

    对派生项按系数绝对值合成两侧误差，并覆盖系数变更带来的额外
    截断贡献：``B = E_b(y) + E_c(y)``；叶子项即两侧各 1 ulp 之和。
    任一版发生饱和时返回 ``inf``，表示该差异无法在定点口径下区分。
    """
    if item.derived and (
        base_bounds.get(item.key) == float("inf") or cand_bounds.get(item.key) == float("inf")
    ):
        return float("inf")
    bound = base_bounds[item.key] + cand_bounds[item.key]
    if item.derived:
        base_item = base_manifest.items[item.key]
        cand_item = cand_manifest.items[item.key]
        for source in sorted(set(base_item.coeffs) | set(cand_item.coeffs)):
            base_coef = abs(base_item.coeffs.get(source, 0.0))
            cand_coef = abs(cand_item.coeffs.get(source, 0.0))
            bound += abs(cand_coef - base_coef) * cand_bounds.get(source, 0.0)
    return bound
