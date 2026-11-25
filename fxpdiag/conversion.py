"""v1 <-> v2 双向转换与降级信息旁路。

v1 升 v2：千分比/偏置码精确换算为浮点，生成精度标签，位宽按码表映射命名方案。
v2 降 v1：浮点四舍五入还原千分比，截断残差、精度标签与被丢弃的命名舍入模式
写入旁路元数据；再次升回时优先从旁路还原降级前可达信息。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from .models import (
    BIT_WIDTH_TO_NO_BIAS_SCHEME,
    BIT_WIDTH_TO_SCHEME,
    DEFAULT_ROUNDING,
    NAMED_ROUNDING_MODES,
    Scheme,
    V1WeightRow,
    V2WeightRow,
)

SIDECAR_SUFFIX = ".sidecar.json"


def precision_tag(bit_width: int) -> str:
    return f"permille-exact-bw{bit_width}"


def v1_to_v2_row(row: V1WeightRow, use_bias: bool = True) -> V2WeightRow:
    scheme = (
        BIT_WIDTH_TO_SCHEME[row.bit_width]
        if use_bias
        else BIT_WIDTH_TO_NO_BIAS_SCHEME[row.bit_width]
    )
    return V2WeightRow(
        layer=row.layer,
        row_index=row.row_index,
        scale=row.scale_permille / 1000.0,
        scheme=scheme.value,
        rounding_mode=DEFAULT_ROUNDING,
    )


def round_half_up_even_negative(x: float) -> int:
    """四舍五入（远离零方向的半值进位），用于千分比还原。"""

    import math

    return int(math.copysign(math.floor(abs(x) + 0.5), x))


@dataclass
class DowngradeSidecar:
    """v2 降 v1 的旁路元数据（逐行）。"""

    layer: str
    row_index: int
    original_scale: float
    restored_scale_permille: int
    scale_residual: float
    original_scheme: str
    precision_tag: str
    dropped_rounding_mode: str

    def to_dict(self) -> dict:
        return {
            "layer": self.layer,
            "row_index": self.row_index,
            "original_scale": self.original_scale,
            "restored_scale_permille": self.restored_scale_permille,
            "scale_residual": self.scale_residual,
            "original_scheme": self.original_scheme,
            "precision_tag": self.precision_tag,
            "dropped_rounding_mode": self.dropped_rounding_mode,
        }


@dataclass
class ConversionRecord:
    layer: str
    row_index: int
    direction: str
    scale_before: float
    scale_after: float
    loss_abs: float
    precision_tag: str
    sidecar_used: bool = False

    def to_dict(self) -> dict:
        return {
            "layer": self.layer,
            "row_index": self.row_index,
            "direction": self.direction,
            "scale_before": self.scale_before,
            "scale_after": self.scale_after,
            "loss_abs": self.loss_abs,
            "precision_tag": self.precision_tag,
            "sidecar_used": self.sidecar_used,
        }


def v2_to_v1_row(row: V2WeightRow) -> tuple[V1WeightRow, DowngradeSidecar]:
    scheme = Scheme(row.scheme)
    permille = round_half_up_even_negative(row.scale * 1000.0)
    residual = row.scale - permille / 1000.0
    v1_row = V1WeightRow(
        layer=row.layer,
        row_index=row.row_index,
        scale_permille=permille,
        bias_code=0,  # v2 不携带整数偏置码，降级时置零并在旁路标注
        bit_width=scheme.bit_width,
    )
    sidecar = DowngradeSidecar(
        layer=row.layer,
        row_index=row.row_index,
        original_scale=row.scale,
        restored_scale_permille=permille,
        scale_residual=residual,
        original_scheme=row.scheme,
        precision_tag=precision_tag(scheme.bit_width),
        dropped_rounding_mode=row.rounding_mode,
    )
    return v1_row, sidecar


def v1_to_v2_with_sidecar(
    v1_row: V1WeightRow,
    sidecar: DowngradeSidecar | None,
) -> tuple[V2WeightRow, ConversionRecord]:
    """再次升回：有旁路时还原降级前可达信息（原始浮点、舍入模式）。"""

    if sidecar is not None:
        scheme = Scheme(sidecar.original_scheme)
        v2_row = V2WeightRow(
            layer=v1_row.layer,
            row_index=v1_row.row_index,
            scale=sidecar.original_scale,
            scheme=scheme.value,
            rounding_mode=sidecar.dropped_rounding_mode
            if sidecar.dropped_rounding_mode in NAMED_ROUNDING_MODES
            else DEFAULT_ROUNDING,
        )
        record = ConversionRecord(
            layer=v1_row.layer,
            row_index=v1_row.row_index,
            direction="v1->v2-restore",
            scale_before=v1_row.scale_permille / 1000.0,
            scale_after=v2_row.scale,
            loss_abs=abs(v2_row.scale - v1_row.scale_permille / 1000.0),
            precision_tag=sidecar.precision_tag,
            sidecar_used=True,
        )
        return v2_row, record

    v2_row = v1_to_v2_row(v1_row, use_bias=True)
    record = ConversionRecord(
        layer=v1_row.layer,
        row_index=v1_row.row_index,
        direction="v1->v2",
        scale_before=v1_row.scale_permille / 1000.0,
        scale_after=v2_row.scale,
        loss_abs=0.0,
        precision_tag=precision_tag(v1_row.bit_width),
    )
    return v2_row, record


def write_sidecar(path: str, sidecars: list[DowngradeSidecar]) -> None:
    payload = {
        "format": "v2->v1-sidecar",
        "count": len(sidecars),
        "rows": [item.to_dict() for item in sidecars],
    }
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def read_sidecar(path: str) -> dict[tuple[str, int], DowngradeSidecar]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    result: dict[tuple[str, int], DowngradeSidecar] = {}
    for item in payload["rows"]:
        result[(item["layer"], item["row_index"])] = DowngradeSidecar(**item)
    return result


def layer_roundtrip_loss(records: list[ConversionRecord]) -> dict[str, float]:
    """逐层汇总往返损耗（缩放因子绝对偏差之和）。"""

    loss: dict[str, float] = {}
    for record in records:
        loss[record.layer] = loss.get(record.layer, 0.0) + record.loss_abs
    return loss


def sidecar_fingerprint(sidecar: DowngradeSidecar) -> str:
    digest = hashlib.sha256(
        json.dumps(sidecar.to_dict(), sort_keys=True).encode("utf-8")
    ).hexdigest()
    return digest[:16]
