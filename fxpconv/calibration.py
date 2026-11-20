"""校准样本集加载与校验。

不合法样本输出 E_SHAPE_MISMATCH 并进入失败清单，且不得进入后续阶段。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from .errors import E_SHAPE_MISMATCH, FailureLedger
from .models import OpDesc


@dataclass(frozen=True)
class CalibrationSample:
    sample_id: str
    activation: tuple[int, ...]
    reference_output: tuple[float, ...]


def load_calibration(
    path: str | Path,
    desc: OpDesc,
) -> tuple[list[CalibrationSample], FailureLedger]:
    ledger = FailureLedger()
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        ledger.add(E_SHAPE_MISMATCH, f"无法解析校准样本集: {exc}", "load", (0, "calib"))
        return [], ledger

    samples_raw = raw.get("samples") if isinstance(raw, dict) else None
    if not isinstance(samples_raw, list) or not samples_raw:
        ledger.add(E_SHAPE_MISMATCH, "校准样本集必须含非空 samples 列表", "validate", (0, "calib"))
        return [], ledger

    act_len = desc.batch_n * desc.in_channels * desc.in_h * desc.in_w
    out_len = desc.total_positions
    valid: list[CalibrationSample] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(samples_raw):
        sample_id = str(item.get("sample_id", f"calib-{index}")) if isinstance(item, dict) else f"calib-{index}"
        sort_prefix = (index, sample_id)
        if sample_id in seen_ids:
            ledger.add(
                E_SHAPE_MISMATCH,
                f"校准样本 ID 重复: {sample_id}",
                "validate",
                (*sort_prefix, "dup"),
            )
            continue
        if not isinstance(item, dict):
            ledger.add(E_SHAPE_MISMATCH, "校准样本必须是对象", "validate", (*sort_prefix, "type"))
            continue
        activation = item.get("activation")
        reference = item.get("reference_output")
        if (
            not isinstance(activation, list)
            or len(activation) != act_len
            or not all(isinstance(v, int) and not isinstance(v, bool) for v in activation)
        ):
            ledger.add(
                E_SHAPE_MISMATCH,
                f"校准样本 {sample_id} 激活长度/类型非法，期望 {act_len} 个整数",
                "validate",
                (*sort_prefix, "activation"),
            )
            continue
        if (
            not isinstance(reference, list)
            or len(reference) != out_len
            or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in reference)
        ):
            ledger.add(
                E_SHAPE_MISMATCH,
                f"校准样本 {sample_id} 参考输出长度/类型非法，期望 {out_len} 个有限浮点",
                "validate",
                (*sort_prefix, "reference"),
            )
            continue
        seen_ids.add(sample_id)
        valid.append(
            CalibrationSample(
                sample_id=sample_id,
                activation=tuple(int(v) for v in activation),
                reference_output=tuple(float(v) for v in reference),
            )
        )
    return valid, ledger
