"""按分组键的整数定点聚合与确定性合并。

每个连接维护自己的局部累加器；合并阶段按分组键（编码字节序）稳定排序，
累加顺序固定为“连接建立顺序（conn_id 字典序）+ 批次顺序”，
结果与线程执行交错无关。

定点规则：
- 浮点输入按会话 scale 向零截断量化为 int64。
- 求和使用 int64 饱和加法，溢出钳制到 INT64_MIN/INT64_MAX。
- count(*) 为精确整数计数（不参与饱和）。
- avg = sat_div(sum, avg_inputs)，向零截断。
"""

from dataclasses import dataclass, field

from .fixedpoint import quantize, sat_add, sat_div, value_error_bound, quantization_error_bound


@dataclass
class KeyAccumulator:
    count: int = 0
    sums: dict = field(default_factory=dict)
    avg_inputs: dict = field(default_factory=dict)


class LocalAggregator:
    def __init__(self, conn_id, aggregates, scale):
        self.conn_id = conn_id
        self.aggregates = aggregates
        self.scale = scale
        self.groups = {}

    def _acc(self, key_encoded):
        acc = self.groups.get(key_encoded)
        if acc is None:
            aliases = [alias for _, _, alias in self.aggregates]
            acc = KeyAccumulator(
                sums={a: 0 for a in aliases},
                avg_inputs={a: 0 for a in aliases},
            )
            self.groups[key_encoded] = acc
        return acc

    def add_row(self, key_encoded, values):
        acc = self._acc(key_encoded)
        for (func, column, alias), raw in zip(self.aggregates, values):
            if func == "count":
                acc.count += 1
                continue
            if raw is None:
                continue
            quantized = quantize(float(raw), self.scale)
            acc.sums[alias] = sat_add(acc.sums[alias], quantized)
            if func == "avg":
                acc.avg_inputs[alias] += 1

    def to_ledger(self):
        return {
            "conn_id": self.conn_id,
            "groups": [
                {
                    "key_encoded_hex": _key_bytes(key).hex(),
                    "count": acc.count,
                    "sums": dict(sorted(acc.sums.items())),
                    "avg_inputs": dict(sorted(acc.avg_inputs.items())),
                }
                for key, acc in sorted(self.groups.items(), key=lambda kv: _key_bytes(kv[0]))
            ],
        }


def _key_bytes(key_encoded):
    return b"".join(bytes((tag,)) + payload for tag, payload in key_encoded)


def merge_aggregators(aggregators, aggregates, scale):
    """确定性合并：分组键按字节序排序，组内按 conn_id 字典序累加。"""
    ordered = sorted(aggregators, key=lambda a: a.conn_id)
    merged = {}
    aliases = [alias for _, _, alias in aggregates]
    for agg in ordered:
        for key, acc in agg.groups.items():
            dst = merged.get(key)
            if dst is None:
                dst = KeyAccumulator(
                    sums={a: 0 for a in aliases},
                    avg_inputs={a: 0 for a in aliases},
                )
                merged[key] = dst
            dst.count += acc.count
            for alias in aliases:
                dst.sums[alias] = sat_add(dst.sums[alias], acc.sums.get(alias, 0))
                dst.avg_inputs[alias] += acc.avg_inputs.get(alias, 0)

    results = []
    for key in sorted(merged, key=_key_bytes):
        acc = merged[key]
        row = {"key_encoded_hex": _key_bytes(key).hex(),
               "aggregates": {}, "error_bounds": {}}
        for func, column, alias in aggregates:
            if func == "count":
                row["aggregates"][alias] = {"fixed_point": acc.count, "kind": "count"}
                row["error_bounds"][alias] = "0"
            elif func == "sum":
                row["aggregates"][alias] = {
                    "fixed_point": acc.sums[alias], "kind": "sum", "scale": scale,
                }
                row["error_bounds"][alias] = quantization_error_bound(acc.count, scale)
            else:
                inputs = acc.avg_inputs[alias]
                avg = sat_div(acc.sums[alias], inputs)
                row["aggregates"][alias] = {
                    "fixed_point": avg, "kind": "avg", "scale": scale,
                    "count_inputs": inputs,
                }
                row["error_bounds"][alias] = quantization_error_bound(inputs, scale)
        results.append(row)
    return {
        "truncation": "toward_zero",
        "overflow": "saturate_int64",
        "scale": scale,
        "single_value_quantization_error": value_error_bound(scale),
        "rows": results,
    }
