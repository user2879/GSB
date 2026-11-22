"""块布局变更下的累加顺序可观测性探针。

利用定点累加器饱和（非结合运算）构造样本：同一 ID 序列、相同行内容，
仅改变行所属块布局（从而改变分块部分和的归并顺序），定点结果可能不同；
float64 参考路径不受饱和影响，两种布局结果保持一致。
"""
import numpy as np

from .quantize import accumulator_bounds, quantize_row, saturating_add, dequantize
from .storage import Table, Row


def build_probe_tables(config):
    """返回 (表A, 表B, 请求ID序列)：两表行内容一致，块布局不同。"""
    dim = config["dim"]
    level = config["quant_level"]
    fbits = config["fractional_bits"]
    headroom = config["accumulator_headroom_bits"]
    acc_min, acc_max = accumulator_bounds(level, headroom)
    step = np.float64(2.0) ** (-fbits)

    # 8 行：4 个接近行量化上限的正行 + 4 个半幅负行。
    # 布局 A：正行集中在同一块，块内部分和触及累加器饱和；
    # 布局 B：正负交错分块，块内均不饱和。两表 ID 序列完全相同，
    # 因此全局顺序累加一致（fixedpoint_max_abs_diff=0），
    # 但分块部分和归并结果不同（block_merge_max_abs_diff>0），
    # float64 参考路径不受饱和影响，两布局完全一致。
    qmax_pos = float((1 << (level - 1)) - 1)
    pos = qmax_pos * 0.9 * step
    neg = -qmax_pos * 0.5 * step
    pos_ids = [9001, 9002, 9003, 9004]
    neg_ids = [9005, 9006, 9007, 9008]
    vectors = {}
    for rid in pos_ids:
        vectors[rid] = [pos] + [0.0] * (dim - 1)
    for rid in neg_ids:
        vectors[rid] = [neg] + [0.0] * (dim - 1)
    sequence = pos_ids + neg_ids

    def make_table(layout):
        table = Table(dim)
        for block_id, members in layout.items():
            slot = table.block_offsets.setdefault(block_id, {})
            for offset, rid in enumerate(members):
                slot[offset] = rid
                table.rows[rid] = Row(rid, block_id, offset, np.asarray(vectors[rid], np.float64))
        return table

    layout_a = {1: pos_ids, 2: neg_ids}
    layout_b = {
        1: [pos_ids[0], neg_ids[0], pos_ids[1], neg_ids[1]],
        2: [pos_ids[2], neg_ids[2], pos_ids[3], neg_ids[3]],
    }
    return make_table(layout_a), make_table(layout_b), sequence, vectors


PROBE_HEADROOM_BITS = 1


def run_probe(config):
    probe_config = dict(config)
    probe_config["accumulator_headroom_bits"] = PROBE_HEADROOM_BITS
    table_a, table_b, sequence, vectors = build_probe_tables(probe_config)
    level = probe_config["quant_level"]
    fbits = probe_config["fractional_bits"]
    headroom = probe_config["accumulator_headroom_bits"]
    acc_min, acc_max = accumulator_bounds(level, headroom)

    def fixed_path(table):
        acc = np.zeros(probe_config["dim"], dtype=np.int64)
        block_partials = {}
        traversal = []
        for rid in sequence:
            row = table.rows[rid]
            int_row, _, _ = quantize_row(row.vector, level, fbits)
            bid = row.block_id
            if bid not in block_partials:
                block_partials[bid] = np.zeros(config["dim"], dtype=np.int64)
                traversal.append(bid)
            block_partials[bid] = saturating_add(block_partials[bid], int_row, acc_min, acc_max)
            acc = saturating_add(acc, int_row, acc_min, acc_max)
        ordered_merge = np.zeros(probe_config["dim"], dtype=np.int64)
        for bid in traversal:
            ordered_merge = saturating_add(ordered_merge, block_partials[bid], acc_min, acc_max)
        return acc, ordered_merge, block_partials, traversal

    def float_path(table):
        total = np.zeros(probe_config["dim"], dtype=np.float64)
        for rid in sequence:
            total = total + table.rows[rid].vector
        return total

    acc_a, merged_a, partials_a, traversal_a = fixed_path(table_a)
    acc_b, merged_b, partials_b, traversal_b = fixed_path(table_b)
    ref_a = float_path(table_a)
    ref_b = float_path(table_b)

    fixed_diff = np.max(np.abs(acc_a.astype(np.int64) - acc_b.astype(np.int64)))
    merge_diff = np.max(np.abs(merged_a - merged_b))
    float_diff = float(np.max(np.abs(ref_a - ref_b)))

    def describe(table, traversal, partials, acc, merged):
        return {
            "block_traversal": traversal,
            "block_partials_int": {
                str(bid): partials[bid].tolist() for bid in traversal
            },
            "block_merged_sum_int": merged.tolist(),
            "final_sequential_sum_int": acc.tolist(),
            "final_sum_int": acc.tolist(),
            "final_sum_float": dequantize(acc, fbits).tolist(),
            "partial_vs_final_diff": {
                str(bid): float(
                    np.max(np.abs(dequantize(partials[bid], fbits) - dequantize(acc, fbits)))
                )
                for bid in traversal
            },
        }

    return {
        "id_sequence": sequence,
        "vectors": {str(k): v for k, v in vectors.items()},
        "accumulator_bounds": [int(acc_min), int(acc_max)],
        "probe_headroom_bits": PROBE_HEADROOM_BITS,
        "layout_a": describe(table_a, traversal_a, partials_a, acc_a, merged_a),
        "layout_b": describe(table_b, traversal_b, partials_b, acc_b, merged_b),
        "fixedpoint_max_abs_diff": int(fixed_diff),
        "block_merge_max_abs_diff": int(merge_diff),
        "float64_reference_max_abs_diff": float_diff,
        "order_effect_observed": bool(fixed_diff > 0 or merge_diff > 0),
        "float_reference_consistent": bool(float_diff == 0.0),
    }
