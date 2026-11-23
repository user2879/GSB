"""段级音色描述向量逐维自适应量化。

量化模式按音频内全部段的统计量构造：
- 步长取该维自然尺度（量纲比例尺度），吸收同说话人段间自然抖动；
- 档位（整数容量）随观测动态范围自适应增大；
- 位宽/档位切换会真实改变整数向量、重构向量与残差；
- 量纲冲突或缺失维度输出 E_DIM_QUANT_CONFLICT 并保留冲突维清单。
"""

from .errors import PipelineError, E_DIM_QUANT_CONFLICT
from .utils import canonical_json, round_fixed, sha256_text

NATURAL_RELATIVE_STEP = 0.08
DIMENSION_RELATIVE_STEP = {
    "f0_median": 0.06,
    "f0_iqr": 0.25,
    "f1_median": 0.08,
    "f2_median": 0.08,
    "f3_median": 0.08,
    "syllable_rate": 0.22,
    "voiced_ratio": 0.12,
    "energy_cv": 0.14,
}
MIN_ABSOLUTE_STEP = {
    "f0_median": 12.0,
    "f0_iqr": 3.0,
    "f1_median": 70.0,
    "f2_median": 160.0,
    "f3_median": 480.0,
    "syllable_rate": 0.9,
    "voiced_ratio": 0.1,
    "energy_cv": 0.1,
}


def choose_level(required_capacity, levels):
    """在候选档位中选择能容纳所需整数容量的最小档位。"""
    candidates = sorted(int(level) for level in levels)
    for level in candidates:
        if level >= required_capacity:
            return level
    return candidates[-1]


def build_dimension_schemas(config, descriptor_rows):
    dimensions = config.dimensions()
    schemas = []
    for dimension in dimensions:
        name = dimension["name"]
        unit = dimension.get("unit", "")
        values = [row[name] for row in descriptor_rows
                  if isinstance(row.get(name), (int, float))]
        if not values:
            schemas.append({
                "name": name, "unit": unit, "selected_level": min(dimension["levels"]),
                "capacity": max(2, min(dimension["levels"])), "step": 1.0,
                "offset": 0.0, "raw_min": 0.0, "raw_max": 0.0,
            })
            continue
        vmin, vmax = min(values), max(values)
        magnitude = max(abs(vmin), abs(vmax), 1.0)
        relative_step = DIMENSION_RELATIVE_STEP.get(name, NATURAL_RELATIVE_STEP)
        natural_step = max(MIN_ABSOLUTE_STEP.get(name, 1.0),
                           magnitude * relative_step)
        span = max(vmax - vmin, natural_step)
        required_capacity = max(2, int(round(span / natural_step)) + 2)
        level = choose_level(required_capacity, dimension["levels"])
        capacity = max(2, int(level))
        fine_step = span / (capacity - 1) if capacity > 1 else natural_step
        if level > required_capacity:
            # 显式更宽档位：用更细步长填满容量，位宽切换会真实改变整数。
            step = max(fine_step, MIN_ABSOLUTE_STEP.get(name, 1e-9) * 0.25)
        else:
            # 自适应档位：保持自然粗步长，吸收同说话人段间自然抖动。
            step = natural_step
        schemas.append({
            "name": name,
            "unit": unit,
            "selected_level": level,
            "capacity": capacity,
            "step": round_fixed(step),
            "offset": round_fixed(vmin),
            "raw_min": round_fixed(vmin),
            "raw_max": round_fixed(vmax),
        })
    return schemas


def _check_conflicts(dimensions, inputs, injection):
    conflict_dims = []
    if injection and injection.get("type") == "dim_mismatch":
        inputs = dict(inputs)
        inputs.pop(injection.get("drop_dim"), None)
    for dimension in dimensions:
        name = dimension["name"]
        value = inputs.get(name)
        if value is None or not isinstance(value, (int, float)):
            conflict_dims.append(name)
            continue
        unit = dimension.get("unit", "")
        if unit == "ratio" and value < 0.0:
            conflict_dims.append(name)
        elif unit == "hz" and value < 0:
            conflict_dims.append(name)
    return conflict_dims, inputs


def quantize_with_schema(feature, schemas, injection=None):
    dimensions_conf = [{"name": schema["name"], "unit": schema["unit"],
                        "levels": [schema["selected_level"]]} for schema in schemas]
    conflict_dims, inputs = _check_conflicts(
        dimensions_conf, feature["descriptor_inputs"], injection)
    if conflict_dims:
        raise PipelineError(
            E_DIM_QUANT_CONFLICT,
            "量纲冲突或缺失维度: {}".format(",".join(conflict_dims)),
            "quantize", feature["segment_id"],
            extra={"injection_id": injection.get("id") if injection else None,
                   "conflict_dims": conflict_dims},
        )
    per_dim = []
    integer_vector = []
    descriptor_vector = []
    for schema in schemas:
        value = float(inputs[schema["name"]])
        step = max(schema["step"], 1e-12)
        integer = int(round_fixed((value - schema["offset"]) / step))
        integer = max(0, min(schema["capacity"] - 1, integer))
        reconstructed = round_fixed(schema["offset"] + integer * step)
        residual = round_fixed(value - reconstructed)
        per_dim.append({
            "name": schema["name"],
            "unit": schema["unit"],
            "raw_value": round_fixed(value),
            "selected_level": schema["selected_level"],
            "level_capacity": schema["capacity"],
            "step": schema["step"],
            "offset": schema["offset"],
            "quantized_integer": integer,
            "reconstructed_value": reconstructed,
            "residual": residual,
        })
        integer_vector.append(integer)
        descriptor_vector.append(reconstructed)
    result = {
        "segment_id": feature["segment_id"],
        "feature_fingerprint": feature["feature_fingerprint"],
        "dimensions": per_dim,
        "integer_vector": integer_vector,
        "descriptor_vector": descriptor_vector,
        "scale": 1000,
        "boundary_sample": False,
        "boundary_rule": None,
        "schema_fingerprint": schema_fingerprint(schemas),
    }
    result["quant_fingerprint"] = quant_fingerprint(
        feature["feature_fingerprint"], result["schema_fingerprint"], result)
    return result


def quantize_segment(config, feature, injection=None):
    schemas = build_dimension_schemas(config, [feature["descriptor_inputs"]])
    return quantize_with_schema(feature, schemas, injection)


def schema_fingerprint(schemas):
    payload = [{key: schema[key] for key in
                ("name", "selected_level", "capacity", "step", "offset")}
               for schema in schemas]
    return "schema_" + sha256_text(canonical_json(payload))[:16]


def quant_fingerprint(feature_hash, schema_hash, result):
    payload = {
        "feature_fingerprint": feature_hash,
        "schema_fingerprint": schema_hash,
        "integer_vector": result["integer_vector"],
        "selected_levels": [item["selected_level"] for item in result["dimensions"]],
        "steps": [item["step"] for item in result["dimensions"]],
    }
    return "quant_" + sha256_text(canonical_json(payload))[:16]
