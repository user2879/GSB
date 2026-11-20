"""期望聚合摘要计算：平台侧与节点侧共用，保证逐字节一致。"""

from .canon import aggregate, canon, fp_div, fp_mul, sha256_hex, tdiv

DERIVED_OPS = ("add", "sub", "mul", "div")


def apply_derived(context, derived):
    """在数值上下文上按拓扑序应用派生关系，返回完整数值表。

    context: {key: int(milli)}；derived: [{"key","op","inputs":[a,b]}]
    派生键可引用上下文键或先前排出的派生键。
    """
    values = dict(context)
    for d in derived:
        a = values[d["inputs"][0]]
        b = values[d["inputs"][1]]
        op = d["op"]
        if op == "add":
            values[d["key"]] = a + b
        elif op == "sub":
            values[d["key"]] = a - b
        elif op == "mul":
            values[d["key"]] = fp_mul(a, b)
        elif op == "div":
            values[d["key"]] = fp_div(a, b)
        else:
            raise ValueError(f"未知派生运算: {op!r}")
    return values


def effective_value(key, value, context, derived):
    """求某键在切片中的生效值：派生键由上下文重算，普通键取自身值。"""
    if any(d["key"] == key for d in derived):
        numeric = {k: v for k, v in context.items()
                   if isinstance(v, int) and not isinstance(v, bool)}
        return apply_derived(numeric, derived)[key]
    return value


def compute_summary(manifest_version, profile, items, series):
    """计算切片的规范化摘要字符串。

    items: [[key, effective_value, key_version], ...]（按键名稳定排序）
    series: {key: [int(milli), ...]} 确定性样本数据
    返回规范化 JSON 字符串；digest 为其 sha256。
    """
    terms = []
    item_rows = []
    for key, value, key_version in sorted(items, key=lambda it: it[0]):
        item_rows.append([key, key_version, canon(value)])
        if isinstance(value, int) and not isinstance(value, bool):
            for element in series.get(key, []):
                terms.append(fp_mul(value, element))
    summary = {
        "agg": aggregate(profile, terms),
        "items": item_rows,
        "mv": manifest_version,
        "profile": profile,
    }
    return canon(summary)


def digest_of(summary):
    return sha256_hex(summary)
