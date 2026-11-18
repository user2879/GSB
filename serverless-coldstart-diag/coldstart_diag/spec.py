"""函数版本清单、调用序列与并发策略的加载与强校验。

任一不合法即以 E_SPEC 整体拒绝加载，并输出全部出错字段路径与违规值。
"""

import json
from pathlib import Path

from .errors import SpecError
from .fixedpoint import power_of_ten

MEMORY_MIN, MEMORY_MAX = 64, 10240
INIT_MS_MIN, INIT_MS_MAX = 1, 600000
CONCURRENCY_MIN, CONCURRENCY_MAX = 1, 1000
INSTANCES_MIN_MAX = 10000
CALLS_MAX = 1_000_000
PERMYRIAD_MIN, PERMYRIAD_MAX = 0, 10000


def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


class ValidationReport:
    def __init__(self):
        self.errors = []

    def fail(self, path, value, reason):
        self.errors.append({"path": path, "value": value, "reason": reason})

    def check_int(self, path, value, lo=None, hi=None):
        if not _is_int(value):
            self.fail(path, value, "必须是整数")
            return False
        ok = True
        if lo is not None and value < lo:
            self.fail(path, value, f"小于下界 {lo}")
            ok = False
        if hi is not None and value > hi:
            self.fail(path, value, f"大于上界 {hi}")
            ok = False
        return ok

    def check_str(self, path, value):
        if not isinstance(value, str) or not value.strip():
            self.fail(path, value, "必须是非空字符串")
            return False
        return True

    def raise_if_any(self):
        if self.errors:
            raise SpecError(
                "清单校验失败，整体拒绝加载",
                details=self.errors,
            )


def load_manifest(path):
    p = Path(path)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SpecError(f"清单文件不存在: {p}", path="<file>")
    except json.JSONDecodeError as exc:
        raise SpecError(f"JSON 解析失败: {exc}", path="<json>")
    return validate_manifest(raw)


def validate_manifest(raw):
    if not isinstance(raw, dict):
        raise SpecError("清单根节点必须是对象", path="<root>", value=type(raw).__name__)
    rep = ValidationReport()

    manifest_id = raw.get("manifest_id")
    rep.check_str("manifest_id", manifest_id)
    declaration_version = raw.get("declaration_version")
    rep.check_int("declaration_version", declaration_version, 1, 2**31 - 1)
    scale = raw.get("scale")
    if rep.check_int("scale", scale, 1):
        try:
            power_of_ten(scale)
        except SpecError:
            rep.fail("scale", scale, "必须是 10 的整数次幂")

    functions = raw.get("functions")
    fn_keys = _validate_functions(functions, rep)
    calls = raw.get("call_sequence")
    call_keys = _validate_calls(calls, rep)
    _validate_policy(raw.get("policy"), rep)

    rep.raise_if_any()

    # 引用完整性：调用序列必须引用已声明的函数与版本
    if isinstance(calls, list):
        for idx, item in enumerate(calls):
            key = (item.get("function_id"), item.get("function_version"))
            if key not in fn_keys:
                rep.fail(
                    f"call_sequence[{idx}].function_id/function_version",
                    list(key),
                    "引用了未声明的函数或版本",
                )
    rep.raise_if_any()
    return raw


def _validate_functions(functions, rep):
    fn_keys = set()
    if not isinstance(functions, list) or not functions:
        rep.fail("functions", functions, "必须是非空数组")
        return fn_keys
    seen_ids = set()
    versions_by_id = {}
    for i, fn in enumerate(functions):
        base = f"functions[{i}]"
        if not isinstance(fn, dict):
            rep.fail(base, fn, "必须是对象")
            continue
        fid = fn.get("function_id")
        rep.check_str(f"{base}.function_id", fid)
        fver = fn.get("function_version")
        rep.check_int(f"{base}.function_version", fver, 1, 2**31 - 1)
        if _is_int(fver):
            versions_by_id.setdefault(fid, []).append((i, fver))
        rep.check_int(f"{base}.memory_mib", fn.get("memory_mib"), MEMORY_MIN, MEMORY_MAX)
        rep.check_int(f"{base}.init_ms", fn.get("init_ms"), INIT_MS_MIN, INIT_MS_MAX)
        rep.check_int(
            f"{base}.concurrency",
            fn.get("concurrency"),
            CONCURRENCY_MIN,
            CONCURRENCY_MAX,
        )
        mn = fn.get("min_instances")
        mx = fn.get("max_instances")
        rep.check_int(f"{base}.min_instances", mn, 0, INSTANCES_MIN_MAX)
        rep.check_int(f"{base}.max_instances", mx, 1, INSTANCES_MIN_MAX)
        if _is_int(mn) and _is_int(mx) and mn > mx:
            rep.fail(f"{base}.min_instances", mn, "min_instances 不得大于 max_instances")
        if fid and _is_int(fver):
            key = (fid, fver)
            if key in fn_keys:
                rep.fail(f"{base}.function_version", fver, "函数标识+版本号必须唯一")
            fn_keys.add(key)
            seen_ids.add(fid)
    # 版本号单调递增（按声明顺序）
    for fid, pairs in versions_by_id.items():
        ordered = sorted(pairs)
        prev = None
        for _, ver in ordered:
            if prev is not None and ver <= prev:
                rep.fail(
                    f"functions[version of {fid}]",
                    ver,
                    "同一函数的版本号必须单调递增",
                )
            prev = ver
    return fn_keys


def _validate_calls(calls, rep):
    keys = set()
    if not isinstance(calls, list) or not calls:
        rep.fail("call_sequence", calls, "必须是非空数组")
        return keys
    seen_minutes = set()
    prev_minute = None
    for i, item in enumerate(calls):
        base = f"call_sequence[{i}]"
        if not isinstance(item, dict):
            rep.fail(base, item, "必须是对象")
            continue
        rep.check_str(f"{base}.function_id", item.get("function_id"))
        rep.check_int(f"{base}.function_version", item.get("function_version"), 1)
        minute = item.get("minute")
        rep.check_int(f"{base}.minute", minute, 0, 2**31 - 1)
        rep.check_int(f"{base}.calls", item.get("calls"), 0, CALLS_MAX)
        rep.check_int(
            f"{base}.burst_permyriad",
            item.get("burst_permyriad"),
            PERMYRIAD_MIN,
            PERMYRIAD_MAX,
        )
        if _is_int(minute):
            if minute in seen_minutes:
                rep.fail(f"{base}.minute", minute, "分钟时间戳重复")
            seen_minutes.add(minute)
            if prev_minute is not None and minute != prev_minute + 1:
                rep.fail(f"{base}.minute", minute, "分钟时间戳必须连续递增（步长 1）")
            prev_minute = minute
        keys.add((item.get("function_id"), item.get("function_version"), minute))
    return keys


def _validate_policy(policy, rep):
    if not isinstance(policy, dict):
        rep.fail("policy", policy, "必须是对象")
        return
    rep.check_int(
        "policy.instance_concurrency_cap",
        policy.get("instance_concurrency_cap"),
        1,
        CONCURRENCY_MAX,
    )
    rep.check_int(
        "policy.global_concurrency_cap",
        policy.get("global_concurrency_cap"),
        1,
        10**9,
    )
    rep.check_int(
        "policy.reserved_instances",
        policy.get("reserved_instances"),
        0,
        INSTANCES_MIN_MAX,
    )
    rep.check_int(
        "policy.warmup_permyriad",
        policy.get("warmup_permyriad"),
        PERMYRIAD_MIN,
        PERMYRIAD_MAX,
    )
    rep.check_int(
        "policy.global_memory_mib",
        policy.get("global_memory_mib"),
        1,
        10**9,
    )


def select_function(manifest, function_id, function_version):
    for fn in manifest["functions"]:
        if fn["function_id"] == function_id and fn["function_version"] == function_version:
            return fn
    raise SpecError(
        "引用的函数版本不存在",
        path="call_sequence",
        value=[function_id, function_version],
    )
