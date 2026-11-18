"""逐分钟冷启动与并发控制模拟引擎（确定性、纯整数定点口径）。

模型规则（全部为整数运算，比例按 floor 向负无穷截断）：

1. 突发并发  burst = floor(calls * burst_permyriad / 10000)
2. 内存压力  pressure_eff_conc = floor(concurrency * global_memory_mib
                                       / (global_memory_mib + memory_mib))
   （下限 1）。内存规格越大、可用单实例并发度越低，从而推高实例需求。
3. 原始需求  raw = max(ceil(burst/eff_conc), ceil(calls/eff_conc))，calls>0 时至少 1。
4. 夹取      desired = min(max(raw, min_instances), max_instances)
5. 全局并发  global_cap_instances = floor(global_concurrency_cap / eff_conc)
6. 内存预算  mem_cap_instances    = floor(global_memory_mib / memory_mib)
   裁决上限  cap = min(max_instances, global_cap_instances, mem_cap_instances)
   预留      warm_target = max(reserved_instances, ceil(desired * warmup_permyriad/10000))
   受上限夹取 warm_target = min(warm_target, cap)；预留与全局上限冲突时以 cap 为准并记录理由。
7. 可用实例  active = min(desired, cap)；预热不足部分按需冷启动；超出容量的调用排队。
8. 热/冷拆分：热容量 = min(warm, active) * eff_conc，热调用 = min(calls, 热容量)，
   其余为冷调用；cold_starts = max(0, active - warm)，prewarm_starts = warm - warm_prev。
9. 排队等待  队列 = max(0, calls - active*eff_conc)；每个排队调用等待一次 init_ms。
   重裁决结果只依赖有序整数集合，不依赖任何 dict/set 遍历顺序。
"""

from dataclasses import dataclass, field

from .fixedpoint import scaled_ratio, single_value_error_bound
from .spec import select_function

PERMYRIAD = 10000


def ceil_div(a, b):
    if b <= 0:
        return 0
    return -((-int(a)) // int(b))


def stable_allocate(total, slots):
    """把 total 个调用确定性地分配到 slots 个有序实例（索引 0..slots-1）。

    采用“先均摊、余数按索引前缀分配”的稳定规则，与任何哈希顺序无关。
    """
    if slots <= 0:
        return []
    base, rem = divmod(int(total), int(slots))
    return [base + (1 if idx < rem else 0) for idx in range(int(slots))]


@dataclass
class MinuteState:
    minute: int
    warm_instances: int = 0


@dataclass
class MinuteResult:
    minute: int
    function_id: str
    function_version: int
    declaration_version: int
    scale: int
    calls: int
    burst_permyriad: int
    burst_calls: int
    eff_concurrency: int
    required_instances_raw: int
    required_instances: int
    global_cap_instances: int
    memory_cap_instances: int
    available_instances: int
    warm_target: int
    warm_instances: int
    prewarm_starts: int
    cold_starts: int
    init_starts: int
    hot_calls: int
    cold_calls: int
    queued_calls: int
    total_wait_ms: int
    avg_wait_ms: int
    reserved_used: int
    reserved_target: int
    cold_ratio_q: int
    wait_ratio_q: int
    occupancy_q: int
    reserved_occupancy_q: int
    arbitration_reasons: list = field(default_factory=list)
    instance_set: list = field(default_factory=list)
    warm_set: list = field(default_factory=list)
    cold_set: list = field(default_factory=list)
    allocation: list = field(default_factory=list)
    truncation: str = "floor（向负无穷截断，非负域即向下截断）"
    intrinsic_ulp_bound: int = 1
    error_block: dict = field(default_factory=dict)

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


class Engine:
    def __init__(self, manifest, function_id, function_version):
        self.manifest = manifest
        self.scale = manifest["scale"]
        self.declaration_version = manifest["declaration_version"]
        self.fn = select_function(manifest, function_id, function_version)
        self.policy = manifest["policy"]
        self.warm = 0

    def run_minute(self, item):
        scale = self.scale
        fn, policy = self.fn, self.policy
        minute, calls = item["minute"], item["calls"]
        burst_pm = item["burst_permyriad"]
        mem, conc = fn["memory_mib"], fn["concurrency"]
        global_mem = policy["global_memory_mib"]

        burst_calls = (calls * burst_pm) // PERMYRIAD

        eff_conc = (conc * global_mem) // (global_mem + mem)
        eff_conc = max(1, eff_conc)

        # 1) 需求推导
        if calls > 0:
            need_burst = ceil_div(burst_calls, eff_conc)
            need_total = ceil_div(calls, eff_conc)
            raw = max(1, need_burst, need_total)
        else:
            raw = 0
        desired = min(max(raw, fn["min_instances"]), fn["max_instances"])

        # 2) 全局并发上限与内存预算
        global_cap_inst = policy["global_concurrency_cap"] // eff_conc
        mem_cap_inst = global_mem // mem
        cap = min(fn["max_instances"], global_cap_inst, mem_cap_inst)

        reasons = []
        if cap < desired:
            reasons.append("CLAMPED_BY_GLOBAL_OR_MEMORY_CAP")
        if global_cap_inst < desired:
            reasons.append("GLOBAL_CONCURRENCY_CAP_BINDING")
        if mem_cap_inst < desired:
            reasons.append("GLOBAL_MEMORY_CAP_BINDING")

        # 3) 预留实例与预热比例（冲突时稳定夹取，记录依据）
        warm_target = max(
            policy["reserved_instances"],
            ceil_div(desired * policy["warmup_permyriad"], PERMYRIAD),
        )
        if warm_target > cap:
            reasons.append("RESERVED_EXCEEDS_CAP_CLAMPED")
            warm_target = cap
        if policy["reserved_instances"] > cap:
            reasons.append("RESERVED_VIOLATES_GLOBAL_CAP")

        # 4) 可用实例裁决（有序整数集合，幂等可重裁）
        active = min(desired, cap)
        instance_set = list(range(active))
        warm_prev = self.warm
        warm_now = min(warm_target, active)
        warm_set = list(range(warm_now))
        cold_set = list(range(warm_now, active))
        prewarm = max(0, warm_now - warm_prev)
        cold_starts = max(0, active - warm_now)
        init_starts = prewarm + cold_starts

        # 5) 热冷拆分（稳定分配，余数给低索引）
        hot_capacity = warm_now * eff_conc
        hot_calls = min(calls, hot_capacity)
        cold_calls = max(0, calls - hot_calls)
        capacity = active * eff_conc
        queued = max(0, calls - capacity)
        allocation = stable_allocate(calls, active)

        # 6) 排队等待（每个排队调用等待一次初始化耗时）
        total_wait = queued * fn["init_ms"]
        avg_wait = total_wait // calls if calls > 0 else 0

        # 7) 定点比例（floor）
        cold_ratio_q = scaled_ratio(cold_calls, calls, scale) if calls else 0
        wait_ratio_q = scaled_ratio(queued, calls, scale) if calls else 0
        occupancy_q = scaled_ratio(calls, capacity, scale) if capacity else 0
        reserved_target = min(policy["reserved_instances"], cap)
        reserved_used = min(warm_now, reserved_target)
        reserved_occupancy_q = (
            scaled_ratio(reserved_used, reserved_target, scale) if reserved_target else 0
        )

        result = MinuteResult(
            minute=minute,
            function_id=fn["function_id"],
            function_version=fn["function_version"],
            declaration_version=self.declaration_version,
            scale=scale,
            calls=calls,
            burst_permyriad=burst_pm,
            burst_calls=burst_calls,
            eff_concurrency=eff_conc,
            required_instances_raw=raw,
            required_instances=desired,
            global_cap_instances=global_cap_inst,
            memory_cap_instances=mem_cap_inst,
            available_instances=active,
            warm_target=warm_target,
            warm_instances=warm_now,
            prewarm_starts=prewarm,
            cold_starts=cold_starts,
            init_starts=init_starts,
            hot_calls=hot_calls,
            cold_calls=cold_calls,
            queued_calls=queued,
            total_wait_ms=total_wait,
            avg_wait_ms=avg_wait,
            reserved_used=reserved_used,
            reserved_target=reserved_target,
            cold_ratio_q=cold_ratio_q,
            wait_ratio_q=wait_ratio_q,
            occupancy_q=occupancy_q,
            reserved_occupancy_q=reserved_occupancy_q,
            arbitration_reasons=reasons,
            instance_set=instance_set,
            warm_set=warm_set,
            cold_set=cold_set,
            allocation=allocation,
        )
        result.error_block = synthesize_error_budget(self, item, result)
        self.warm = warm_now
        return result

    def run_all(self, call_sequence):
        return [self.run_minute(item) for item in sorted(call_sequence, key=lambda x: x["minute"])]


def _recompute_cold_ratio(engine, item, warm_prev, mem, conc, warmup_pm):
    """以给定扰动参数独立重算单分钟冷启动比例（不改动引擎状态）。"""
    policy, fn = engine.policy, engine.fn
    scale, calls = engine.scale, item["calls"]
    burst_pm = item["burst_permyriad"]
    global_mem = policy["global_memory_mib"]
    burst_calls = (calls * burst_pm) // PERMYRIAD
    eff_conc = max(1, (conc * global_mem) // (global_mem + mem))
    raw = (
        max(1, ceil_div(burst_calls, eff_conc), ceil_div(calls, eff_conc))
        if calls > 0
        else 0
    )
    desired = min(max(raw, fn["min_instances"]), fn["max_instances"])
    global_cap_inst = policy["global_concurrency_cap"] // eff_conc
    mem_cap_inst = global_mem // mem
    cap = min(fn["max_instances"], global_cap_inst, mem_cap_inst)
    warm_target = max(
        policy["reserved_instances"],
        ceil_div(desired * warmup_pm, PERMYRIAD),
    )
    warm_target = min(warm_target, cap)
    active = min(desired, cap)
    warm_now = min(warm_target, active)
    hot_capacity = warm_now * eff_conc
    cold_calls = max(0, calls - min(calls, hot_capacity))
    return scaled_ratio(cold_calls, calls, scale) if calls else 0


def synthesize_error_budget(engine, item, result):
    """逐项合成冷启动比例估算的端到端定点误差上界（单位：ulp）。

    输入量化步长：内存 1 MiB、单实例并发度 1、预热比例 1/10000。
    对每个输入做 ±1 步长扰动重算，取差值绝对值的最大值作为该输入贡献；
    再叠加 floor 单点量化误差 1 ulp。各级贡献占比按定点口径输出。
    """
    base = result.cold_ratio_q
    warm_prev = max(0, result.warm_instances - result.prewarm_starts)
    fn, policy = engine.fn, engine.policy

    perturbations = (
        ("memory_mib", [fn["memory_mib"] - 1, fn["memory_mib"] + 1],
         lambda m: _recompute_cold_ratio(engine, item, warm_prev, m, fn["concurrency"],
                                         policy["warmup_permyriad"])),
        ("concurrency", [fn["concurrency"] - 1, fn["concurrency"] + 1],
         lambda c: _recompute_cold_ratio(engine, item, warm_prev, fn["memory_mib"], c,
                                         policy["warmup_permyriad"])),
        ("warmup_permyriad",
         [policy["warmup_permyriad"] - 1, policy["warmup_permyriad"] + 1],
         lambda w: _recompute_cold_ratio(engine, item, warm_prev, fn["memory_mib"],
                                         fn["concurrency"], w)),
    )
    contributions = {}
    bound_total = single_value_error_bound()  # 本征 floor 量化 1 ulp
    for name, values, fnc in perturbations:
        worst = 0
        for value in values:
            if value < 0:
                continue
            q = fnc(value)
            worst = max(worst, abs(q - base))
        contributions[name] = worst
        bound_total += worst

    share_scale = engine.scale
    raw_sum = sum(contributions.values()) + 1
    shares = {}
    for name in ("memory_mib", "concurrency", "warmup_permyriad"):
        numerator = contributions[name] * share_scale
        shares[name] = numerator // raw_sum if raw_sum else 0
    shares["intrinsic_quantization"] = share_scale // raw_sum if raw_sum else 0

    return {
        "unit": "ulp",
        "scale": engine.scale,
        "base_cold_ratio_q": base,
        "end_to_end_bound_ulp": bound_total,
        "contributions_ulp": {
            "memory_mib": contributions["memory_mib"],
            "concurrency": contributions["concurrency"],
            "warmup_permyriad": contributions["warmup_permyriad"],
            "intrinsic_quantization": 1,
        },
        "contribution_shares_q": shares,
        "rounding": "floor（向下截断），真实值位于 [q, q+bound] ulp",
    }


def aggregate(results, scale):
    """按分钟聚合：均值与比例均为整数定点（floor），附量化误差上界与舍入方向。"""
    n = len(results)
    if n == 0:
        return {}
    total_calls = sum(r.calls for r in results)
    total_cold_calls = sum(r.cold_calls for r in results)
    total_queued = sum(r.queued_calls for r in results)
    total_wait = sum(r.total_wait_ms for r in results)
    total_cold_starts = sum(r.cold_starts for r in results)
    total_prewarm = sum(r.prewarm_starts for r in results)

    avg_cold_ratio_q = sum(r.cold_ratio_q for r in results) // n
    avg_wait_ratio_q = sum(r.wait_ratio_q for r in results) // n
    avg_occupancy_q = sum(r.occupancy_q for r in results) // n
    avg_wait_ms = total_wait // total_calls if total_calls else 0
    overall_cold_ratio_q = scaled_ratio(total_cold_calls, total_calls, scale)
    overall_wait_ratio_q = scaled_ratio(total_queued, total_calls, scale)
    # 误差上界：分钟均值的误差不超过单分钟上界；端到端界取各分钟最大值
    max_e2e = max(r.error_block["end_to_end_bound_ulp"] for r in results)
    return {
        "declaration_version": results[0].declaration_version,
        "minutes": n,
        "total_calls": total_calls,
        "total_cold_calls": total_cold_calls,
        "total_queued_calls": total_queued,
        "total_cold_starts": total_cold_starts,
        "total_prewarm_starts": total_prewarm,
        "total_wait_ms": total_wait,
        "avg_wait_ms": avg_wait_ms,
        "overall_cold_ratio_q": overall_cold_ratio_q,
        "overall_wait_ratio_q": overall_wait_ratio_q,
        "avg_cold_ratio_q": avg_cold_ratio_q,
        "avg_wait_ratio_q": avg_wait_ratio_q,
        "avg_occupancy_q": avg_occupancy_q,
        "agg_error_bound_ulp": max_e2e + 1,
        "rounding": "floor（向下截断）",
        "scale": scale,
    }
