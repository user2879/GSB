"""流水线编排: 列级失效判定、预算约束并发调度、阶段提交 (业务要求 6/8)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from . import fingerprints as fp
from .calibration import calibrate_column, derive_scale_zero_point
from .config import Config
from .dequant import dequant_column
from .errors import CqError, E_BUDGET_EXCEEDED, E_SAMPLE_INVALID, FailureList
from .quantize import quantize_column
from .samples import SampleSet
from .store import SnapshotStore, STAGES
from .stratify import stratify_groups


@dataclass
class StageStats:
    recomputed: list[str] = field(default_factory=list)
    reused: list[str] = field(default_factory=list)


@dataclass
class RunResult:
    columns: dict  # col_key -> 聚合结果 (calib/scale/quant/dequant)
    stratify: dict  # tensor -> 分层结果
    failures: FailureList
    stats: dict  # stage -> StageStats
    budget_spent: int
    budget_log: list
    version_change: str
    recovery_actions: list
    global_version: int
    config_hash: str


def _sort_key(stage: str, col_key: str) -> str:
    """由列标识与阶段名派生的稳定排序键."""
    return f"{stage}|{col_key}"


class Pipeline:
    def __init__(self, config: Config, samples: SampleSet, store: SnapshotStore):
        self.config = config
        self.samples = samples
        self.store = store
        self.failures = FailureList()
        self.stats = {s: StageStats() for s in STAGES}
        self.budget_spent = 0
        self.budget_log: list[dict] = []
        self._failed_columns: set[str] = set()

    # ---- 列枚举与上下文 ----
    def _all_columns(self) -> list[str]:
        keys = []
        for spec in self.config.tensors:
            for col in range(spec.num_columns):
                keys.append(f"{spec.name}:{col}")
        return sorted(keys)

    def _column_ctx(self, col_key: str) -> dict:
        tensor, col_s = col_key.split(":")
        col = int(col_s)
        st = self.samples.tensor(tensor)
        values, missing = st.column_view(col)
        return {"tensor": tensor, "col": col, "values": values, "missing": missing,
                "content_fp": fp.column_content_fp(values, missing)}

    # ---- 各阶段缓存键 ----
    def _keys(self, ctx: dict) -> dict:
        cfg = self.config
        ck = fp.calib_key(ctx["content_fp"],
                          fp.calib_param_fp(cfg.quantile_lo_permille, cfg.quantile_hi_permille))
        sk = fp.scale_key(ck, cfg.scheme, cfg.quant_bits)
        qk = fp.quant_key(sk, ctx["content_fp"], cfg.frac_bits, cfg.saturation)
        dk = fp.dequant_key(qk, ctx["content_fp"], cfg.abs_tol, cfg.rel_tol)
        return {"calib": ck, "scale": sk, "quant": qk, "dequant": dk}

    # ---- 预算扣减 (按列标识顺序, 确定性) ----
    def _budget_check(self, col_key: str, n_elements: int) -> bool:
        if self.budget_spent + n_elements > self.config.element_budget:
            self.failures.add(
                E_BUDGET_EXCEEDED,
                "单批标定元素预算耗尽, 列跳过",
                column=col_key,
                needed=n_elements,
                budget=self.config.element_budget,
                spent=self.budget_spent,
            )
            self._failed_columns.add(col_key)
            self.budget_log.append({"column": col_key, "action": "rejected",
                                    "needed": n_elements, "spent": self.budget_spent})
            return False
        self.budget_spent += n_elements
        self.budget_log.append({"column": col_key, "action": "charged",
                                "charged": n_elements, "spent": self.budget_spent})
        return True

    # ---- 并发执行: 任务纯函数, 结果按稳定键排序组装 ----
    def _run_stage(self, stage: str, tasks: list[tuple[str, callable]]) -> dict:
        """tasks: [(col_key, thunk)], 返回 {col_key: payload}. 顺序与单线程一致."""
        results: dict[str, dict] = {}
        if not tasks:
            return results
        ordered = sorted(tasks, key=lambda t: _sort_key(stage, t[0]))
        workers = max(1, min(self.config.max_workers, len(ordered)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [(key, pool.submit(thunk)) for key, thunk in ordered]
            for key, fut in futures:  # 按排序键顺序收集, 与完成顺序无关
                try:
                    results[key] = fut.result()
                except CqError as exc:
                    self.failures.add_error(exc)
                    self._failed_columns.add(key)
                except ValueError as exc:
                    self.failures.add(E_SAMPLE_INVALID, str(exc), column=key, stage=stage)
                    self._failed_columns.add(key)
        return results

    def run(self) -> RunResult:
        cfg = self.config
        version_change = self.store.detect_version_change(fp.config_fp(cfg))
        columns = self._all_columns()
        ctxs = {k: self._column_ctx(k) for k in columns}
        keys = {k: self._keys(ctxs[k]) for k in columns}

        artifacts: dict[str, dict[str, dict]] = {s: {} for s in STAGES}

        def current(stage: str, col_key: str) -> dict | None:
            if col_key in artifacts[stage]:
                return artifacts[stage][col_key]
            stored = self.store.get(stage, col_key)
            if stored is not None and stored["cache_key"] == keys[col_key][stage]:
                return stored
            return None

        # ---- calib 阶段 (预算约束) ----
        calib_tasks = []
        for col_key in columns:
            cached = current("calib", col_key)
            if cached is not None:
                artifacts["calib"][col_key] = cached
                self.stats["calib"].reused.append(col_key)
                continue
            n = int(ctxs[col_key]["values"].size)
            if not self._budget_check(col_key, n):
                continue
            ctx = ctxs[col_key]
            calib_tasks.append((col_key, lambda ctx=ctx, col_key=col_key: {
                "cache_key": keys[col_key]["calib"],
                "data": calibrate_column(ctx["values"], ctx["missing"],
                                         cfg.quantile_lo_permille,
                                         cfg.quantile_hi_permille),
            }))
        for key, payload in self._run_stage("calib", calib_tasks).items():
            artifacts["calib"][key] = payload
            self.stats["calib"].recomputed.append(key)

        # ---- scale 阶段 ----
        scale_tasks = []
        for col_key in columns:
            if col_key in self._failed_columns:
                continue
            cached = current("scale", col_key)
            if cached is not None:
                artifacts["scale"][col_key] = cached
                self.stats["scale"].reused.append(col_key)
                continue
            calib = artifacts["calib"].get(col_key)
            if calib is None:
                continue
            scale_tasks.append((col_key, lambda calib=calib, col_key=col_key: {
                "cache_key": keys[col_key]["scale"],
                "data": derive_scale_zero_point(calib["data"], cfg.scheme, cfg.quant_bits),
            }))
        for key, payload in self._run_stage("scale", scale_tasks).items():
            artifacts["scale"][key] = payload
            self.stats["scale"].recomputed.append(key)

        # ---- quant 阶段 ----
        quant_tasks = []
        for col_key in columns:
            if col_key in self._failed_columns:
                continue
            cached = current("quant", col_key)
            if cached is not None:
                artifacts["quant"][col_key] = cached
                self.stats["quant"].reused.append(col_key)
                continue
            scale = artifacts["scale"].get(col_key)
            if scale is None:
                continue
            ctx = ctxs[col_key]
            sd = scale["data"]
            quant_tasks.append((col_key, lambda ctx=ctx, sd=sd, col_key=col_key: {
                "cache_key": keys[col_key]["quant"],
                "data": quantize_column(ctx["values"], ctx["missing"], sd["scale"],
                                        sd["zero_point"], sd["qmin"], sd["qmax"],
                                        cfg.frac_bits, cfg.saturation),
            }))
        for key, payload in self._run_stage("quant", quant_tasks).items():
            artifacts["quant"][key] = payload
            self.stats["quant"].recomputed.append(key)

        # ---- dequant 阶段 ----
        dequant_tasks = []
        for col_key in columns:
            if col_key in self._failed_columns:
                continue
            cached = current("dequant", col_key)
            if cached is not None:
                artifacts["dequant"][col_key] = cached
                self.stats["dequant"].reused.append(col_key)
                continue
            calib = artifacts["calib"].get(col_key)
            scale = artifacts["scale"].get(col_key)
            quant = artifacts["quant"].get(col_key)
            if calib is None or scale is None or quant is None:
                continue
            ctx = ctxs[col_key]
            cd, sd, qd = calib["data"], scale["data"], quant["data"]
            dequant_tasks.append((col_key, lambda ctx=ctx, cd=cd, sd=sd, qd=qd, col_key=col_key: {
                "cache_key": keys[col_key]["dequant"],
                "data": dequant_column(ctx["values"], ctx["missing"], qd, sd["scale"],
                                       sd["zero_point"], cd["q_lo"], cd["q_hi"],
                                       cfg.abs_tol, cfg.rel_tol),
            }))
        for key, payload in self._run_stage("dequant", dequant_tasks).items():
            artifacts["dequant"][key] = payload
            self.stats["dequant"].recomputed.append(key)

        # ---- stratify 阶段 (按张量) ----
        stratify_results: dict[str, dict] = {}
        for spec in cfg.tensors:
            tensor_key = f"{spec.name}:groups"
            col_keys = [f"{spec.name}:{c}" for c in range(spec.num_columns)]
            ready = [k for k in col_keys
                     if k not in self._failed_columns and k in artifacts["dequant"]]
            deq_keys = [artifacts["dequant"][k]["cache_key"] for k in ready]
            skey = fp.stratify_key(spec.name, cfg.groups[spec.name], deq_keys,
                                   cfg.group_oor_limit)
            cached = self.store.get("stratify", tensor_key)
            if cached is not None and cached["cache_key"] == skey:
                artifacts["stratify"][tensor_key] = cached
                stratify_results[spec.name] = cached["data"]
                self.stats["stratify"].reused.append(tensor_key)
                continue
            column_results = {}
            for k in ready:
                col = int(k.split(":")[1])
                column_results[col] = {
                    "dequant": artifacts["dequant"][k]["data"],
                    "quant": artifacts["quant"][k]["data"],
                    "valid_count": artifacts["calib"][k]["data"]["valid_count"],
                }
            data = stratify_groups(spec.name, cfg.groups[spec.name], column_results,
                                   cfg.quant_bits, cfg.group_oor_limit)
            payload = {"cache_key": skey, "data": data}
            artifacts["stratify"][tensor_key] = payload
            stratify_results[spec.name] = data
            self.stats["stratify"].recomputed.append(tensor_key)

        # ---- 提交 ----
        new_artifacts = {
            stage: {k: v for k, v in artifacts[stage].items()
                    if k in self.stats[stage].recomputed}
            for stage in STAGES
        }
        completed = [k for k in columns
                     if k not in self._failed_columns and k in artifacts["dequant"]]
        self.store.commit(new_artifacts, fp.config_fp(cfg), completed)

        # ---- 汇总列结果 ----
        col_results = {}
        for col_key in columns:
            if col_key in self._failed_columns:
                continue
            entry = {}
            for stage in ("calib", "scale", "quant", "dequant"):
                art = artifacts[stage].get(col_key)
                if art is not None:
                    entry[stage] = art["data"]
            if entry:
                col_results[col_key] = entry

        return RunResult(
            columns=col_results,
            stratify=stratify_results,
            failures=self.failures,
            stats={s: {"recomputed": sorted(self.stats[s].recomputed),
                       "reused": sorted(self.stats[s].reused)} for s in STAGES},
            budget_spent=self.budget_spent,
            budget_log=list(self.budget_log),
            version_change=version_change,
            recovery_actions=list(self.store.recovery_log),
            global_version=self.store.manifest["global_version"],
            config_hash=fp.config_fp(cfg),
        )
