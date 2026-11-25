"""定点化诊断管线主体：状态、增量失效、双版本行计算与跨层累计。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .conversion import DowngradeSidecar
from .error_track import (
    StageError,
    RowDiagnosis,
    build_stage_error,
    compare_tiers,
    first_exceeded_stage,
)
from .errors import ErrorCode, FailureList
from .fingerprints import (
    config_fp,
    param_fp,
    row_activation_fp,
    short_digest,
)
from .fixedpoint import QuantSpec
from .ledger import LedgerDecision, ReuseLedger, RowCacheEntry, build_stage_keys
from .models import (
    BIT_WIDTH_TO_SCHEME,
    OpConfig,
    Scheme,
    TensorPackage,
    V1WeightRow,
    V2WeightRow,
)
from .operators import (
    RowOpResult,
    compute_row,
    float64_layernorm,
    float64_softmax,
)
from .scheduler import Job, plan_batch, stable_jobs


def _v1_scheme(bit_width: int) -> Scheme:
    return BIT_WIDTH_TO_SCHEME[bit_width]


def _v2_scheme(scheme_name: str) -> Scheme:
    return Scheme(scheme_name)


@dataclass
class VersionWeightState:
    """单版本（v1/v2）下逐行参数与旁路。"""

    v1_rows: dict[tuple[str, int], V1WeightRow] = field(default_factory=dict)
    v2_rows: dict[tuple[str, int], V2WeightRow] = field(default_factory=dict)
    sidecars: dict[tuple[str, int], DowngradeSidecar] = field(default_factory=dict)


class PipelineState:
    """管线可变状态：激活、双版本权重、配置、缓存与账本。"""

    def __init__(self, package: TensorPackage, config: OpConfig, active_version: str = "v1"):
        self.package = package
        self.config = config
        self.active_version = active_version
        self.weights = VersionWeightState()
        self.cache: dict[tuple[str, str, int], RowCacheEntry] = {}
        self.ledger = ReuseLedger()
        self.failures = FailureList()
        self.conversion_records: list[dict] = []
        self.layer_order = self._topological_order()

    # ------------------------------------------------------------------ #
    def _topological_order(self) -> list[str]:
        """按依赖关系拓扑排序，同层内名字典序，保证稳定。"""

        visited: set[str] = set()
        order: list[str] = []

        def visit(layer: str) -> None:
            if layer in visited:
                return
            for dep in sorted(self.package.depends_on.get(layer, [])):
                if dep in self.package.layers:
                    visit(dep)
            visited.add(layer)
            order.append(layer)

        for layer in sorted(self.package.layers):
            visit(layer)
        return order

    def row_width(self, layer: str) -> int:
        return self.package.dims(layer)[1]

    def upstream_rows(self, layer: str, row_index: int) -> list[tuple[str, int]]:
        """行级上游集合：直接依赖层的全部行（跨层累计逐元素引用同序号，
        但任一上游行变化都会改变该层可用的累计输入集合，故全部记录）。"""

        refs: list[tuple[str, int]] = []
        for dep in self.package.depends_on.get(layer, []):
            if dep in self.package.layers:
                for upstream_row in range(len(self.package.layers[dep])):
                    refs.append((dep, upstream_row))
        return refs

    # ------------------------------------------------------------------ #
    # 版本参数解析
    # ------------------------------------------------------------------ #
    def resolve_params(self, layer: str, row_index: int) -> tuple[float, float, Scheme, str]:
        key = (layer, row_index)
        if self.active_version == "v1":
            v1 = self.weights.v1_rows[key]
            scheme = _v1_scheme(v1.bit_width)
            gamma = v1.scale_permille / 1000.0
            beta = v1.bias_code * (2.0 ** (-scheme.frac_bits))
            rounding = self.config.rounding_mode
            return gamma, beta, scheme, rounding
        v2 = self.weights.v2_rows[key]
        scheme = _v2_scheme(v2.scheme)
        gamma = v2.scale
        beta = 0.0
        rounding = v2.rounding_mode
        return gamma, beta, scheme, rounding

    def quant_spec(self, rounding: str) -> QuantSpec:
        return QuantSpec(
            bit_width=self._effective_bit_width(),
            frac_bits=self.config.frac_bits,
            tier=self.config.tier,
            rounding_mode=rounding,
        )

    def _effective_bit_width(self) -> int:
        # 位宽由权重方案决定；无权重时由配置档位给默认 16
        return getattr(self, "_forced_bit_width", 16)

    def row_bit_width(self, layer: str, row_index: int) -> int:
        _, _, scheme, _ = self.resolve_params(layer, row_index)
        return scheme.bit_width

    # ------------------------------------------------------------------ #
    # 单行计算（定点双算子 + float64 真值 + 四阶段误差 + 跨层累计）
    # ------------------------------------------------------------------ #
    def run_row(self, layer: str, row_index: int) -> RowCacheEntry:
        row = self.package.layers[layer][row_index]
        gamma, beta, scheme, rounding = self.resolve_params(layer, row_index)
        spec = QuantSpec(
            bit_width=scheme.bit_width,
            frac_bits=self.config.frac_bits,
            tier=self.config.tier,
            rounding_mode=rounding,
        )
        result = compute_row(
            layer, row_index, row, gamma, beta, scheme, spec, self.active_version
        )
        # float64 无量化真值
        truth_sm = float64_softmax(row)
        truth_ln = float64_layernorm(row, gamma, beta, scheme.use_bias)
        tolerances = self.config.tolerances
        stage_errors: list[StageError] = []
        # 环节 1：行内统计量（均值/方差对比）
        stats = result.ln_stats
        approx_stats = [stats["mean_float"], stats["var_float"]]
        truth_stats = [truth_ln["stats"]["mean"], truth_ln["stats"]["variance"]]
        stage_errors.append(
            build_stage_error(
                "row_stats", approx_stats, truth_stats,
                tolerances["abs"], tolerances["rel"],
            )
        )
        # 环节 2：Softmax 归一化
        stage_errors.append(
            build_stage_error(
                "softmax_norm", result.softmax_float, truth_sm,
                tolerances["abs"], tolerances["rel"],
            )
        )
        # 环节 3：LayerNorm 缩放偏置
        stage_errors.append(
            build_stage_error(
                "ln_scale_bias", result.ln_float, truth_ln["float"],
                tolerances["abs"], tolerances["rel"],
            )
        )
        # 环节 4：跨层累计（本层输入含上游输出时才有意义）
        cross = self._cross_layer_accumulate(
            layer, row_index, row, result, spec, truth_sm, truth_ln
        )
        cross_dict = cross["stage_error"]
        stage_errors.append(
            StageError(
                stage=cross_dict["stage"],
                max_abs_error=cross_dict["max_abs_error"],
                max_rel_error=cross_dict["max_rel_error"],
                zero_truth_branch=cross_dict["zero_truth_branch"],
                exceeded=cross_dict["exceeded"],
            )
        )
        first_stage = first_exceeded_stage(stage_errors)
        # 下游结论翻转归因：对比另一档位（coarse vs fine）
        flip = self._flip_attribution(layer, row_index, row, gamma, beta, scheme, spec, result)
        diagnosis = RowDiagnosis(
            layer=layer,
            row_index=row_index,
            version=self.active_version,
            stage_errors=stage_errors,
            first_exceed_stage=first_stage,
            flip=flip,
            truth_argmax=max(range(len(truth_sm)), key=lambda col: (truth_sm[col], -col)),
            fixed_argmax=result.softmax_argmax,
        )
        result_dict = result.to_dict()
        result_dict["truth"] = {
            "softmax": truth_sm,
            "layernorm": truth_ln["float"],
            "stats": truth_ln["stats"],
        }
        result_dict["diagnosis"] = diagnosis.to_dict()
        result_dict["cross_layer"] = cross
        activation_fingerprint = row_activation_fp(layer, row_index, row)
        parameter_fingerprint = param_fp(gamma, beta, scheme.value, rounding)
        configuration_fingerprint = config_fp(
            self.config.frac_bits, self.config.tier, rounding, self.active_version
        )
        upstream = self.upstream_rows(layer, row_index)
        upstream_fps = self._upstream_fingerprints(upstream)
        weight_fp = self._weight_fingerprint(layer, row_index)
        sidecar = self.weights.sidecars.get((layer, row_index))
        sidecar_fp = short_digest(sidecar.to_dict()) if sidecar else None
        keys = build_stage_keys(
            layer, row_index, self.active_version, activation_fingerprint,
            parameter_fingerprint, spec.key(), upstream_fps, weight_fp, sidecar_fp,
        )
        entry = RowCacheEntry(
            layer=layer,
            row_index=row_index,
            version=self.active_version,
            upstream_rows=upstream,
            activation_fp=activation_fingerprint,
            param_fp=parameter_fingerprint,
            config_fp=configuration_fingerprint,
            keys=keys,
            result=result_dict,
            cross_layer=cross,
        )
        self.cache[(self.active_version, layer, row_index)] = entry
        return entry

    def _upstream_fingerprints(
        self, upstream: list[tuple[str, int]]
    ) -> dict[tuple[str, int], str]:
        fps: dict[tuple[str, int], str] = {}
        for ref in upstream:
            entry = self.cache.get((self.active_version, ref[0], ref[1]))
            if entry is not None:
                # 以上游 Softmax 阶段缓存键作为行级产物指纹，传导算子输出变化
                fps[ref] = entry.keys["softmax"]
        return fps

    def _weight_fingerprint(self, layer: str, row_index: int) -> str:
        key = (layer, row_index)
        if self.active_version == "v1":
            v1 = self.weights.v1_rows[key]
            return short_digest(
                {"v": "v1", "s": v1.scale_permille, "b": v1.bias_code, "bw": v1.bit_width}
            )
        v2 = self.weights.v2_rows[key]
        return short_digest(
            {"v": "v2", "s": v2.scale, "sch": v2.scheme, "r": v2.rounding_mode}
        )

    # ------------------------------------------------------------------ #
    def _cross_layer_accumulate(
        self, layer, row_index, row, result: RowOpResult, spec: QuantSpec,
        truth_sm, truth_ln,
    ) -> dict:
        """跨层累计：本层激活 + 上游 LayerNorm 输出（定点逐元素相加）。"""

        upstream = self.upstream_rows(layer, row_index)
        if not upstream:
            zero = [0.0] * len(row)
            stage_error = build_stage_error(
                "cross_layer", zero, zero,
                self.config.tolerances["abs"], self.config.tolerances["rel"],
            )
            return {
                "upstream_rows": [],
                "fixed_accum": result.ln_float,
                "truth_accum": truth_ln["float"],
                "stage_error": stage_error.to_dict(),
                "note": "no-upstream",
            }
        from .fixedpoint import Quantizer

        quantizer = Quantizer(spec)
        fixed_accum_q = list(result.ln_int)
        truth_accum = list(truth_ln["float"])
        used: list[list] = []
        for ref_layer, ref_row in upstream:
            ref_entry = self.cache.get((self.active_version, ref_layer, ref_row))
            if ref_entry is None:
                continue
            ref_int = ref_entry.result["layernorm"]["int"]
            ref_float = ref_entry.result["truth"]["layernorm"]
            used.append([ref_layer, ref_row])
            for col in range(len(fixed_accum_q)):
                fixed_accum_q[col] = quantizer.add(
                    fixed_accum_q[col], ref_int[col], "cross.accum", col
                )
                truth_accum[col] += ref_float[col]
        fixed_accum = [quantizer.dequant(value) for value in fixed_accum_q]
        stage_error = build_stage_error(
            "cross_layer", fixed_accum, truth_accum,
            self.config.tolerances["abs"], self.config.tolerances["rel"],
        )
        return {
            "upstream_rows": used,
            "fixed_accum": fixed_accum,
            "truth_accum": truth_accum,
            "stage_error": stage_error.to_dict(),
            "note": "accumulated",
        }

    def _flip_attribution(
        self, layer, row_index, row, gamma, beta, scheme: Scheme,
        spec: QuantSpec, result: RowOpResult,
    ) -> str:
        """用 coarse/fine 两档位重算 Softmax argmax，判断结论是否翻转并归因。"""

        coarse_spec = QuantSpec(
            bit_width=scheme.bit_width, frac_bits=spec.frac_bits,
            tier="coarse", rounding_mode=spec.rounding_mode,
        )
        fine_spec = QuantSpec(
            bit_width=scheme.bit_width, frac_bits=spec.frac_bits,
            tier="fine", rounding_mode=spec.rounding_mode,
        )
        coarse = compute_row(layer, row_index, row, gamma, beta, scheme,
                             coarse_spec, self.active_version)
        fine = compute_row(layer, row_index, row, gamma, beta, scheme,
                           fine_spec, self.active_version)
        cross_inconsistent = (not result.degenerate) and any(
            event["direction"] != "EXACT"
            for event in result.softmax_events
        ) and self.upstream_rows(layer, row_index)
        return compare_tiers(
            coarse.softmax_argmax,
            fine.softmax_argmax,
            degenerate=result.degenerate,
            cross_layer_inconsistent=bool(cross_inconsistent),
        )

    # ------------------------------------------------------------------ #
    # 增量失效
    # ------------------------------------------------------------------ #
    def affected_rows(
        self, layer: str, row_index: int | None
    ) -> set[tuple[str, int]]:
        """参数/单行编辑影响的行集合：目标行 + 下游传递闭包。"""

        rows, cols = self.package.dims(layer)
        targets = (
            [(layer, r) for r in range(rows)]
            if row_index is None
            else [(layer, row_index)]
        )
        affected: set[tuple[str, int]] = set(targets)
        # 层级下游（依赖图反向传递闭包）：跨层累计引用整个上游层，
        # 故上游任一行变化都会使下游层全部行失效
        downstream_layers = self._downstream_layers(layer)
        for down_layer in downstream_layers:
            down_rows, _ = self.package.dims(down_layer)
            for r in range(down_rows):
                affected.add((down_layer, r))
        return affected

    def _downstream_layers(self, layer: str) -> list[str]:
        reverse: dict[str, list[str]] = {}
        for node, deps in self.package.depends_on.items():
            for dep in deps:
                reverse.setdefault(dep, []).append(node)
        seen: set[str] = set()
        stack = list(reverse.get(layer, []))
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(reverse.get(current, []))
        return sorted(seen)


class PipelineRunner:
    """负责增量/全量执行、预算调度与线程池并发。"""

    def __init__(self, state: PipelineState):
        self.state = state

    def _all_jobs(self) -> list[Job]:
        rows_per_layer = {
            layer: state_rows
            for layer, matrix in self.state.package.layers.items()
            for state_rows in [len(matrix)]
        }
        width_per_layer = {
            layer: self.state.row_width(layer)
            for layer in self.state.package.layers
        }
        upstream_per_layer = {
            layer: [
                item
                for row_index in range(rows_per_layer[layer])
                for item in self.state.upstream_rows(layer, row_index)
            ]
            for layer in self.state.package.layers
        }
        return stable_jobs(
            self.state.layer_order,
            rows_per_layer,
            width_per_layer,
            upstream_per_layer,
        )

    def execute(self, force_recompute: bool = False, parallel: bool = True) -> dict:
        """执行当前版本；force_recompute=True 时忽略缓存做全量重建。"""

        state = self.state
        state.ledger = ReuseLedger()
        # 预算挂起是单次运行的调度信号：仅保留非挂起失败，重新记录本运行挂起
        state.failures = _without_budget_pending(state.failures)
        jobs = self._all_jobs()
        pending_jobs = jobs
        batch_no = 0
        while pending_jobs:
            batch_no += 1
            plan = plan_batch(
                pending_jobs,
                state.config.element_budget,
                state.config.layer_row_quota,
                state.failures,
                source=f"batch-{batch_no}",
            )
            if not plan.ready and plan.pending:
                # 无进展时强制放行首个挂起作业，保证有限步内完成
                plan.ready = [plan.pending[0]]
                plan.pending = plan.pending[1:]
            # 规划前先把缓存中已重算行的新键传播到下游，确保同批失效判定正确
            self._refresh_all_upstream_keys()
            self._run_batch(plan.ready, force_recompute, parallel, batch_no)
            if not plan.pending:
                break
            pending_jobs = plan.pending
        return self.collect_artifacts(batch_no)

    def _run_batch(
        self, jobs: list[Job], force_recompute: bool, parallel: bool, batch_no: int
    ) -> None:
        state = self.state
        decided: dict[tuple[str, int], str] = {}
        reasons: dict[tuple[str, int], str] = {}

        def decide(job: Job) -> str:
            reason_id = (job.layer, job.row_index)
            entry = state.cache.get(
                (state.active_version, job.layer, job.row_index)
            )
            if force_recompute or entry is None:
                reasons[reason_id] = "force-full" if force_recompute else "cache-miss"
                return "RECOMPUTE"
            if self._is_fingerprint_hit(job, entry):
                reasons[reason_id] = "fingerprint-hit"
                return "REUSE"
            reasons[reason_id] = "fingerprint-miss"
            return "RECOMPUTE"

        for job in jobs:
            decided[(job.layer, job.row_index)] = decide(job)

        # 记录重算前各条目持有的上游键快照，用于识别同批上游变化
        upstream_snapshot = {
            (job.layer, job.row_index): dict(
                state.cache[(state.active_version, job.layer, job.row_index)].keys
            )
            for job in jobs
            if (state.active_version, job.layer, job.row_index) in state.cache
        }
        first_pass = [
            job for job in jobs if decided[(job.layer, job.row_index)] == "RECOMPUTE"
        ]
        self._execute_recompute(first_pass, parallel)

        # 复判：对比旧快照与当前上游键，变化则升级为重算（不先传播）
        for job in jobs:
            reason_id = (job.layer, job.row_index)
            if decided[reason_id] != "REUSE":
                continue
            live = state.upstream_rows(job.layer, job.row_index)
            current_fps = state._upstream_fingerprints(live)
            old_keys = upstream_snapshot[reason_id]
            changed = any(
                current_fps.get(ref) != old_keys.get(f"up:{ref[0]}:{ref[1]}")
                for ref in live
            )
            if changed:
                decided[reason_id] = "RECOMPUTE"
                reasons[reason_id] = "upstream-key-changed"
        second_pass = [
            job for job in jobs
            if decided[(job.layer, job.row_index)] == "RECOMPUTE"
            and job not in first_pass
        ]
        self._execute_recompute(second_pass, parallel)

        for job in jobs:
            reason_id = (job.layer, job.row_index)
            entry = state.cache.get(
                (state.active_version, job.layer, job.row_index)
            )
            key_str = entry.keys["softmax"] if entry is not None else "pending"
            state.ledger.record(
                LedgerDecision(
                    job.layer, job.row_index, state.active_version,
                    decided[reason_id], reasons[reason_id], key_str,
                    list(state.upstream_rows(job.layer, job.row_index)),
                )
            )

    def _execute_recompute(self, jobs: list[Job], parallel: bool) -> None:
        state = self.state
        if not jobs:
            return
        if parallel and len(jobs) > 1:
            workers = min(state.config.max_workers, len(jobs))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(lambda job: state.run_row(job.layer, job.row_index), jobs))
        else:
            for job in jobs:
                state.run_row(job.layer, job.row_index)
        self._propagate_from_jobs(jobs)

    def _is_fingerprint_hit(self, job: Job, entry: RowCacheEntry) -> bool:
        state = self.state
        gamma, beta, scheme, rounding = state.resolve_params(job.layer, job.row_index)
        current_act_fp = row_activation_fp(
            job.layer, job.row_index, state.package.layers[job.layer][job.row_index]
        )
        current_param_fp = param_fp(gamma, beta, scheme.value, rounding)
        current_cfg_fp = config_fp(
            state.config.frac_bits, state.config.tier, rounding, state.active_version
        )
        live_upstream = state.upstream_rows(job.layer, job.row_index)
        upstream_fps = state._upstream_fingerprints(live_upstream)
        conversion_key = entry.keys.get("version_conversion")
        return entry.matches(
            current_act_fp, current_param_fp, current_cfg_fp,
            upstream_fps, conversion_key,
        )

    def _refresh_all_upstream_keys(self) -> None:
        state = self.state
        current_keys = {
            (layer, row_index): entry.keys["softmax"]
            for (version, layer, row_index), entry in state.cache.items()
            if version == state.active_version
        }
        for (version, layer, row_index), entry in state.cache.items():
            if version != state.active_version:
                continue
            for ref in state.upstream_rows(layer, row_index):
                if ref in current_keys:
                    entry.keys[f"up:{ref[0]}:{ref[1]}"] = current_keys[ref]

    def _propagate_from_jobs(self, recomputed_jobs: list[Job]) -> None:
        """把本批重算行的新阶段键刷新到引用它的下游缓存条目。"""

        state = self.state
        changed = {
            (job.layer, job.row_index):
            state.cache[(state.active_version, job.layer, job.row_index)]
                .keys["softmax"]
            for job in recomputed_jobs
        }
        for (version, layer, row_index), entry in state.cache.items():
            if version != state.active_version:
                continue
            for ref in state.upstream_rows(layer, row_index):
                if ref in changed:
                    entry.keys[f"up:{ref[0]}:{ref[1]}"] = changed[ref]

    def collect_artifacts(self, batches: int) -> dict:
        state = self.state
        rows: list[dict] = []
        for layer in state.layer_order:
            count = len(state.package.layers[layer])
            for row_index in range(count):
                entry = state.cache.get((state.active_version, layer, row_index))
                if entry is not None:
                    rows.append(entry.result)
        return {
            "active_version": state.active_version,
            "batches": batches,
            "rows": rows,
            "ledger": state.ledger.to_dicts(),
            "ledger_summary": state.ledger.summary(),
            "failures": state.failures.to_dicts(),
            "conversion_records": list(state.conversion_records),
        }


def _without_budget_pending(failures: FailureList) -> FailureList:
    kept = FailureList()
    for record in failures:
        if record.code is not ErrorCode.E_BUDGET_PENDING:
            kept.add(record.code, record.source, record.message, record.context)
    return kept
