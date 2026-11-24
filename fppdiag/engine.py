"""Streaming, block-wise execution of the fixed-point feature pipeline.

Two scheduling strategies are supported to prove block/order independence:
* ``stage_major``  - fully finish one stage over all blocks before the next;
* ``pipeline_major`` - advance a sliding window of blocks through all stages.

Both consume the same deterministic source columns and the same block
boundaries, so every intermediate and final value is byte-identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import fixedpoint as fp
from . import stages as st
from .errors import ParamError
from .spec import PipelineSpec, StageSpec
from .sources import load_or_generate
from .util import content_digest


@dataclass
class StageRun:
    spec: StageSpec
    scale_exp: int
    blocks: list[st.BlockResult] = field(default_factory=list)

    def all_values(self) -> list[int | None]:
        out: list[int | None] = []
        for blk in self.blocks:
            out.extend(blk.values)
        return out

    def digest(self) -> str:
        payload = {"stage": self.spec.stage_id, "scale_exp": self.scale_exp,
                   "values": self.all_values(),
                   "saturated": [b for blk in self.blocks for b in blk.saturated],
                   "boundary_uncertain": [b for blk in self.blocks
                                          for b in blk.boundary_uncertain],
                   "nonpos_skipped": [b for blk in self.blocks
                                      for b in blk.nonpos_skipped]}
        return content_digest(payload)

    def summary(self) -> dict:
        vals = self.all_values()
        sat = sum(b for blk in self.blocks for b in blk.saturated)
        unc = sum(b for blk in self.blocks for b in blk.boundary_uncertain)
        skip = sum(b for blk in self.blocks for b in blk.nonpos_skipped)
        nulls = sum(v is None for v in vals)
        return {"stage_id": self.spec.stage_id, "type": self.spec.stype,
                "scale_exp": self.scale_exp, "rows": len(vals),
                "nulls": nulls, "saturated": sat,
                "boundary_uncertain": unc, "nonpos_skipped": skip,
                "digest": self.digest()}


def block_ranges(total: int, block_size: int) -> list[range]:
    return [range(i, min(i + block_size, total)) for i in range(0, total, block_size)]


def _validate_dependency_scales(spec) -> None:
    """Fail fast (E_PARAM) when a product joins differently scaled inputs."""
    by_id = {s.stage_id: s for s in spec.stages}
    for stage in spec.stages:
        if stage.stype == "product":
            scales = {by_id[r].scale_exp for r in stage.refs}
            if len(scales) != 1:
                raise ParamError(
                    "product inputs must share a scale exponent",
                    location=stage.stage_id, field="refs")
        if stage.stype == "log":
            if stage.params.get("base_flag") not in (0, 1, 2):
                raise ParamError("log base_flag must be 0(e),1(2) or 2(10)",
                                 location=stage.stage_id,
                                 field="params.base_flag")


def _validate_stage_params(stage: StageSpec) -> None:
    p = stage.params
    if stage.stype == "standardize":
        if p.get("sd", 0) <= 0:
            raise ParamError("standardize sd must be strictly positive",
                             location=stage.stage_id, field="params.sd")
        for key in ("mean", "sd"):
            if not (fp.INT32_MIN <= p[key] <= fp.INT32_MAX):
                raise ParamError(f"standardize {key} outside 32-bit envelope",
                                 location=stage.stage_id, field=f"params.{key}")
        if p.get("sd_scale_exp") not in fp.ALLOWED_SCALE_EXPS:
            raise ParamError("standardize sd_scale_exp not allowed",
                             location=stage.stage_id, field="params.sd_scale_exp")
    if stage.stype == "truncate" and p["upper"] < p["lower"]:
        raise ParamError("truncate upper must be >= lower",
                         location=stage.stage_id, field="params.upper")
    if stage.stype == "bucket":
        edges = p.get("edges", [])
        if not edges or any(edges[i] >= edges[i + 1] for i in range(len(edges) - 1)):
            raise ParamError("bucket edges must be a non-empty strictly increasing list",
                             location=stage.stage_id, field="params.edges")


class PipelineEngine:
    def __init__(self, spec: PipelineSpec, workdir: str):
        self.spec = spec
        self.workdir = workdir
        self.columns: dict[str, list[int | None]] = {}
        self.runs: dict[str, StageRun] = {}
        self.rows = next(iter(spec.sources.values())).rows
        self.ranges = block_ranges(self.rows, spec.block_size)
        for stage in spec.stages:
            _validate_stage_params(stage)
        _validate_dependency_scales(spec)
        # Provisional bounds (zero observed magnitudes) drive bucket boundary
        # uncertainty during the value pass; refreshed after values exist.
        from .errorchain import build_error_chains
        self.error_bounds = build_error_chains(spec, self)

    # -- source loading ----------------------------------------------------
    def load_sources(self) -> None:
        for sid, sspec in self.spec.sources.items():
            col = load_or_generate(self.workdir, sspec)
            if len(col) != sspec.rows:
                from .errors import DataError
                raise DataError(f"source {sid} row count mismatch",
                                location=sid, field="rows")
            self.columns[sid] = col

    def _input_for(self, stage: StageSpec, rng: range):
        if stage.source_id is not None:
            col = self.columns[stage.source_id]
            in_exp = self.spec.sources[stage.source_id].ingest_scale
            return [col[i] for i in rng], in_exp
        upstream = self.runs[stage.refs[0]]
        return [upstream.blocks[self._block_index(rng)].values[k]
                for k in range(len(rng))], upstream.scale_exp

    def _block_index(self, rng: range) -> int:
        for idx, r in enumerate(self.ranges):
            if r.start == rng.start:
                return idx
        raise KeyError(rng)

    def _run_stage_block(self, stage: StageSpec, rng: range) -> st.BlockResult:
        if stage.stype in ("standardize", "log"):
            vals, in_exp = self._input_for(stage, rng)
            if stage.stype == "standardize":
                return st.run_standardize(stage, vals, in_exp)
            return st.run_log(stage, vals, in_exp)
        if stage.stype == "bucket":
            vals, in_exp = self._input_for(stage, rng)
            bound = self.error_bounds[stage.refs[0]]["bound_ulp"] if stage.refs else 0
            return st.run_bucket(stage, vals, in_exp, bound)
        if stage.stype == "truncate":
            vals, in_exp = self._input_for(stage, rng)
            return st.run_truncate(stage, vals, in_exp)
        # product: two upstream refs at equal scale (validated below)
        left_run = self.runs[stage.refs[0]]
        right_run = self.runs[stage.refs[1]]
        bi = self._block_index(rng)
        left = left_run.blocks[bi].values
        right = right_run.blocks[bi].values
        if left_run.scale_exp != right_run.scale_exp:
            raise ParamError("product inputs must share a scale exponent",
                             location=stage.stage_id, field="refs")
        return st.run_product(stage, left, right, left_run.scale_exp)

    def run_stage(self, stage: StageSpec) -> StageRun:
        run = StageRun(stage, stage.scale_exp)
        for rng in self.ranges:
            run.blocks.append(self._run_stage_block(stage, rng))
        self.runs[stage.stage_id] = run
        return run

    def execute(self, schedule: str = "stage_major") -> dict[str, StageRun]:
        """Pass 1 compute integer values; pass 2 fold magnitudes into bounds."""
        self.load_sources()
        order = [s.stage_id for s in self.spec.stages]
        if schedule == "stage_major":
            for stage in self.spec.stages:
                self.run_stage(stage)
        elif schedule == "pipeline_major":
            pending = list(self.spec.stages)
            done: set[str] = set()
            while pending:
                progressed = False
                for stage in list(pending):
                    if all(r in done or r in self.columns for r in stage.refs):
                        self.run_stage(stage)
                        done.add(stage.stage_id)
                        pending.remove(stage)
                        progressed = True
                if not progressed:
                    raise RuntimeError("pipeline_major scheduling deadlock")
        else:
            raise ValueError(f"unknown schedule {schedule!r}")
        assert list(self.runs) == order
        from .errorchain import build_error_chains
        self.error_bounds = build_error_chains(self.spec, self)
        return self.runs
