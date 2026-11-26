"""Orchestration: per-step processing, caching and exchange construction."""

from __future__ import annotations

import threading
from dataclasses import dataclass

from .cache import ConclusionCache
from .config import DecodeConfig
from .errors import BizCode, BizError
from .exchange import ExchangeDocument, ExchangeRecord
from .kernel import run_kernel
from .models import Request, CandidateSequence, Step, stable_task_key
from .sampling import SamplingConclusion, sample
from .truncation import truncate


@dataclass
class StepOutcome:
    request_id: str
    seq_id: int
    step_index: int
    task_key: str
    conclusion: SamplingConclusion
    boundary_dicts: list[dict]
    record_dicts: list[dict]
    request_no: int
    from_cache: bool


class DecodePipeline:
    def __init__(self, config: DecodeConfig, cache: ConclusionCache | None = None,
                 request_order: dict[str, int] | None = None):
        self.config = config
        self.cache = cache if cache is not None else ConclusionCache()
        self.request_order = dict(request_order or {})
        self._lock = threading.Lock()

    def request_no(self, request_id: str) -> int:
        return self.request_order.get(request_id,
                                     hash_ord(request_id))

    def process_step(self, request: Request, seq: CandidateSequence,
                     step: Step) -> StepOutcome:
        with self._lock:
            cached = self.cache.get(request, seq.seq_id, step.index, self.config)
        if cached is not None:
            return self._outcome_from_cache(request, seq, step, cached)

        kr = run_kernel(step.sorted_tokens(),
                        [step.logprobs[t] for t in step.sorted_tokens()],
                        seq.temperature, self.config)
        tr = truncate(kr, self.config)
        step_fp = seq.step_fingerprint(step.index)
        conclusion = sample(
            request.request_id, seq.seq_id, step.index,
            step_fp, self.config.fingerprint(), kr, tr)
        with self._lock:
            self.cache.put(request, seq.seq_id, step.index, self.config,
                           conclusion)

        boundary_dicts = [b.__dict__ for b in tr.boundaries]
        record_dicts = [r.__dict__ for r in kr.records]
        return StepOutcome(
            request_id=request.request_id, seq_id=seq.seq_id,
            step_index=step.index,
            task_key=stable_task_key(request.request_id, seq.seq_id,
                                     step.index),
            conclusion=conclusion, boundary_dicts=boundary_dicts,
            record_dicts=record_dicts,
            request_no=self.request_no(request.request_id), from_cache=False)

    def _outcome_from_cache(self, request, seq, step, conclusion) -> StepOutcome:
        return StepOutcome(
            request_id=request.request_id, seq_id=seq.seq_id,
            step_index=step.index,
            task_key=stable_task_key(request.request_id, seq.seq_id,
                                     step.index),
            conclusion=conclusion, boundary_dicts=[], record_dicts=[],
            request_no=self.request_no(request.request_id), from_cache=True)

    def build_exchange_document(self, outcomes: list[StepOutcome],
                                token_names: dict[int, str] | None = None
                                ) -> ExchangeDocument:
        token_names = token_names or {}
        records: list[ExchangeRecord] = []
        for outcome in sorted(outcomes, key=lambda o: o.task_key):
            conclusion = outcome.conclusion
            for token in conclusion.kept_tokens:
                # Working integer -> per-mille for the v1-compatible field.
                working = conclusion.kept_quant[token]
                mille = round(working * 1000 / self.config.scale)
                residual = conclusion.residuals.get(token, 0)
                records.append(ExchangeRecord(
                    request_no=outcome.request_no,
                    request_id=outcome.request_id,
                    seq_id=outcome.seq_id,
                    step_index=outcome.step_index,
                    token=token,
                    mille_prob=mille,
                    token_name=token_names.get(token),
                    quant_residual=residual,
                    ref_int=conclusion.ref_quant.get(token),
                    covered_ref=f"residual:{residual}" if residual != 0 else None,
                    precision_label=f"W{self.config.width}"))
        return ExchangeDocument(
            version="v2", width=self.config.width, tier=self.config.tier,
            config_fingerprint=self.config.fingerprint(),
            request_index=dict(self.request_order),
            token_names=token_names, records=records)


def hash_ord(request_id: str) -> int:
    import hashlib
    return int(hashlib.sha256(request_id.encode("utf-8")).hexdigest()[:6], 16)
