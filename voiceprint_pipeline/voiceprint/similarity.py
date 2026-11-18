"""Fixed-point cosine similarity, threshold decisions and boundary detection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import PairRecord, PipelineConfig
from .embedding import EmbeddingResult
from .errors import BoundarySensitiveError
from .fixedpoint import FixedPoint, integer_cosine
from .util import fingerprint, sha256_text

SIM_Q_BITS = 30


@dataclass
class PairSimilarity:
    pair_id: str
    sample_a: str
    sample_b: str
    relation: str
    width: int
    similarity: float
    similarity_q30: int
    dot: int
    norm_a: int
    norm_b: int
    decision: str
    is_threshold_boundary: bool
    decision_basis: str
    pair_fp: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "sample_a": self.sample_a,
            "sample_b": self.sample_b,
            "relation": self.relation,
            "width": self.width,
            "similarity": self.similarity,
            "similarity_q30": self.similarity_q30,
            "dot": self.dot,
            "norm_a": self.norm_a,
            "norm_b": self.norm_b,
            "decision": self.decision,
            "is_threshold_boundary": self.is_threshold_boundary,
            "decision_basis": self.decision_basis,
            "pair_fp": self.pair_fp,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PairSimilarity":
        return cls(
            pair_id=payload["pair_id"],
            sample_a=payload["sample_a"],
            sample_b=payload["sample_b"],
            relation=payload["relation"],
            width=int(payload["width"]),
            similarity=float(payload["similarity"]),
            similarity_q30=int(payload["similarity_q30"]),
            dot=int(payload["dot"]),
            norm_a=int(payload["norm_a"]),
            norm_b=int(payload["norm_b"]),
            decision=payload["decision"],
            is_threshold_boundary=bool(payload["is_threshold_boundary"]),
            decision_basis=payload["decision_basis"],
            pair_fp=payload["pair_fp"],
        )


def compare_pair(pair: PairRecord, emb_a: EmbeddingResult, emb_b: EmbeddingResult,
                 config: PipelineConfig, width: int) -> PairSimilarity:
    if width != emb_a.width or width != emb_b.width:
        raise ValueError("embedding widths must match the requested similarity width")
    result = integer_cosine(emb_a.embedding_int, emb_b.embedding_int, width)
    threshold = config.similarity_threshold
    sim = result["similarity"]
    boundary = abs(sim - threshold) <= 0.5 / (1 << SIM_Q_BITS)
    decision, basis = _decide(sim, threshold, config.tie_rule, boundary)
    pair_fp = sha256_text(
        fingerprint(
            {
                "pair_id": pair.pair_id,
                "emb_fp_a": emb_a.embedding_fp,
                "emb_fp_b": emb_b.embedding_fp,
                "similarity_q30": result["similarity_q30"],
                "tie_rule": config.tie_rule,
                "threshold": threshold,
                "width": width,
            }
        )
    )
    return PairSimilarity(
        pair_id=pair.pair_id,
        sample_a=pair.sample_a,
        sample_b=pair.sample_b,
        relation=pair.relation,
        width=width,
        similarity=sim,
        similarity_q30=result["similarity_q30"],
        dot=result["dot"],
        norm_a=result["norm_a"],
        norm_b=result["norm_b"],
        decision=decision,
        is_threshold_boundary=boundary,
        decision_basis=basis,
        pair_fp=pair_fp,
    )


def _decide(sim: float, threshold: float, tie_rule: str, boundary: bool) -> tuple[str, str]:
    if boundary:
        decision = "same_speaker" if tie_rule == "same_speaker" else "different_speaker"
        basis = (
            f"similarity exactly equals threshold {threshold}; tie_rule={tie_rule} -> {decision}"
        )
        return decision, basis
    if sim > threshold:
        return "same_speaker", f"similarity {sim} > threshold {threshold}"
    return "different_speaker", f"similarity {sim} < threshold {threshold}"


def check_boundary_sensitivity(base: PairSimilarity, ref: PairSimilarity) -> dict[str, Any] | None:
    """Flag pairs whose decision flips between the configured and higher width."""
    if base.decision != ref.decision:
        exc = BoundarySensitiveError(
            f"pair {base.pair_id} decision flips: width {base.width} -> {base.decision}, "
            f"width {ref.width} -> {ref.decision}"
        )
        return {
            "pair_id": base.pair_id,
            "code": exc.code,
            "width_base": base.width,
            "width_ref": ref.width,
            "decision_base": base.decision,
            "decision_ref": ref.decision,
            "similarity_base": base.similarity,
            "similarity_ref": ref.similarity,
            "message": str(exc),
        }
    return None


def threshold_q30(config: PipelineConfig) -> int:
    fp = FixedPoint(SIM_Q_BITS)
    return int(fp.quantize(np.array(config.similarity_threshold, dtype=np.float64)))
