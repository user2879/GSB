"""跨版本回执归并与灰度结论合成。

采纳规则（稳定、与遍历顺序无关）：
    参数版本序号降序 -> 格式版本降序 -> 归并序号升序。
被否决回执保留为对照并记录否决理由。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .receipts import FORMAT_RANK, NormalizedReceipt


@dataclass
class MergeCandidate:
    receipt: NormalizedReceipt
    merge_seq: int


@dataclass
class RejectedReceipt:
    merge_seq: int
    source_format: str
    param_version: int
    reason: str


@dataclass
class GrayConclusion:
    batch_id: str
    route: str
    sample_ids: tuple[str, ...]
    param_digest: str
    param_version: int
    source_formats: tuple[str, ...]
    adopted_seq: int
    adopted_format: str
    rejected: list[RejectedReceipt] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "batch_id": self.batch_id,
            "route": self.route,
            "sample_ids": list(self.sample_ids),
            "param_digest": self.param_digest,
            "param_version": self.param_version,
            "source_formats": list(self.source_formats),
            "adopted_seq": self.adopted_seq,
            "adopted_format": self.adopted_format,
            "rejected": [
                {
                    "merge_seq": r.merge_seq,
                    "source_format": r.source_format,
                    "param_version": r.param_version,
                    "reason": r.reason,
                }
                for r in self.rejected
            ],
        }

    @staticmethod
    def from_dict(data: dict) -> "GrayConclusion":
        return GrayConclusion(
            batch_id=data["batch_id"],
            route=data["route"],
            sample_ids=tuple(data["sample_ids"]),
            param_digest=data["param_digest"],
            param_version=data["param_version"],
            source_formats=tuple(data["source_formats"]),
            adopted_seq=data["adopted_seq"],
            adopted_format=data["adopted_format"],
            rejected=[
                RejectedReceipt(
                    merge_seq=r["merge_seq"],
                    source_format=r["source_format"],
                    param_version=r["param_version"],
                    reason=r["reason"],
                )
                for r in data.get("rejected", [])
            ],
        )


def _adoption_key(candidate: MergeCandidate):
    receipt = candidate.receipt
    return (
        -receipt.param_version,
        -FORMAT_RANK[receipt.source_format],
        candidate.merge_seq,
    )


def _reject_reason(candidate: MergeCandidate, adopted: MergeCandidate) -> str:
    receipt, winner = candidate.receipt, adopted.receipt
    if receipt.param_version != winner.param_version:
        return (
            f"param_version {receipt.param_version} 低于采纳值 "
            f"{winner.param_version}"
        )
    if FORMAT_RANK[receipt.source_format] != FORMAT_RANK[winner.source_format]:
        return (
            f"format_version {receipt.source_format} 低于采纳值 "
            f"{winner.source_format}"
        )
    return f"merge_seq {candidate.merge_seq} 晚于采纳序号 {adopted.merge_seq}"


def merge_receipts(candidates: list[MergeCandidate]) -> GrayConclusion:
    if not candidates:
        raise ValueError("归并输入不能为空")
    ordered = sorted(candidates, key=_adoption_key)
    adopted = ordered[0]
    rejected = [
        RejectedReceipt(
            merge_seq=c.merge_seq,
            source_format=c.receipt.source_format,
            param_version=c.receipt.param_version,
            reason=_reject_reason(c, adopted),
        )
        for c in ordered[1:]
    ]
    source_formats = tuple(
        sorted({c.receipt.source_format for c in candidates}, key=FORMAT_RANK.get)
    )
    winner = adopted.receipt
    return GrayConclusion(
        batch_id=winner.batch_id,
        route=winner.route,
        sample_ids=winner.sample_ids,
        param_digest=winner.param_digest,
        param_version=winner.param_version,
        source_formats=source_formats,
        adopted_seq=adopted.merge_seq,
        adopted_format=winner.source_format,
        rejected=rejected,
    )
