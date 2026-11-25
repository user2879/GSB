"""分钟流归并：迟到批窗口归属、稳定去重与覆盖留痕。"""

from __future__ import annotations

from dataclasses import dataclass

from .models import MinuteBatch, MinutePoint


@dataclass(frozen=True)
class CoverageRecord:
    """重复分钟的择一采纳裁决（稳定规则，可重放）。"""

    service_id: str
    minute: int
    kept_line: int
    kept_batch_seq: int
    kept_stream: str
    covered_line: int
    covered_batch_seq: int
    covered_stream: str
    rule: str


# 流优先级：迟到/重发流后到，按“后到覆盖先到”稳定采纳；
# 同流内物理行号小者优先（先到先得）。被覆盖批次逐条留痕，可重放。
_STREAM_PRIORITY = {"regular": 1, "late": 0}


def merge_streams(
    regular: list[MinuteBatch],
    late: list[MinuteBatch],
) -> tuple[dict[str, list[MinutePoint]], list[CoverageRecord]]:
    """把常规流与迟到流归并为每服务唯一分钟序列。

    稳定择一规则（同一分钟重复到达）：
      1. 流优先级 regular < late；
      2. 同流内物理行号更小者优先；
      3. 行号相同则批次序号更小者优先。
    被覆盖批次逐条记录，输出按 (service_id, minute) 稳定排序。
    """
    chosen: dict[tuple[str, int], MinutePoint] = {}
    origins: dict[tuple[str, int], MinuteBatch] = {}
    coverage: list[CoverageRecord] = []

    def consider(batch: MinuteBatch) -> None:
        key = (batch.service_id, batch.minute)
        candidate = MinutePoint(
            minute=batch.minute,
            successes=batch.successes,
            failures=batch.failures,
            source_line=batch.line,
            source_batch_seq=batch.batch_seq,
            source_stream=batch.stream,
        )
        if key not in chosen:
            chosen[key] = candidate
            origins[key] = batch
            return
        incumbent_batch = origins[key]
        incumbent = chosen[key]
        candidate_rank = (
            _STREAM_PRIORITY.get(batch.stream, 99),
            batch.line,
            batch.batch_seq,
        )
        incumbent_rank = (
            _STREAM_PRIORITY.get(incumbent_batch.stream, 99),
            incumbent_batch.line,
            incumbent_batch.batch_seq,
        )
        if candidate_rank < incumbent_rank:
            coverage.append(
                CoverageRecord(
                    service_id=batch.service_id,
                    minute=batch.minute,
                    kept_line=candidate.source_line,
                    kept_batch_seq=candidate.source_batch_seq,
                    kept_stream=candidate.source_stream,
                    covered_line=incumbent.source_line,
                    covered_batch_seq=incumbent.source_batch_seq,
                    covered_stream=incumbent.source_stream,
                    rule="late-arrival-overrides:stream-priority,line,batch-seq",
                )
            )
            chosen[key] = candidate
            origins[key] = batch
        else:
            coverage.append(
                CoverageRecord(
                    service_id=batch.service_id,
                    minute=batch.minute,
                    kept_line=incumbent.source_line,
                    kept_batch_seq=incumbent.source_batch_seq,
                    kept_stream=incumbent.source_stream,
                    covered_line=candidate.source_line,
                    covered_batch_seq=candidate.source_batch_seq,
                    covered_stream=candidate.source_stream,
                    rule="late-arrival-overrides:stream-priority,line,batch-seq",
                )
            )

    for batch in sorted(regular, key=lambda b: (b.service_id, b.minute, b.line)):
        consider(batch)
    for batch in sorted(late, key=lambda b: (b.service_id, b.minute, b.line)):
        consider(batch)

    merged: dict[str, list[MinutePoint]] = {}
    for (service_id, minute), point in sorted(chosen.items()):
        merged.setdefault(service_id, []).append(point)
    for service_id in merged:
        merged[service_id].sort(key=lambda point: point.minute)
    coverage.sort(key=lambda record: (record.service_id, record.minute, record.covered_line))
    return merged, coverage
