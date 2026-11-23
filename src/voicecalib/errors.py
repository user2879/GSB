"""业务错误码与失败清单记录。

所有违规项都映射到稳定的业务码，收集进失败清单后继续处理其余合规项，
单轮失败不会中断整个会话，也不会污染其它轮次的结果。
"""
from __future__ import annotations

from dataclasses import dataclass, field


class ErrorCode:
    E_AUDIO_FORMAT = "E_AUDIO_FORMAT"
    E_AUDIO_MISSING = "E_AUDIO_MISSING"
    E_SCRIPT_STRUCTURE = "E_SCRIPT_STRUCTURE"
    E_SPEAKER_EMPTY = "E_SPEAKER_EMPTY"
    E_EVIDENCE_FORMAT = "E_EVIDENCE_FORMAT"
    E_EVIDENCE_NONFINITE = "E_EVIDENCE_NONFINITE"
    E_CONFIG_VALUE = "E_CONFIG_VALUE"
    E_INJECTION_DIM_MISMATCH = "E_INJECTION_DIM_MISMATCH"
    E_NONFINITE_SAMPLE = "E_NONFINITE_SAMPLE"
    E_ZERO_GAIN = "E_ZERO_GAIN"
    E_ROUND_OUT_OF_RANGE = "E_ROUND_OUT_OF_RANGE"
    E_CALIB_BUDGET_PENDING = "E_CALIB_BUDGET_PENDING"
    E_UPSTREAM_BLOCKED = "E_UPSTREAM_BLOCKED"
    E_SNAPSHOT_STALE = "E_SNAPSHOT_STALE"


@dataclass
class ErrorRecord:
    code: str
    stage: str
    message: str
    round: int | None = None
    file: str | None = None
    injection_id: str | None = None

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "stage": self.stage,
            "message": self.message,
            "round": self.round,
            "file": self.file,
            "injection_id": self.injection_id,
        }


@dataclass
class ErrorCollector:
    records: list[ErrorRecord] = field(default_factory=list)

    def add(self, code: str, stage: str, message: str, round: int | None = None,
            file: str | None = None, injection_id: str | None = None) -> ErrorRecord:
        record = ErrorRecord(code, stage, message, round, file, injection_id)
        self.records.append(record)
        return record

    def codes(self) -> list[str]:
        return [record.code for record in self.records]

    def sorted_dict_list(self) -> list[dict]:
        # 失败清单固定按 (轮次, 阶段, 业务码, 文件) 排序，与线程调度顺序无关。
        ordered = sorted(
            self.records,
            key=lambda r: (
                r.round if r.round is not None else -1,
                r.stage,
                r.code,
                r.file or "",
                r.injection_id or "",
            ),
        )
        return [record.to_dict() for record in ordered]

    def has(self, code: str) -> bool:
        return any(record.code == code for record in self.records)
