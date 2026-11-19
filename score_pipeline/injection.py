"""异常注入配置。

注入配置本身是可序列化 JSON，随运行记录落盘，保证同配置可精确复现。
参数层异常由各模块直接抛 PipelineError；提交中断通过 HardExit 表达，
CLI 工作进程捕获后以 os._exit(137) 模拟强制终止（Windows 同样适用）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .errors import ERROR_CODES
from .model import canonical_bytes, content_hash, read_json, write_json

# 支持的注入点（全部为确定性、可复现的业务异常）
FAULT_TEMPO_NON_MONOTONIC = "tempo_non_monotonic"
FAULT_NOTE_NON_POSITIVE = "note_non_positive"
FAULT_PEDAL_OVERLAP = "pedal_overlap"
FAULT_VERSION_REGRESS = "version_regress"
FAULT_SNAPSHOT_STALE = "snapshot_stale"
FAULT_TIMELINE_HASH = "timeline_hash_corrupt"
FAULT_COMMIT_CRASH_AFTER_BLOB = "commit_crash_after_blob"
FAULT_COMMIT_CRASH_AFTER_JOURNAL = "commit_crash_after_journal"

FAULT_TO_CODE = {
    FAULT_TEMPO_NON_MONOTONIC: "E_VELOCITY_MONO",
    FAULT_NOTE_NON_POSITIVE: "E_BEAT_RANGE",
    FAULT_PEDAL_OVERLAP: "E_PEDAL_OVERLAP",
    FAULT_VERSION_REGRESS: "E_VERSION_REGRESS",
    FAULT_SNAPSHOT_STALE: "E_SNAPSHOT_STALE",
    FAULT_TIMELINE_HASH: "E_TIMELINE_HASH",
}


class HardExit(BaseException):
    """模拟强制终止：CLI 层捕获后调用 os._exit，不执行任何 finally 落盘。"""

    def __init__(self, point: str, exit_code: int = 137):
        super().__init__(point)
        self.point = point
        self.exit_code = exit_code


@dataclass
class InjectionConfig:
    # 参数层故障：fault_name -> 在第几次调用触发（1 基）；0 表示不启用
    param_faults: dict = field(default_factory=dict)
    # 提交中断点：在指定 op 版本提交时触发
    crash_at_version: int = 0
    crash_point: str = ""

    def to_dict(self) -> dict:
        return {
            "schema": "injection-config/1",
            "param_faults": dict(sorted(self.param_faults.items())),
            "crash_at_version": self.crash_at_version,
            "crash_point": self.crash_point,
            "config_sha256": content_hash(
                {
                    "param_faults": dict(sorted(self.param_faults.items())),
                    "crash_at_version": self.crash_at_version,
                    "crash_point": self.crash_point,
                }
            ),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "InjectionConfig":
        return cls(
            param_faults=dict(data.get("param_faults", {})),
            crash_at_version=int(data.get("crash_at_version", 0)),
            crash_point=data.get("crash_point", ""),
        )

    def save(self, path: str) -> None:
        write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: str) -> "InjectionConfig":
        return cls.from_dict(read_json(path))

    def check_commit_crash(self, version: int) -> None:
        if self.crash_at_version == version and self.crash_point:
            raise HardExit(self.crash_point)


def build_faulty_payload(fault: str, score, base_tempo, base_dynamics, base_pedals):
    """按故障类型构造参数层异常输入，返回 (kind, tempo, dynamics, pedals)。"""
    if fault == FAULT_TEMPO_NON_MONOTONIC:
        tempo = [dict(point) for point in base_tempo]
        tempo[1]["beat"] = tempo[0]["beat"]  # 拍号非严格递增
        return "tempo", tempo, base_dynamics, base_pedals
    if fault == FAULT_NOTE_NON_POSITIVE:
        # 直接破坏内存中的乐谱副本（通过引擎校验链路）
        from .model import Note

        first_part = sorted(score.parts)[0]
        notes = list(score.parts[first_part])
        notes[0] = Note(
            note_id=notes[0].note_id,
            pitch=notes[0].pitch,
            start_beat=notes[0].start_beat,
            duration_beats=0.0,
            velocity=notes[0].velocity,
            articulation=notes[0].articulation,
        )
        score.parts[first_part] = notes
        return "dynamics", base_tempo, base_dynamics, base_pedals
    if fault == FAULT_PEDAL_OVERLAP:
        pedals = [dict(pedal) for pedal in base_pedals]
        first_part = sorted({pedal["part_id"] for pedal in pedals})[0]
        part_pedals = [p for p in pedals if p["part_id"] == first_part]
        part_pedals[1]["start_beat"] = part_pedals[0]["start_beat"] + 0.5
        return "pedals", base_tempo, base_dynamics, pedals
    raise ValueError(f"unsupported param fault: {fault}")


def assert_known_codes() -> None:
    for code in FAULT_TO_CODE.values():
        assert code in ERROR_CODES
