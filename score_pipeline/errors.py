"""业务错误码定义。

所有业务异常都带稳定错误码，经 JSON 字段或 stderr 输出，
注入配置与运行记录中只引用错误码字符串，保证可复现。
"""

from __future__ import annotations

E_BEAT_RANGE = "E_BEAT_RANGE"
E_VELOCITY_MONO = "E_VELOCITY_MONO"
E_PEDAL_OVERLAP = "E_PEDAL_OVERLAP"
E_VERSION_REGRESS = "E_VERSION_REGRESS"
E_SNAPSHOT_STALE = "E_SNAPSHOT_STALE"
E_TIMELINE_HASH = "E_TIMELINE_HASH"

ERROR_CODES = (
    E_BEAT_RANGE,
    E_VELOCITY_MONO,
    E_PEDAL_OVERLAP,
    E_VERSION_REGRESS,
    E_SNAPSHOT_STALE,
    E_TIMELINE_HASH,
)

# 错误码语义说明（同时写入报告，避免名称歧义）
ERROR_CODE_MEANING = {
    E_BEAT_RANGE: "拍点越界或音符时值非正（start/end 拍超出 [0,total_beats]，duration<=0）",
    E_VELOCITY_MONO: "序列单调性冲突：速度曲线关键点拍号必须严格递增",
    E_PEDAL_OVERLAP: "同一声部踏板区间交叉或重叠",
    E_VERSION_REGRESS: "提交版本戳回退：新版本必须等于当前版本+1",
    E_SNAPSHOT_STALE: "快照版本戳落后于当前稳定状态或内容哈希不匹配，禁止覆盖",
    E_TIMELINE_HASH: "时间轴分片内容哈希与清单不一致，持久化数据被破坏或提交不完整",
}


class PipelineError(Exception):
    """带稳定业务码的管线异常。"""

    def __init__(self, code: str, message: str, detail: dict | None = None):
        if code not in ERROR_CODES:
            raise ValueError(f"unknown error code: {code}")
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail or {}

    def to_record(self) -> dict:
        return {
            "ok": False,
            "code": self.code,
            "meaning": ERROR_CODE_MEANING[self.code],
            "message": self.message,
            "detail": self.detail,
        }
