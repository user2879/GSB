"""业务码定义与阶段异常类型。

所有校验/处理失败均以业务码形式进入失败清单，绝不中断整体加载或运行。
"""
from __future__ import annotations

# ---- 加载校验阶段 ----
E_TILE_DUPLICATE = "E_TILE_DUPLICATE"                 # 瓦片 ID 不唯一
E_TILE_SENSOR_UNKNOWN = "E_TILE_SENSOR_UNKNOWN"       # 传感器不在定标表内
E_TILE_BAND_UNDECLARED = "E_TILE_BAND_UNDECLARED"     # 波段号未在该传感器声明
E_PIXEL_NEGATIVE = "E_PIXEL_NEGATIVE"                 # 像素计数为负
E_PIXEL_NOT_INTEGER = "E_PIXEL_NOT_INTEGER"           # 像素计数非整数
E_PIXEL_OVERFLOW = "E_PIXEL_OVERFLOW"                 # 计数超出量化位宽上限
E_TIMESTAMP_NOT_ASCENDING = "E_TIMESTAMP_NOT_ASCENDING"  # 时间戳未严格递增
E_GAIN_INVALID = "E_GAIN_INVALID"                     # 增益非正或非有限
E_OFFSET_INVALID = "E_OFFSET_INVALID"                 # 偏移非有限
E_REG_SOURCE_EMPTY = "E_REG_SOURCE_EMPTY"             # 回归源白名单为空
E_REG_SOURCE_UNKNOWN = "E_REG_SOURCE_UNKNOWN"         # 回归引用传感器不存在
E_BACKFILL_TIMESTAMP_ORDER = "E_BACKFILL_TIMESTAMP_ORDER"  # 回填时间戳早于游标

# ---- 定标阶段 ----
E_CALIB_FAILED = "E_CALIB_FAILED"                     # 定标计算异常
E_CALIB_BUDGET_PENDING = "E_CALIB_BUDGET_PENDING"     # 预算不足待办
E_FIXEDPOINT_OVERFLOW = "E_FIXEDPOINT_OVERFLOW"       # 定点位宽越界

# ---- 回归阶段 ----
E_REGRESS_FAILED = "E_REGRESS_FAILED"                 # 回归求解/对齐异常

# ---- 快照 ----
E_SNAPSHOT_STALE = "E_SNAPSHOT_STALE"                 # 快照版本/配置哈希不一致

STAGE_LOAD = "load"
STAGE_CALIB = "calib"
STAGE_REGRESS = "regress"


class StageError(Exception):
    """携带业务码、瓦片、波段与阶段名的受控异常。"""

    def __init__(self, code: str, message: str, *, tile_id: str | None = None,
                 band: int | None = None, stage: str = STAGE_LOAD,
                 sensor_id: str | None = None, injection_id: str | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.tile_id = tile_id
        self.band = band
        self.stage = stage
        self.sensor_id = sensor_id
        self.injection_id = injection_id

    def to_failure(self) -> dict:
        rec = {
            "code": self.code,
            "stage": self.stage,
            "message": self.message,
        }
        if self.tile_id is not None:
            rec["tile_id"] = self.tile_id
        if self.band is not None:
            rec["band"] = self.band
        if self.sensor_id is not None:
            rec["sensor_id"] = self.sensor_id
        if self.injection_id is not None:
            rec["injection_id"] = self.injection_id
        return rec
