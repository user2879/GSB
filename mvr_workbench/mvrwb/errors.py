"""业务错误码定义。

所有错误均为确定性的、可映射到注入配置的业务码，写入诊断报告失败清单。
"""

from __future__ import annotations

# 输入视图 / 相机参数 / 配置校验类
E_VIEW_INVALID = "E_VIEW_INVALID"
# 全局重建点数预算耗尽
E_POINT_BUDGET_EXHAUSTED = "E_POINT_BUDGET_EXHAUSTED"
# 快照版本 / 配置哈希陈旧，不能续跑
E_SNAPSHOT_STALE = "E_SNAPSHOT_STALE"
# 三角化最小视差越界（注入故障用）
E_PARALLAX_OUT_OF_RANGE = "E_PARALLAX_OUT_OF_RANGE"
# 特征值非有限（注入故障用）
E_NON_FINITE_FEATURE = "E_NON_FINITE_FEATURE"
# 阶段执行通用失败（兜底，保留具体原因信息）
E_STAGE_FAILED = "E_STAGE_FAILED"

ALL_CODES = (
    E_VIEW_INVALID,
    E_POINT_BUDGET_EXHAUSTED,
    E_SNAPSHOT_STALE,
    E_PARALLAX_OUT_OF_RANGE,
    E_NON_FINITE_FEATURE,
    E_STAGE_FAILED,
)


class StageError(Exception):
    """携带业务码、阶段名、视图 / 视图对标识与注入项编号的阶段异常。"""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        stage: str | None = None,
        view_id: int | None = None,
        pair_id: str | None = None,
        injection_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.stage = stage
        self.view_id = view_id
        self.pair_id = pair_id
        self.injection_id = injection_id

    def to_failure(self) -> dict:
        record = {
            "code": self.code,
            "message": self.message,
            "stage": self.stage,
            "view_id": self.view_id,
            "pair_id": self.pair_id,
            "injection_id": self.injection_id,
        }
        return {key: value for key, value in record.items() if value is not None}
