"""业务错误码与统一异常。

所有可预期的输入/运行期错误都抛出 :class:`RenderError`，
CLI 层负责把它序列化为带 ``E_xxx`` 业务码的 JSON 错误文档。
"""
from __future__ import annotations

import json
from typing import Any, Mapping


class RenderError(Exception):
    """携带业务码的引擎错误。"""

    def __init__(self, code: str, message: str, detail: Mapping[str, Any] | None = None):
        if not code.startswith("E_"):
            raise ValueError("业务错误码必须以 E_ 开头")
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = dict(detail or {})

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"ok": False, "error_code": self.code, "message": self.message}
        if self.detail:
            payload["detail"] = self.detail
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)


# 校验阶段（E1xxx）
E_JSON_PARSE = "E_1001"          # JSON 无法解析
E_SCHEMA = "E_1002"              # 结构缺失/类型不符
E_BOX = "E_1003"                 # 介质盒非法
E_LIGHT = "E_1004"               # 光源非法
E_CAMERA = "E_1005"              # 相机非法
E_GRID = "E_1006"                # 渲染/分块参数非法
E_PATH = "E_1007"                # 路径不可读写

# 几何/渲染阶段（E2xxx）
E_INTERSECT = "E_2001"           # 求交失败
E_RENDER = "E_2002"              # 像素渲染异常
E_MEMORY = "E_2003"              # 单块需求即超过内存上限
E_TILE = "E_2004"                # 分块运行期异常

# 持久化/IO 阶段（E3xxx）
E_STATE = "E_3001"               # 状态文件损坏
E_IO = "E_3002"                  # 原子写出失败

# 中断（E4xxx，属于约定控制流而非缺陷）
E_INTERRUPTED = "E_4001"

# 约定退出码
EXIT_OK = 0
EXIT_INTERRUPTED = 130
EXIT_ERROR = 1
