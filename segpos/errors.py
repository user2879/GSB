"""业务错误码与异常类型。"""


class ComponentError(Exception):
    """组件业务异常基类。"""

    code = "E_COMPONENT"

    def __init__(self, message, code=None, stage=None, exception_type=None):
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        self.stage = stage
        self.exception_type = exception_type or type(self).__name__


class ConfigError(ComponentError):
    """输入配置或样本文件非法。"""

    code = "E_CONFIG_INVALID"


class ConfigMismatchError(ComponentError):
    """续跑快照与当前输入的版本/哈希不一致。"""

    code = "E_CONFIG_MISMATCH"


class StageError(ComponentError):
    """句子在某个处理阶段失败（异常隔离使用）。"""

    code = "E_STAGE_FAILED"

    def __init__(self, message, stage, code="E_STAGE_FAILED", exception_type=None):
        super().__init__(message, code=code, stage=stage, exception_type=exception_type)
        self.stage = stage
