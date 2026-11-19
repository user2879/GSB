"""业务错误码定义与带序号的解析异常。"""

# 解析层
E_PARSE = "E_PARSE"
# 路径穿越（父目录回退 / 绝对路径越出受检根）
E_TRAVERSAL = "E_TRAVERSAL"
# 链接跳转后落点越出受检根
E_LINK_ESCAPE = "E_LINK_ESCAPE"
# 拒绝项被同级或更近的允许项覆盖
E_DENY_SHADOW = "E_DENY_SHADOW"
# 继承传播使子路径获得父路径未授权权限
E_INHERIT = "E_INHERIT"
# 不同规范化策略下结论冲突
E_POLICY_CONFLICT = "E_POLICY_CONFLICT"
# 单文件墙钟超时
E_TIMEOUT = "E_TIMEOUT"
# 单文件记录数超预算
E_BUDGET = "E_BUDGET"
# 检查点并发写锁冲突
E_LOCK = "E_LOCK"
# 检查点版本不匹配（触发整体重建，不作为记录级违规）
E_CHECKPOINT_VERSION = "E_CHECKPOINT_VERSION"
# 真值对照不一致
E_TRUTH_MISMATCH = "E_TRUTH_MISMATCH"

ALL_RECORD_CODES = (
    E_PARSE,
    E_TRAVERSAL,
    E_LINK_ESCAPE,
    E_DENY_SHADOW,
    E_INHERIT,
    E_POLICY_CONFLICT,
    E_TIMEOUT,
    E_BUDGET,
    E_LOCK,
    E_TRUTH_MISMATCH,
)


class AuditError(Exception):
    """带业务错误码的异常基类。"""

    code = E_PARSE

    def __init__(self, message, code=None):
        super().__init__(message)
        if code is not None:
            self.code = code


class RecordParseError(AuditError):
    """单条记录解析失败，携带记录序号。"""

    code = E_PARSE

    def __init__(self, message, record_index=None):
        super().__init__(message, E_PARSE)
        self.record_index = record_index
