"""MVCC 可见性诊断组件。

规则版本在 :data:`RULE_VERSION` 中显式声明；所有判定、重放回执与账本
摘要均携带该版本，以便独立复核。
"""

RULE_VERSION = "mvcc-rules-1.0.0"
SCHEMA_VERSION = 1

# 排除理由（稳定枚举，判定结果与账本中只允许出现这些取值）。
REASON_UNCOMMITTED = "uncommitted"
REASON_COMMIT_TOO_LATE = "commit_too_late"
REASON_SUPERSEDED = "superseded"
REASON_CHAIN_BROKEN = "chain_broken"
REASON_ROLLED_BACK = "rolled_back"

# 稳定的理由排序：不依赖任何遍历顺序，始终按该字典序输出。
REASON_ORDER = (
    REASON_CHAIN_BROKEN,
    REASON_COMMIT_TOO_LATE,
    REASON_ROLLED_BACK,
    REASON_SUPERSEDED,
    REASON_UNCOMMITTED,
)

# 不可区分裁决的三类成因。
CAUSE_TIE = "error_indistinguishable"
CAUSE_RULE_CONFLICT = "rule_version_conflict"
CAUSE_CHAIN_BROKEN = "version_chain_broken"

# 可见性规则（业务要求 2 显式声明）。
VISIBILITY_POLICY = (
    "1) 写入事务已提交(committed)且提交序号<=快照序号；"
    "2) 读事务对自身未提交/已提交版本可见(回滚版本除外)；"
    "3) 仅取沿版本链回溯遇到的首个可见版本，其余可见候选标记为 superseded；"
    "4) 断链候选标记为 chain_broken；"
    "5) 不可区分时按 提交序号降序->版本号降序->事务标识序 决胜。"
)
