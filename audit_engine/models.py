"""引擎使用的不可变数据模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# 四个基础权限位：读 1，写 2，执行 4，删除 8
PERM_READ = 1
PERM_WRITE = 2
PERM_EXECUTE = 4
PERM_DELETE = 8
ALL_PERM_BITS = PERM_READ | PERM_WRITE | PERM_EXECUTE | PERM_DELETE

OP_WHITELIST = ("read", "write", "execute", "delete")
OP_REQUIRED_BIT = {
    "read": PERM_READ,
    "write": PERM_WRITE,
    "execute": PERM_EXECUTE,
    "delete": PERM_DELETE,
}

REP_INT_MASK = "int_mask"
REP_OCTAL_STRING = "octal_string"
REPRESENTATION_WHITELIST = (REP_INT_MASK, REP_OCTAL_STRING)

# 八进制字符串的两种舍入口径
ROUND_TRUNCATE = "truncate"
ROUND_NEAREST = "nearest"

# 检查点结构版本与规范化策略版本（任一变化即整体重建）
CHECKPOINT_STRUCT_VERSION = 1
NORMALIZATION_POLICY_VERSION = 1


@dataclass(frozen=True)
class NormalizationPolicy:
    """路径规范化策略参数。"""

    fold_case: bool = True
    unicode_nfc: bool = True
    resolve_links: bool = True
    name: str = "default"

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "fold_case": self.fold_case,
            "unicode_nfc": self.unicode_nfc,
            "resolve_links": self.resolve_links,
        }


@dataclass(frozen=True)
class LinkDecl:
    """符号链接 / 硬链接声明（仅作为路径解析输入，不依赖真实 FS 对象）。"""

    at: str                 # 链接所在的虚拟绝对路径
    target: str             # 链接目标（虚拟路径）
    link_type: str          # "symlink" 或 "hardlink"
    steps: int = 1          # 该链接计入的解析步数


@dataclass(frozen=True)
class AclEntry:
    """单条 ACL 声明。"""

    subject: str
    access: str             # "allow" 或 "deny"
    mask: int               # 规范化后的权限位
    inherited: bool = False # 是否由父目录继承而来
    source_depth: int = 0   # 来源路径深度（越近深度越大）
    epsilon: float = 0.0    # 该条目解析误差上界
    order: int = 0          # 原始排列序号（仅调试，裁决不依赖它）


@dataclass(frozen=True)
class TruthMarker:
    """发送端同步标记与真值区间（真值唯一来源）。"""

    sync_marker: str
    raw_path_bytes: str                 # 原始路径字节的确定性编码（hex）
    raw_perm_value: Any                 # 原始权限数值（int 或 str）
    expected_epsilon_lo: float
    expected_epsilon_hi: float


@dataclass(frozen=True)
class Record:
    """解析后的受检记录。"""

    index: int
    record_id: str
    subject: str
    operation: str
    target_path: str
    representation: str
    perm_value: Any
    rounding: str
    links: tuple[LinkDecl, ...]
    acl: tuple[AclEntry, ...]
    truth: Optional[TruthMarker]
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class PathNormResult:
    """路径规范化结果与分步骤量化误差。"""

    canonical_key: str
    encoding_error: int = 0       # Unicode 归一化改变的码点数
    casefold_error: int = 0       # 大小写折叠改变的码点数
    link_steps: int = 0           # 链接解析累计步数
    traversal_depth: int = 0      # 越出受检根的段数（0 表示未越界）
    link_escape: bool = False
    reachable_path: str = ""
    blocked_code: Optional[str] = None

    def total_error(self) -> int:
        return self.encoding_error + self.casefold_error + self.link_steps


@dataclass
class EffectivePermission:
    """主体有效权限裁决结果。"""

    subject: str
    allowed: int
    denied: int
    effective: int
    epsilon: float
    critical: bool
    shadowed_denies: list[dict[str, Any]] = field(default_factory=list)
    inherit_violations: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class RecordVerdict:
    """单条记录在单个策略下的裁决。"""

    index: int
    record_id: str
    policy_name: str
    canonical_key: str
    reachable_path: str
    path_error: dict[str, int]
    effective_permission: Optional[EffectivePermission]
    perm_epsilon: float
    critical: bool
    codes: list[str]
    traversal_depth: int


@dataclass
class RejectedRecord:
    """解析阶段被拒绝的记录。"""

    index: int
    code: str
    reason: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class FileResult:
    """单个清单文件的审计结果。"""

    file_id: str
    path: str
    status: str                     # ok / timeout / budget / error
    codes: list[str]
    verdicts: list[RecordVerdict]
    rejections: list[RejectedRecord]
    elapsed_seconds: float
    processed_records: int
    dropped_records: int
    truth: dict[str, Any]
    policy_conflicts: list[dict[str, Any]] = field(default_factory=list)
