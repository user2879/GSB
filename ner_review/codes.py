"""Fixed business codes used by validation, interaction and recovery."""

from __future__ import annotations


class Code:
    OK = "NER-0000"

    INVALID_JSONL = "NER-E1001"
    DUPLICATE_SAMPLE_ID = "NER-E1002"
    TAG_LENGTH_MISMATCH = "NER-E1003"
    UNKNOWN_TAG = "NER-E1004"
    INVALID_BIESO_SPAN = "NER-E1005"
    INVALID_SAMPLE = "NER-E1006"
    LEXICON_INVALID = "NER-E1007"

    PACKAGE_TRUNCATED = "NER-E2001"
    INVALID_FEATURE_DIM = "NER-E2002"
    INVALID_BUCKET_COUNT = "NER-E2003"
    FEATURE_DIM_OUT_OF_RANGE = "NER-E2004"
    NON_FINITE_WEIGHT = "NER-E2005"
    INVALID_MODEL_VERSION = "NER-E2006"
    INVALID_PACKAGE = "NER-E2007"

    UNKNOWN_COMMAND = "NER-E3001"
    STALE_SUMMARY = "NER-E3002"
    SUMMARY_PENDING = "NER-E3003"
    COMMAND_VALIDATION = "NER-E3004"

    SNAPSHOT_VERSION_MISMATCH = "NER-E4001"
    SNAPSHOT_CORRUPT = "NER-E4002"
    SNAPSHOT_REBUILD = "NER-E4003"

    REPAIR_I_WITHOUT_B = "NER-W4101"
    REPAIR_E_WITHOUT_B = "NER-W4102"
    REPAIR_ORPHAN_B = "NER-W4103"
    REPAIR_LABEL_JUMP = "NER-W4104"
    REPAIR_UNCLOSED_SPAN = "NER-W4105"


MESSAGES = {
    Code.OK: "ok",
    Code.INVALID_JSONL: "JSONL 行无法解析",
    Code.DUPLICATE_SAMPLE_ID: "样本标识重复",
    Code.TAG_LENGTH_MISMATCH: "字序列与标签序列长度不一致",
    Code.UNKNOWN_TAG: "存在 BIESO 枚举外标签",
    Code.INVALID_BIESO_SPAN: "BIESO 片段起止、连续性或实体类型非法",
    Code.INVALID_SAMPLE: "样本结构非法",
    Code.LEXICON_INVALID: "词表条目结构或实体类型非法",
    Code.PACKAGE_TRUNCATED: "参数包缺失、截断或不是合法 JSON",
    Code.INVALID_FEATURE_DIM: "特征维度必须是正整数",
    Code.INVALID_BUCKET_COUNT: "哈希桶数必须是正整数",
    Code.FEATURE_DIM_OUT_OF_RANGE: "特征维度、桶编号或权重形状越界",
    Code.NON_FINITE_WEIGHT: "参数包权重包含 NaN/Infinity 等非有限数",
    Code.INVALID_MODEL_VERSION: "模型版本号缺失或非法",
    Code.INVALID_PACKAGE: "参数包结构非法",
    Code.UNKNOWN_COMMAND: "未知命令或命令参数非法",
    Code.STALE_SUMMARY: "交互预算内未完成最新代次，返回陈旧摘要",
    Code.SUMMARY_PENDING: "首个摘要仍在生成",
    Code.COMMAND_VALIDATION: "命令对象未通过校验",
    Code.SNAPSHOT_VERSION_MISMATCH: "快照版本或配置哈希不兼容，已重建",
    Code.SNAPSHOT_CORRUPT: "快照校验失败，已从空白状态重建",
    Code.SNAPSHOT_REBUILD: "启动判定为全量重建",
    Code.REPAIR_I_WITHOUT_B: "I 前无 B，按规则转为 B",
    Code.REPAIR_E_WITHOUT_B: "E 前无 B/I，按规则转为 S",
    Code.REPAIR_ORPHAN_B: "B 后未形成完整 I/E 结构，孤片段按 S 闭合",
    Code.REPAIR_LABEL_JUMP: "BIESO 或实体类型跳跃，已闭合旧片段并开启新片段",
    Code.REPAIR_UNCLOSED_SPAN: "片段在 O 或文末未闭合，已按 E/S 规则闭合",
}


def diagnostic(code: str, detail: str | None = None) -> str:
    message = MESSAGES.get(code, "未登记业务码")
    return f"{code} {message}" + (f"：{detail}" if detail else "")
