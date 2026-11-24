"""业务错误码与失败清单条目。

校验/执行阶段的异常均以业务码形式进入失败清单，单个节点失败不会中断
其余合法节点的处理。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict


# ---- 通用 / 拓扑校验 ----
E_NODE_DUPLICATE = "E_NODE_DUPLICATE"
E_EDGE_ENDPOINT = "E_EDGE_ENDPOINT"
E_GRAPH_CYCLE = "E_GRAPH_CYCLE"
E_MULTIPLE_PRODUCER = "E_MULTIPLE_PRODUCER"
E_INPUT_UNBOUND = "E_INPUT_UNBOUND"
E_ACTIVATION_PARAM = "E_ACTIVATION_PARAM"
E_PERM_INVALID = "E_PERM_INVALID"
E_ENUM_INVALID = "E_ENUM_INVALID"
E_UNKNOWN_OP = "E_UNKNOWN_OP"
E_INPLACE_CONFLICT = "E_INPLACE_CONFLICT"
E_UPSTREAM_FAILED = "E_UPSTREAM_FAILED"
E_FUSION_TOLERANCE = "E_FUSION_TOLERANCE"
E_RECOMPUTE_BUDGET_PENDING = "E_RECOMPUTE_BUDGET_PENDING"

# ---- 异常注入命中（映射到对应业务码） ----
E_INJ_NONFINITE = "E_INJ_NONFINITE"
E_INJ_DUPLICATE_PERM = "E_INJ_DUPLICATE_PERM"
E_INJ_ACTIVATION_PARAM = "E_INJ_ACTIVATION_PARAM"
E_INJ_IMPL_MISSING = "E_INJ_IMPL_MISSING"


@dataclass
class Failure:
    """失败清单中的一条记录，字段顺序稳定，可逐字段比较。"""

    topo_index: int
    node_id: str
    stage: str
    code: str
    message: str
    injection_id: str | None = None
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
