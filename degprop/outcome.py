"""确定性结局规则（固定整数种子的线性同余生成器）。

服务端与采集端共享同一规则，保证演示可重复；四类结局：
    success / slow_success / connection_refused / timeout

注意语义边界：
    * connection_refused 永远来自对“已绑定未监听”保留端口的真实 connect 失败；
    * timeout 永远来自客户端真实 socket.settimeout 到期，服务端用真实睡眠
      制造慢响应，组件从不以本地 sleep 冒充超时。
"""
from __future__ import annotations

from dataclasses import dataclass

from .models import Outcome

LCG_MULTIPLIER = 1_103_515_245
LCG_INCREMENT = 12_345
LCG_MODULUS = 2**31

# 四档结局的整数桶边界（对 LCG 取模 100 判定）
SLOW_ABOVE = 75       # 75..84 慢响应成功
REFUSE_ABOVE = 85     # 85..94 连接拒绝
TIMEOUT_ABOVE = 95    # 95..99 真实套接字超时


@dataclass(frozen=True)
class PlannedOutcome:
    outcome: str
    bucket: int
    handle_ms: int
    connect_port_kind: str  # "live" 或 "refused"


def _lcg(seed: int, seq: int) -> int:
    state = seed & (LCG_MODULUS - 1)
    for _ in range(seq):
        state = (LCG_MULTIPLIER * state + LCG_INCREMENT) % LCG_MODULUS
    return state


def plan_outcome(seed: int, seq: int, *, slow_ms: int = 400,
                 handle_ms: int = 12) -> PlannedOutcome:
    """按 (服务 seed, 事件序号) 生成确定性结局。"""
    state = _lcg(seed, seq)
    bucket = state % 100
    if bucket < SLOW_ABOVE:
        return PlannedOutcome(Outcome.SUCCESS.value, bucket, handle_ms, "live")
    if bucket < REFUSE_ABOVE:
        return PlannedOutcome(Outcome.SLOW_SUCCESS.value, bucket, slow_ms, "live")
    if bucket < TIMEOUT_ABOVE:
        return PlannedOutcome(Outcome.REFUSED.value, bucket, 0, "refused")
    return PlannedOutcome(Outcome.TIMEOUT.value, bucket, slow_ms, "live")
