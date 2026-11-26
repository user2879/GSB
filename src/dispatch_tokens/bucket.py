"""整数定点令牌桶。

关键语义（与技术说明一致）：

* 桶内余额以定点整数保存，单位 = 逻辑令牌 * ``scale``，``scale`` 取 10 的
  0..3 次幂；所有算术只用整数，截断方向固定为向零截断
  （:data:`dispatch_tokens.fixedpoint.TRUNC_TOWARD_ZERO`）。
* 补充公式：``added_raw = elapsed_ms * rate * scale // 1000``，整除余数
  累积在 ``remainder`` 中，下一轮补充时优先消化余数。因此无论事件如何
  批量切分、真实等待多久，只要经过的单调毫秒相同，补充量就逐位一致。
* 夹取采用饱和规则：余额夹在 ``[0, burst * scale]``；夹取发生时余数保持
  不变（被夹掉的是整令牌而非分数余数），同样保证可重放。
* 修订提交只在提交时刻做一次再量化结转（新因子下先反量化再量化），
  绝不重算历史补充。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

from .fixedpoint import (SATURATE, TRUNC_TOWARD_ZERO, quantize, rescale,
                         saturate, validate_scale)


@dataclass
class RefillRecord:
    node_id: str
    elapsed_ms: int
    added: int
    before_clamp: int
    after_clamp: int
    cap: int
    rate: int
    scale: int
    param_version: int
    clamp_applied: bool
    truncation: str = TRUNC_TOWARD_ZERO
    saturation: str = SATURATE


@dataclass
class BucketState:
    node_id: str
    balance: int
    remainder: int
    rate: int
    burst: int
    scale: int
    last_ms: int
    param_version: int
    bucket_key: int

    def cap(self) -> int:
        return quantize(self.burst, self.scale)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TokenBucket:
    """单节点令牌桶；余额单位为当前 scale 下的定点整数。"""

    def __init__(self, node_id: str, rate: int, burst: int, scale: int,
                 bucket_key: int, start_ms: int, param_version: int) -> None:
        validate_scale(scale)
        if rate < 0 or burst < 0:
            raise ValueError("rate and burst must be non-negative integers")
        self.state = BucketState(
            node_id=node_id,
            balance=quantize(burst, scale),
            remainder=0,
            rate=rate,
            burst=burst,
            scale=scale,
            last_ms=start_ms,
            param_version=param_version,
            bucket_key=bucket_key,
        )

    @classmethod
    def from_state(cls, state: BucketState) -> "TokenBucket":
        bucket = cls.__new__(cls)
        bucket.state = state
        return bucket

    def refill(self, now_ms: int, param_version: int | None = None
               ) -> RefillRecord:
        """按单调毫秒补充并饱和夹取，返回审计记录。"""
        state = self.state
        elapsed = now_ms - state.last_ms
        if elapsed < 0:
            raise ValueError(
                f"clock moved backwards for node {state.node_id}: "
                f"{now_ms} < {state.last_ms}")
        raw_scaled = elapsed * state.rate * state.scale + state.remainder
        added = raw_scaled // 1000
        state.remainder = raw_scaled % 1000
        before = state.balance
        before_clamp = before + added
        cap = self.state.cap()
        clamped = saturate(before_clamp, 0, cap)
        clamp_applied = clamped != before_clamp
        state.balance = clamped
        state.last_ms = now_ms
        if param_version is not None:
            state.param_version = param_version
        return RefillRecord(
            node_id=state.node_id, elapsed_ms=elapsed, added=added,
            before_clamp=before_clamp, after_clamp=clamped, cap=cap,
            rate=state.rate, scale=state.scale,
            param_version=state.param_version, clamp_applied=clamp_applied,
        )

    def try_consume(self, amount_logical: int) -> tuple[bool, int]:
        """尝试按逻辑令牌整数扣减，返回 (是否成功, 扣减后余额)。"""
        if amount_logical < 0:
            raise ValueError("consume amount must be non-negative")
        need = quantize(amount_logical, self.state.scale)
        if self.state.balance >= need:
            self.state.balance -= need
            return True, self.state.balance
        return False, self.state.balance

    def refund(self, amount_logical: int) -> int:
        """取消回收：饱和归还，返回回收后余额。"""
        amount = quantize(amount_logical, self.state.scale)
        cap = self.state.cap()
        self.state.balance = saturate(self.state.balance + amount, 0, cap)
        return self.state.balance

    def apply_revision(self, new_rate: int | None, new_burst: int | None,
                       new_scale: int | None, committed_ms: int,
                       new_version: int) -> dict[str, Any]:
        """修订边界结转：先补到提交时刻，再换参数/再量化，不重算历史。"""
        refill_record = self.refill(committed_ms)
        state = self.state
        old_scale = state.scale
        carried_balance = state.balance
        carried_remainder = state.remainder
        if new_scale is not None and new_scale != old_scale:
            carried_balance = rescale(state.balance, old_scale, new_scale)
            carried_remainder = rescale(state.remainder, old_scale, new_scale)
        state.scale = new_scale if new_scale is not None else old_scale
        state.rate = new_rate if new_rate is not None else state.rate
        state.burst = new_burst if new_burst is not None else state.burst
        state.balance = saturate(carried_balance, 0, self.state.cap())
        state.remainder = carried_remainder
        state.last_ms = committed_ms
        state.param_version = new_version
        return {
            "node_id": state.node_id,
            "committed_ms": committed_ms,
            "old_scale": old_scale,
            "new_scale": state.scale,
            "carried_balance": state.balance,
            "carried_remainder": state.remainder,
            "new_rate": state.rate,
            "new_burst": state.burst,
            "param_version": new_version,
            "refill": asdict(refill_record),
        }
