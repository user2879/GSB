"""强制终止后的恢复：只承认已完整提交的盖章记录。

SQLite 每条盖章在独立事务中提交（见 LedgerSession.upsert_stamp），
因此崩溃时未返回的令牌不会留下半条记录。恢复阶段：

* 已在 stamps 表中的令牌 => committed，直接复用；
* 不在表中的请求令牌 => unknown，标记要求重盖，
  且不生成任何“退出报告”。
"""

from __future__ import annotations

from dataclasses import dataclass

from .ledger import LedgerSession
from .spec import Spec


@dataclass(frozen=True)
class RecoveryReport:
    committed_token_ids: tuple[str, ...]
    unknown_token_ids: tuple[str, ...]
    stamp_seq_by_backend: dict[str, int]

    @property
    def needs_restamp(self) -> tuple[str, ...]:
        return self.unknown_token_ids


def recover(ledger: LedgerSession, spec: Spec) -> RecoveryReport:
    stamped = {row.token_id for row in ledger.list_stamps()}
    committed: list[str] = []
    unknown: list[str] = []
    for request in spec.requests:
        if request.token_id in stamped:
            committed.append(request.token_id)
        else:
            unknown.append(request.token_id)

    seq_by_backend = {
        backend.backend_id: ledger.max_stamp_seq(backend.backend_id)
        for backend in spec.backends
    }
    return RecoveryReport(
        committed_token_ids=tuple(committed),
        unknown_token_ids=tuple(unknown),
        stamp_seq_by_backend=seq_by_backend,
    )


class ForcedCrash(RuntimeError):
    """模拟强制终止：在指定令牌盖章“提交前”抛出。"""


def stamp_batch_with_crash_point(
    stamper,
    requests,
    crash_after_token_id: str | None = None,
) -> list:
    """顺序盖章；在 crash_after_token_id 完成后立即模拟强制终止。

    崩溃点之后的请求不会执行，模拟进程被 kill，
    已提交事务仍完整落盘。
    """
    stamped = []
    for request in requests:
        stamped.append(stamper.stamp(request))
        if crash_after_token_id == request.token_id:
            raise ForcedCrash(
                f"模拟强制终止：已提交 {request.token_id}，"
                "后续令牌未处理"
            )
    return stamped
