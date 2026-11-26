"""三格式回执的归一视图与采纳规则。

归一视图：每个归一结果携带来源格式版本、被补齐字段集合与补齐依据。
补齐规则：
  r1 缺 param_version/elapsed_ms/worker_id/rank/dup_of；
  r2 缺 rank/dup_of；
  rank 缺失时按批内请求标识字典序推导稳定名次。

采纳规则（同一 (批次, 请求) 混入多格式回执时）：
  1) 真实参数版本序号降序（被补齐的版本号视为 -1，不得覆盖高版本真实值）；
  2) 格式版本降序；
  3) 来源类别：当期计算（worker:// 或 local://）优先于磁盘历史文件；
  4) 源文件名升序、源行号升序（稳定决胜，不依赖哈希顺序）。
被否决回执保留为对照并记录否决理由。
"""

import json
from dataclasses import dataclass, field

DEFAULT_FILLS = {
    "param_version": (0, "absent-in-format:default-0"),
    "elapsed_ms": (0, "absent-in-format:default-0"),
    "worker_id": ("unknown", "absent-in-format:default-unknown"),
    "dup_of": (None, "absent-in-format:default-none"),
}


@dataclass
class NormView:
    receipt: object
    src_format: int
    filled: frozenset = field(default=frozenset)
    fill_basis: dict = field(default_factory=dict)
    param_version: int = 0
    elapsed_ms: int = 0
    worker_id: str = "unknown"
    rank: int = 0
    dup_of: str = None

    @property
    def batch_id(self):
        return self.receipt.batch_id

    @property
    def request_id(self):
        return self.receipt.request_id

    @property
    def output_digest(self):
        return self.receipt.output_digest

    def real_param_version(self):
        """被补齐的参数版本不得参与覆盖比较，视为 -1。"""
        return -1 if "param_version" in self.filled else self.param_version

    def source_rank(self):
        """当期计算来源优先于磁盘历史文件（同版本同格式时）。"""
        src = self.receipt.src_file
        return 0 if src.startswith(("worker://", "local://")) else 1

    def adoption_key(self):
        return (-self.real_param_version(), -self.src_format, self.source_rank(),
                self.receipt.src_file, self.receipt.src_line)


def normalize(receipts):
    """把任意格式回执归一为统一视图；rank 缺失时按批内请求标识序推导。"""
    batch_requests = {}
    for rec in receipts:
        batch_requests.setdefault(rec.batch_id, set()).add(rec.request_id)
    derived_rank = {
        bid: {rid: rank for rank, rid in enumerate(sorted(rids))}
        for bid, rids in batch_requests.items()
    }
    views = []
    for rec in receipts:
        view = NormView(receipt=rec, src_format=rec.src_format)
        filled = set()
        basis = {}
        for name, (default, why) in DEFAULT_FILLS.items():
            if rec.real(name):
                setattr(view, name, getattr(rec, name))
            else:
                setattr(view, name, default)
                filled.add(name)
                basis[name] = why
        if rec.real("rank"):
            view.rank = rec.rank
        else:
            view.rank = derived_rank[rec.batch_id][rec.request_id]
            filled.add("rank")
            basis["rank"] = "derived:request-id-lexicographic"
        view.filled = frozenset(filled)
        view.fill_basis = basis
        views.append(view)
    return views


def adopt(views):
    """按 (批次, 请求) 分组采纳；返回 {key: (adopted, [(rejected, reason), ...])}。"""
    groups = {}
    for view in views:
        groups.setdefault((view.batch_id, view.request_id), []).append(view)
    result = {}
    for key in sorted(groups):
        ordered = sorted(groups[key], key=lambda v: v.adoption_key())
        winner = ordered[0]
        rejected = []
        for loser in ordered[1:]:
            reason = (
                f"superseded-by:{winner.receipt.src_file}#{winner.receipt.src_line}"
                f":param_version={winner.real_param_version()}>={loser.real_param_version()}"
                f":format=r{winner.src_format}>=r{loser.src_format}"
            )
            rejected.append((loser, reason))
        result[key] = (winner, rejected)
    return result


def filled_json(view):
    return json.dumps(sorted(view.filled), ensure_ascii=False)


def basis_json(view):
    return json.dumps(view.fill_basis, sort_keys=True, ensure_ascii=False)
