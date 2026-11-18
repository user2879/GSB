"""Gather operator: ordered fixed-point aggregation with contributions."""

from .errors import E_EMBED_INVALID, E_ID_MISSING, E_POINT_BUDGET_EXHAUSTED
from .fingerprints import gather_fingerprint
from .fixedpoint import fixed_mean


class GatherRequest:
    def __init__(self, index, ids, request_id=None):
        self.index = index
        self.ids = list(ids)
        self.request_id = request_id or f"req-{index:04d}"

    def task_keys(self):
        """Stable sort keys derived from (request index, id position)."""
        return [(self.index, pos) for pos in range(len(self.ids))]


class GatherResult:
    def __init__(self, request):
        self.request_index = request.index
        self.request_id = request.request_id
        self.ids = list(request.ids)
        self.status = "ok"
        self.code = None
        self.detail = None
        self.mode = None
        self.quant_level = None
        self.frac_bits = None
        self.contributions = []
        self.acc = []
        self.fingerprint = ""
        self.referenced = []       # ordered [word_id, row_fp] pairs
        self.points_used = 0
        self.error_position = None

    def to_dict(self):
        return {
            "request_index": self.request_index,
            "request_id": self.request_id,
            "ids": list(self.ids),
            "status": self.status,
            "code": self.code,
            "detail": self.detail,
            "mode": self.mode,
            "quant_level": self.quant_level,
            "frac_bits": self.frac_bits,
            "contributions": [list(c) for c in self.contributions],
            "acc": list(self.acc),
            "fingerprint": self.fingerprint,
            "referenced": [list(p) for p in self.referenced],
            "points_used": self.points_used,
            "error_position": self.error_position,
        }

    @classmethod
    def from_dict(cls, data):
        req = GatherRequest(data["request_index"], data["ids"],
                            data["request_id"])
        res = cls(req)
        res.status = data["status"]
        res.code = data["code"]
        res.detail = data["detail"]
        res.mode = data["mode"]
        res.quant_level = data["quant_level"]
        res.frac_bits = data["frac_bits"]
        res.contributions = [list(c) for c in data["contributions"]]
        res.acc = list(data["acc"])
        res.fingerprint = data["fingerprint"]
        res.referenced = [list(p) for p in data["referenced"]]
        res.points_used = data["points_used"]
        res.error_position = data["error_position"]
        return res


def execute_gather(request, table, mode, quant_level, frac_bits, quota,
                   extra_ids=()):
    """Execute one gather request deterministically in ID sequence order.

    ``quota`` is the per-request row-access point allowance; exceeding it
    aborts the request with E_POINT_BUDGET_EXHAUSTED.
    """
    ids = list(request.ids) + list(extra_ids)
    effective = GatherRequest(request.index, ids, request.request_id)
    result = GatherResult(effective)
    result.mode = mode
    result.quant_level = quant_level
    result.frac_bits = frac_bits
    dim = table.config.dim
    acc = [0] * dim
    # Tasks are processed in stable sort-key order: (request_index, position).
    for pos, key in enumerate(sorted(
            effective.task_keys(), key=lambda k: (k[0], k[1]))):
        wid = ids[key[1]]
        if result.points_used + 1 > quota:
            result.status = "error"
            result.code = E_POINT_BUDGET_EXHAUSTED
            result.error_position = pos
            result.detail = {
                "request_index": request.index,
                "position": pos,
                "quota": quota,
            }
            return result
        result.points_used += 1
        row = table.rows.get(wid)
        if row is None:
            result.status = "error"
            result.error_position = pos
            if wid in table.failed_word_ids:
                result.code = E_EMBED_INVALID
                result.detail = {
                    "request_index": request.index,
                    "position": pos,
                    "word_id": wid,
                    "reason": "row rejected at validation",
                }
            else:
                result.code = E_ID_MISSING
                result.detail = {
                    "request_index": request.index,
                    "position": pos,
                    "word_id": wid,
                }
            return result
        for d in range(dim):
            acc[d] += row.qvec[d]
        result.contributions.append(list(row.qvec))
        result.referenced.append([wid, row.row_fp])
    if mode == "mean":
        acc = [fixed_mean(v, len(ids)) for v in acc]
    result.acc = acc
    result.fingerprint = gather_fingerprint(mode, quant_level, frac_bits,
                                            ids, acc)
    return result
