"""升级路径与直达路径的逐字段一致性校验."""

from .canon import digest_of
from .errors import E_DIVERGE, LedgerError

COMPARE_FIELDS = ("request_id", "group_rank", "backend_fingerprint",
                  "latency_ms", "output_digest")


def compare_views(reference_views, candidate_views, label):
    """reference/candidate 为 request_id -> view. 返回 (digest, count)."""
    for rid in sorted(reference_views):
        if rid not in candidate_views:
            raise LedgerError(E_DIVERGE, f"{label}: record missing in candidate",
                              record_id=rid, field="<record>", stage="consistency")
        for field in COMPARE_FIELDS:
            if reference_views[rid][field] != candidate_views[rid][field]:
                raise LedgerError(
                    E_DIVERGE,
                    f"{label}: field {field} diverges",
                    record_id=rid, field=field, stage="consistency",
                    extra={"reference": reference_views[rid][field],
                           "candidate": candidate_views[rid][field]},
                )
    digest = digest_of({rid: {f: reference_views[rid][f] for f in COMPARE_FIELDS}
                        for rid in sorted(reference_views)})
    return digest, len(reference_views)
