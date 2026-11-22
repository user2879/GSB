"""独立验章核对工具。

仅读取账本与原始声明，自行重算身份摘要、盖章校验值与验章结论，
与账本逐项比较；首个不一致令牌以 E_AUDIT 定位。

用法：
  python scripts/audit.py --spec samples/spec.base.json --ledger run/main.db
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference_stamper.app import build_identities  # noqa: E402
from inference_stamper.errors import AuditError, SnapshotError  # noqa: E402
from inference_stamper.ledger import open_ledger  # noqa: E402
from inference_stamper.spec import load_spec  # noqa: E402
from inference_stamper.stamping import (  # noqa: E402
    canonical_token,
    compute_checksum,
    local_key,
)


def audit(spec_path: str, ledger_path: str) -> int:
    spec = load_spec(spec_path)
    try:
        ledger, report = open_ledger(ledger_path, on_mismatch="raise")
    except SnapshotError as exc:
        raise AuditError(
            "账本快照校验未通过，无法进行逐项核对",
            field=exc.field,
            detail=exc.detail,
        ) from exc

    identities = build_identities(spec)
    key = local_key()
    verification_map = {
        row.token_id: row for row in ledger.list_verifications()
    }

    for row in ledger.list_stamps():
        token_id = row.token_id
        identity = identities.get(row.backend_id)
        if identity is None:
            raise AuditError("账本令牌引用了声明中不存在的后端",
                              item_id=token_id, field="backend_id")
        if row.identity_digest != identity.identity_digest:
            raise AuditError(
                "重算身份摘要与账本不一致",
                item_id=token_id, field="identity_digest",
                detail={"stored": row.identity_digest,
                        "recomputed": identity.identity_digest},
            )
        content = canonical_token(
            token_id=row.token_id,
            backend_id=row.backend_id,
            identity_digest=row.identity_digest,
            token_spec_version=row.token_spec_version,
            request_seq=row.request_seq,
            stamp_seq=row.stamp_seq,
        )
        expected = compute_checksum(key, content)
        if expected != row.checksum:
            raise AuditError(
                "重算盖章校验值与账本不一致",
                item_id=token_id, field="checksum",
                detail={"stored": row.checksum, "recomputed": expected},
            )
        expected_accepted = (
            row.identity_digest == identity.identity_digest
            and row.token_spec_version == identity.token_spec_version
        )
        stored_verification = verification_map.get(token_id)
        if stored_verification is not None and (
            bool(stored_verification.accepted) != expected_accepted
        ):
            raise AuditError(
                "重算验章结论与账本验章状态不一致",
                item_id=token_id, field="verification.accepted",
                detail={"stored": str(bool(stored_verification.accepted)),
                        "recomputed": str(expected_accepted)},
            )

    count = len(ledger.list_stamps())
    print(f"[AUDIT OK] 共核对 {count} 条盖章记录，全部一致")
    ledger.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="独立验章核对工具")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--ledger", required=True)
    args = parser.parse_args()
    try:
        return audit(args.spec, args.ledger)
    except AuditError as exc:
        print(f"[AUDIT FAIL] {exc.code}: {exc.message} "
              f"(标识={exc.item_id or '-'}, 字段={exc.field or '-'})")
        if exc.detail:
            print(f"  依据: {exc.detail}")
        return 3


if __name__ == "__main__":
    sys.exit(main())
