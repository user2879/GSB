"""恢复后的一致性重放校验（E_REPLAY）。

依据账本记录的 identity_version / token_spec_version 重新推导
身份摘要，重算盖章校验值并重放验章，定位首个不一致令牌与字段。
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import ReplayError
from .ledger import LedgerSession
from .parameters import build_identity
from .spec import Spec
from .stamping import TOKEN_SPEC_VERSION, canonical_token, compute_checksum, local_key


@dataclass(frozen=True)
class ReplayItem:
    token_id: str
    field: str
    stored: str
    recomputed: str


def rederive_identities(spec: Spec) -> dict[str, "object"]:
    """按当前声明重新推导全部后端身份摘要。"""
    return {
        backend.backend_id: build_identity(
            backend_id=backend.backend_id,
            model_dim=backend.model_dim,
            hidden_dim=backend.hidden_dim,
            seed=backend.seed,
            quant=backend.quant,
            identity_version=backend.identity_version,
            token_spec_version=backend.token_spec_version,
        )
        for backend in spec.backends
    }


def replay_ledger(
    ledger: LedgerSession,
    spec: Spec,
    *,
    key_material_path: str | None = None,
) -> list[ReplayItem]:
    """逐项重算并比较；发现首个差异即抛 E_REPLAY。"""
    key = local_key(key_material_path)
    identities = rederive_identities(spec)
    backend_by_id = {b.backend_id: b for b in spec.backends}
    mismatches: list[ReplayItem] = []

    for row in ledger.list_stamps():
        token_id = row.token_id
        backend = backend_by_id.get(row.backend_id)
        if backend is None:
            raise ReplayError(
                "账本令牌引用了声明中不存在的后端",
                item_id=token_id, field="backend_id",
                detail={"stored": row.backend_id, "recomputed": ""},
            )

        identity = identities[row.backend_id]

        if row.identity_version != backend.identity_version:
            raise ReplayError(
                "账本身份版本与声明不一致",
                item_id=token_id, field="identity_version",
                detail={"stored": str(row.identity_version),
                        "recomputed": str(backend.identity_version)},
            )
        if row.identity_digest != identity.identity_digest:
            raise ReplayError(
                "重推身份摘要与账本绑定摘要不一致",
                item_id=token_id, field="identity_digest",
                detail={"stored": row.identity_digest,
                        "recomputed": identity.identity_digest},
            )
        if row.token_spec_version != TOKEN_SPEC_VERSION:
            raise ReplayError(
                "账本令牌口径版本与当前盖章规则口径不一致",
                item_id=token_id, field="token_spec_version",
                detail={"stored": str(row.token_spec_version),
                        "recomputed": str(TOKEN_SPEC_VERSION)},
            )

        content = canonical_token(
            token_id=row.token_id,
            backend_id=row.backend_id,
            identity_digest=row.identity_digest,
            token_spec_version=row.token_spec_version,
            request_seq=row.request_seq,
            stamp_seq=row.stamp_seq,
        )
        expected_checksum = compute_checksum(key, content)
        if expected_checksum != row.checksum:
            raise ReplayError(
                "重算盖章校验值与账本记录不一致",
                item_id=token_id, field="checksum",
                detail={"stored": row.checksum,
                        "recomputed": expected_checksum},
            )

        expected_accepted = (
            row.identity_digest == identity.identity_digest
            and row.token_spec_version == identity.token_spec_version
        )
        verification = _verification_map(ledger).get(token_id)
        if verification is not None:
            if bool(verification.accepted) != expected_accepted:
                raise ReplayError(
                    "重放验章结论与账本验章状态不一致",
                    item_id=token_id, field="verification.accepted",
                    detail={"stored": str(bool(verification.accepted)),
                            "recomputed": str(expected_accepted)},
                )
        mismatches.append(ReplayItem(
            token_id=token_id, field="",
            stored=row.checksum, recomputed=expected_checksum,
        ))

    return mismatches


def _verification_map(ledger: LedgerSession) -> dict:
    return {row.token_id: row for row in ledger.list_verifications()}
