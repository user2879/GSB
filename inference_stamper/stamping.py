"""推理令牌盖章与验章。

令牌口径 TOKEN_SPEC_VERSION 随盖章规则版本走；后端声明中的
token_spec_version 表示该后端当前接受的令牌口径。

校验值 checksum = HMAC-SHA256(本地固定密钥, 规范化令牌内容)，
规范化字段固定顺序：token_id, backend_id, identity_digest,
token_spec_version, request_seq, stamp_seq。
同一令牌重复盖章直接返回账本中的原记录，序号不重复递增。
"""

from __future__ import annotations

import hashlib
import hmac
import time
from dataclasses import dataclass
from pathlib import Path

from .errors import IdentityError
from .ledger import LedgerSession, StampRow, VerificationRow
from .parameters import IdentityArtifact
from .spec import Spec, TokenRequest

TOKEN_SPEC_VERSION = 1
_LOCAL_KEY_DOMAIN = b"inference-stamper/local-hmac-key/v1"
_CANONICAL_FIELDS = (
    "token_id",
    "backend_id",
    "identity_digest",
    "token_spec_version",
    "request_seq",
    "stamp_seq",
)


def local_key(key_material_path: str | Path | None = None) -> bytes:
    """派生本地确定性 HMAC 密钥；不随进程随机变化。"""
    if key_material_path is not None:
        material = Path(key_material_path).read_bytes()
    else:
        material = b"deterministic-local-seed:20261123"
    return hashlib.sha256(_LOCAL_KEY_DOMAIN + material).digest()


def canonical_token(
    token_id: str,
    backend_id: str,
    identity_digest: str,
    token_spec_version: int,
    request_seq: int,
    stamp_seq: int,
) -> bytes:
    parts: list[bytes] = []
    values = (
        token_id, backend_id, identity_digest, token_spec_version,
        request_seq, stamp_seq,
    )
    for field, value in zip(_CANONICAL_FIELDS, values):
        if isinstance(value, int):
            payload = int(value).to_bytes(8, "big", signed=True)
        else:
            payload = str(value).encode("utf-8")
        tag = field.encode("utf-8")
        parts.append(len(tag).to_bytes(2, "big") + tag
                     + len(payload).to_bytes(8, "big") + payload)
    return b"".join(parts)


def compute_checksum(key: bytes, content: bytes) -> str:
    return hmac.new(key, content, hashlib.sha256).hexdigest()


@dataclass(frozen=True)
class StampedToken:
    token_id: str
    backend_id: str
    identity_digest: str
    token_spec_version: int
    request_seq: int
    stamp_seq: int
    checksum: str
    stamped_at_ms: int
    reused: bool

    @classmethod
    def from_row(cls, row: StampRow, reused: bool) -> "StampedToken":
        return cls(
            token_id=row.token_id,
            backend_id=row.backend_id,
            identity_digest=row.identity_digest,
            token_spec_version=row.token_spec_version,
            request_seq=row.request_seq,
            stamp_seq=row.stamp_seq,
            checksum=row.checksum,
            stamped_at_ms=row.stamped_at_ms,
            reused=reused,
        )


@dataclass(frozen=True)
class VerifyResult:
    token_id: str
    accepted: bool
    code: str
    reason: str
    field: str
    verified_at_ms: int


class Stamper:
    """围绕账本提供盖章、验章能力，身份摘要按后端实时提供。"""

    def __init__(self, ledger: LedgerSession,
                 identities: dict[str, IdentityArtifact],
                 key_material_path: str | Path | None = None) -> None:
        self.ledger = ledger
        self.identities = identities
        self._key = local_key(key_material_path)

    def stamp(self, request: TokenRequest) -> StampedToken:
        existing = self.ledger.get_stamp(request.token_id)
        if existing is not None:
            return StampedToken.from_row(existing, reused=True)

        identity = self.identities[request.backend_id]
        stamp_seq = self.ledger.max_stamp_seq(request.backend_id) + 1
        content = canonical_token(
            token_id=request.token_id,
            backend_id=request.backend_id,
            identity_digest=identity.identity_digest,
            token_spec_version=TOKEN_SPEC_VERSION,
            request_seq=request.request_seq,
            stamp_seq=stamp_seq,
        )
        checksum = compute_checksum(self._key, content)
        row = StampRow(
            token_id=request.token_id,
            backend_id=request.backend_id,
            identity_digest=identity.identity_digest,
            identity_version=identity.identity_version,
            token_spec_version=TOKEN_SPEC_VERSION,
            request_seq=request.request_seq,
            stamp_seq=stamp_seq,
            checksum=checksum,
            stamped_at_ms=time.monotonic_ns() // 1_000_000,
        )
        self.ledger.upsert_stamp(row)
        return StampedToken.from_row(row, reused=False)

    def verify(self, token_id: str, *, persist: bool = True) -> VerifyResult:
        row = self.ledger.get_stamp(token_id)
        if row is None:
            raise IdentityError("账本中不存在该盖章记录",
                                item_id=token_id, field="token_id")
        identity = self.identities.get(row.backend_id)
        now_ms = time.monotonic_ns() // 1_000_000

        def finish(accepted: bool, code: str, reason: str,
                   field: str) -> VerifyResult:
            result = VerifyResult(
                token_id=token_id, accepted=accepted, code=code,
                reason=reason, field=field, verified_at_ms=now_ms,
            )
            if persist:
                self.ledger.replace_verification(VerificationRow(
                    token_id=token_id, accepted=accepted, code=code,
                    reason=reason, field=field, verified_at_ms=now_ms,
                ))
            return result

        if identity is None:
            return finish(False, "E_STALE",
                          "后端已不存在，无法取得当前身份摘要",
                          "backend_id")

        # 1) 校验本地盖章校验值是否被篡改
        content = canonical_token(
            token_id=row.token_id,
            backend_id=row.backend_id,
            identity_digest=row.identity_digest,
            token_spec_version=row.token_spec_version,
            request_seq=row.request_seq,
            stamp_seq=row.stamp_seq,
        )
        expected_checksum = compute_checksum(self._key, content)
        if not hmac.compare_digest(expected_checksum, row.checksum):
            return finish(False, "E_IDENTITY",
                          "盖章校验值与重算结果不一致，记录疑似被篡改",
                          "checksum")

        # 2) 身份摘要演进判定
        if row.identity_digest != identity.identity_digest:
            return finish(
                False, "E_STALE",
                "身份摘要已演进，旧章绑定的身份摘要与当前后端不一致",
                "identity_digest",
            )

        # 3) 令牌口径版本演进判定
        if row.token_spec_version != identity.token_spec_version:
            return finish(
                False, "E_STALE",
                "令牌口径版本已演进，旧章口径与当前后端口径不一致",
                "token_spec_version",
            )

        return finish(True, "OK", "验章通过", "")

    def verify_all(self, spec: Spec) -> list[VerifyResult]:
        results: list[VerifyResult] = []
        for request in spec.requests:
            if self.ledger.get_stamp(request.token_id) is None:
                continue
            results.append(self.verify(request.token_id))
        return results
