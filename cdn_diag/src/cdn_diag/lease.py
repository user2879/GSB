"""分片下发租约：签发、续约、过期回收与冲突裁决。

过期判定使用本地真实单调时钟（time.monotonic）。
冲突裁决规则（显式且稳定）：同一路径同一节点出现指向不同内容版本的
有效租约时，新内容版本优先，旧租约被撤销并记录依据。
"""

import itertools
import time

from .errors import LeaseError

LEASE_ACTIVE = "active"
LEASE_EXPIRED = "expired"
LEASE_REVOKED = "revoked_superseded"  # 被更新内容版本取代
LEASE_CONFLICT_RULE = "newer_content_version_wins"


def _now_mono_ms():
    return int(time.monotonic() * 1000)


class LeaseRegistry:
    def __init__(self):
        self._leases = {}
        self._seq = itertools.count(1)
        self.conflict_log = []

    def _check_expired(self, lease):
        if lease["status"] == LEASE_ACTIVE and _now_mono_ms() >= lease["expires_mono_ms"]:
            lease["status"] = LEASE_EXPIRED
        return lease["status"]

    def issue(self, node_id, path_versions, ttl_ms):
        """path_versions: {path: etag}。返回新租约；冲突按稳定规则裁决。"""
        conflicts = []
        for existing in self._leases.values():
            if existing["node_id"] != node_id:
                continue
            if self._check_expired(existing) != LEASE_ACTIVE:
                continue
            for path, etag in existing["content_versions"].items():
                new_etag = path_versions.get(path)
                if new_etag is not None and new_etag != etag:
                    existing["status"] = LEASE_REVOKED
                    conflicts.append(
                        {
                            "rule": LEASE_CONFLICT_RULE,
                            "node_id": node_id,
                            "path": path,
                            "revoked_lease_id": existing["lease_id"],
                            "old_content_version": etag,
                            "new_content_version": new_etag,
                        }
                    )
                    break
        now = _now_mono_ms()
        lease_id = f"L{next(self._seq):06d}"
        lease = {
            "lease_id": lease_id,
            "node_id": node_id,
            "paths": sorted(path_versions),
            "content_versions": dict(path_versions),
            "lease_version": 1,
            "issued_mono_ms": now,
            "expires_mono_ms": now + ttl_ms,
            "status": LEASE_ACTIVE,
        }
        self._leases[lease_id] = lease
        self.conflict_log.extend(conflicts)
        return lease, conflicts

    def renew(self, lease_id, lease_version, ttl_ms):
        lease = self._leases.get(lease_id)
        if lease is None:
            raise LeaseError("租约不存在", lease_id=lease_id)
        if lease["lease_version"] != lease_version:
            raise LeaseError(
                "租约版本不符，乐观校验失败",
                lease_id=lease_id,
                expected_version=lease["lease_version"],
                got_version=lease_version,
            )
        if self._check_expired(lease) != LEASE_ACTIVE:
            raise LeaseError("租约已失效，不可续约", lease_id=lease_id, status=lease["status"])
        lease["lease_version"] += 1
        lease["expires_mono_ms"] = _now_mono_ms() + ttl_ms
        return lease

    def validate_for_pull(self, lease_id, lease_version, path, etag):
        lease = self._leases.get(lease_id)
        if lease is None:
            raise LeaseError("拉取凭据无效：租约不存在", lease_id=lease_id)
        if lease["lease_version"] != lease_version:
            raise LeaseError(
                "拉取凭据无效：租约版本不符",
                lease_id=lease_id,
                expected_version=lease["lease_version"],
                got_version=lease_version,
            )
        if self._check_expired(lease) != LEASE_ACTIVE:
            raise LeaseError("租约已过期，回收后拒绝下发", lease_id=lease_id, status=lease["status"])
        if path not in lease["paths"]:
            raise LeaseError("路径不在租约可下发集合内", lease_id=lease_id, path=path)
        if lease["content_versions"].get(path) != etag:
            raise LeaseError(
                "租约内容版本与请求不符",
                lease_id=lease_id,
                path=path,
                lease_content_version=lease["content_versions"].get(path),
                requested=etag,
            )
        return True

    def reclaim_expired(self):
        reclaimed = []
        for lease in self._leases.values():
            if lease["status"] == LEASE_ACTIVE and _now_mono_ms() >= lease["expires_mono_ms"]:
                lease["status"] = LEASE_EXPIRED
                reclaimed.append(lease["lease_id"])
        return sorted(reclaimed)

    def dump(self):
        return [self._leases[k] for k in sorted(self._leases)]
