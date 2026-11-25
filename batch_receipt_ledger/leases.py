"""批处理租约：签发、续约乐观校验、真实过期与配额回收（需求 5）。

配额回收只有一个守卫路径 :meth:`LeaseManager._recycle`，到期回收与
批次撤销回收都经过它，通过租约状态机保证同一份并发配额不会被两次回收。
"""

import threading

from .errors import LeaseError

STATE_HELD = "held"
STATE_RELEASED_EXPIRED = "released_expired"
STATE_RELEASED_DONE = "released_done"
STATE_RELEASED_CANCELLED = "released_cancelled"


class Lease:
    def __init__(self, lease_id, batch_id, worker_id, lease_ms, now_ms):
        self.lease_id = lease_id
        self.batch_id = batch_id
        self.worker_id = worker_id
        self.lease_ms = int(lease_ms)
        self.version = 1
        self.granted_at = now_ms
        self.expires_at = now_ms + int(lease_ms)
        self.state = STATE_HELD
        self.renew_count = 0
        self.history = [
            {"version": 1, "at": now_ms, "expires_at": self.expires_at,
             "event": "grant"}
        ]

    def is_expired(self, now_ms):
        return self.state == STATE_HELD and now_ms >= self.expires_at

    def to_dict(self):
        return {
            "lease_id": self.lease_id,
            "batch_id": self.batch_id,
            "worker_id": self.worker_id,
            "version": self.version,
            "granted_at": self.granted_at,
            "expires_at": self.expires_at,
            "state": self.state,
            "renew_count": self.renew_count,
            "history": list(self.history),
        }


class LeaseManager:
    """维护每工作进程并发配额、租约表与后台过期回收。"""

    def __init__(self, spec, clock, on_lease_change=None):
        self.clock = clock
        self.spec = spec
        self._cond = threading.Condition()
        self._quotas = {
            w.worker_id: spec.effective_quota(w.worker_id) for w in spec.workers
        }
        self._leases = {}
        self._on_lease_change = on_lease_change
        self._stopped = False
        self._reaper = None
        self._seq = 0

    def start_reaper(self, interval_ms=10):
        self._reaper = threading.Thread(
            target=self._reap_loop, args=(interval_ms,), daemon=True,
            name="lease-reaper")
        self._reaper.start()

    def stop_reaper(self):
        with self._cond:
            self._stopped = True
            self._cond.notify_all()
        if self._reaper is not None:
            self._reaper.join(timeout=2)

    def _reap_loop(self, interval_ms):
        while True:
            with self._cond:
                if self._stopped:
                    return
                self._cond.wait(interval_ms / 1000.0)
                if self._stopped:
                    return
                now_ms = self.clock.now_ms()
                for lease in list(self._leases.values()):
                    if lease.is_expired(now_ms):
                        self._recycle(lease, STATE_RELEASED_EXPIRED, now_ms)

    def acquire(self, batch_id, worker_id, wait_timeout_ms=None):
        """在配额内签发租约；配额不足则等待，返回 (lease, wait_ms)。"""
        import time

        with self._cond:
            start = self.clock.now_ms()
            virtual_clock = type(self.clock).__name__ == "FakeClock"
            wall_deadline = (None if wait_timeout_ms is None
                             else time.monotonic() + wait_timeout_ms / 1000.0)
            deadline = None if wait_timeout_ms is None else start + wait_timeout_ms
            while not self._stopped:
                self._sweep_expired_locked()
                if self._free_quota_locked(worker_id) > 0:
                    self._seq += 1
                    lease_id = f"lease-{batch_id}-{self._seq}"
                    lease = Lease(
                        lease_id, batch_id, worker_id,
                        self.spec.lease_ms, self.clock.now_ms())
                    self._leases[lease_id] = lease
                    self._notify_locked(lease)
                    return lease, self.clock.now_ms() - start
                if virtual_clock and wall_deadline is not None \
                        and time.monotonic() >= wall_deadline:
                    self.clock.advance(wait_timeout_ms + 1)
                    return None, self.clock.now_ms() - start
                if not virtual_clock and deadline is not None \
                        and self.clock.now_ms() >= deadline:
                    return None, self.clock.now_ms() - start
                if virtual_clock:
                    self.clock.advance(1)
                    self._cond.wait(0.01)
                else:
                    remaining = None
                    if deadline is not None:
                        remaining = max(0.001,
                                        (deadline - self.clock.now_ms()) / 1000.0)
                    self._cond.wait(remaining)
            raise LeaseError("租约管理器已停止", field="manager")

    def renew(self, lease_id, expected_version, renew_ms, worker_client=None):
        """续约：先向工作进程做版本号乐观校验，再本地更新。"""
        with self._cond:
            lease = self._leases.get(lease_id)
            if lease is None:
                raise LeaseError(f"租约 {lease_id!r} 不存在", field="lease_id")
            if lease.state != STATE_HELD:
                raise LeaseError(
                    f"租约 {lease_id!r} 已处于 {lease.state}，不可续约",
                    field="state")
            now_ms = self.clock.now_ms()
            if now_ms >= lease.expires_at:
                self._recycle(lease, STATE_RELEASED_EXPIRED, now_ms)
                raise LeaseError(f"租约 {lease_id!r} 已过期", field="expires_at")
            if lease.version != int(expected_version):
                raise LeaseError(
                    f"租约版本不符: 当前 {lease.version}, 期望 {expected_version}",
                    field="lease_version")

        if worker_client is not None:
            new_version = lease.version + 1
            reply = worker_client.request({
                "type": "lease_renew",
                "lease_id": lease_id,
                "expected_version": lease.version,
                "new_version": new_version,
            })
            if reply.get("type") == "error" and reply.get("code") == "E_LEASE":
                raise LeaseError(reply["message"], field="lease_version")

        with self._cond:
            now_ms = self.clock.now_ms()
            if lease.state != STATE_HELD:
                raise LeaseError("续约期间租约被回收", field="state")
            if lease.version != int(expected_version):
                raise LeaseError("续约期间租约版本发生变化", field="lease_version")
            lease.version += 1
            lease.renew_count += 1
            lease.expires_at = now_ms + int(renew_ms)
            lease.history.append(
                {"version": lease.version, "at": now_ms,
                 "expires_at": lease.expires_at, "event": "renew"})
            self._notify_locked(lease)
            return lease

    def grant_on_worker(self, lease, worker_client):
        """把租约版本注册到工作进程（租约签发的远端部分）。"""
        reply = worker_client.request({
            "type": "lease_grant",
            "lease_id": lease.lease_id,
            "lease_version": lease.version,
        })
        if reply.get("type") != "lease_ack":
            self.release(lease.lease_id, STATE_RELEASED_CANCELLED)
            raise LeaseError(f"工作进程拒绝租约: {reply}", field="lease_grant")

    def release(self, lease_id, state=STATE_RELEASED_DONE):
        with self._cond:
            lease = self._leases.get(lease_id)
            if lease is None:
                return None
            return self._recycle(lease, state, self.clock.now_ms())

    def cancel(self, lease_id):
        return self.release(lease_id, STATE_RELEASED_CANCELLED)

    def _recycle(self, lease, target_state, now_ms):
        """唯一配额回收路径：状态机保证幂等，杜绝重复回收。"""
        if lease.state != STATE_HELD:
            return lease
        lease.state = target_state
        lease.history.append(
            {"version": lease.version, "at": now_ms, "event": target_state})
        self._cond.notify_all()
        self._notify_locked(lease)
        return lease

    def _sweep_expired_locked(self):
        now_ms = self.clock.now_ms()
        for lease in list(self._leases.values()):
            if lease.is_expired(now_ms):
                self._recycle(lease, STATE_RELEASED_EXPIRED, now_ms)

    def _free_quota_locked(self, worker_id):
        quota = self._quotas[worker_id]
        used = sum(
            1 for lease in self._leases.values()
            if lease.worker_id == worker_id and lease.state == STATE_HELD)
        return quota - used

    def _notify_locked(self, lease):
        if self._on_lease_change is not None:
            self._on_lease_change({
                "event": "lease",
                "lease_id": lease.lease_id,
                "batch_id": lease.batch_id,
                "worker_id": lease.worker_id,
                "version": lease.version,
                "state": lease.state,
                "expires_at": lease.expires_at,
            })

    def active_count(self):
        with self._cond:
            return sum(1 for lease in self._leases.values()
                       if lease.state == STATE_HELD)

    def held_leases(self):
        with self._cond:
            return [lease for lease in self._leases.values()
                    if lease.state == STATE_HELD]

    def all_leases(self):
        with self._cond:
            return list(self._leases.values())

    def snapshot_leases(self):
        with self._cond:
            return {lease_id: lease.to_dict()
                    for lease_id, lease in self._leases.items()}

    def get(self, lease_id):
        with self._cond:
            return self._leases.get(lease_id)
