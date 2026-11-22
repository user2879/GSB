"""恢复后一致性重放校验。

从快照恢复后，按账本记录的请求序列与内容版本重新驱动节点，
逐请求比较命中来源、返回内容版本、剩余 TTL、逐出事件与传播轨迹；
首个不一致定位到批次、节点、路径与字段，以 E_REPLAY 报明依据。

录制时每个批次为两阶段（集中查找 -> 统一回源写入），重放按相同
阶段语义驱动，保证逐字节一致。
"""

import base64

from .errors import ReplayError


class Replayer:
    def __init__(self, orchestrator):
        self.orch = orchestrator
        self.ledger = orchestrator.ledger
        self.policy = orchestrator.policy
        self.compared = 0

    def _fail(self, batch, node, path, field, expected, actual):
        raise ReplayError(
            "重放与账本不一致",
            batch=batch,
            node=node,
            path=path,
            field=field,
            expected=expected,
            actual=actual,
        )

    def _cmp(self, batch, node, path, field, expected, actual):
        if expected != actual:
            self._fail(batch, node, path, field, expected, actual)
        self.compared += 1

    def run(self):
        events = self.ledger.events()
        contents = self.ledger.load_contents().contents
        i = 0
        n = len(events)
        while i < n:
            ev = events[i]
            if ev["kind"] == "request":
                # 收集同一批次的连续请求事件，按两阶段重放
                batch_no = ev["payload"]["batch"]
                group = []
                while (
                    i < n
                    and events[i]["kind"] == "request"
                    and events[i]["payload"]["batch"] == batch_no
                ):
                    group.append(events[i]["payload"])
                    i += 1
                self._replay_batch(group)
                continue
            if ev["kind"] == "propagation":
                self._replay_propagation(ev["payload"], contents)
            i += 1
        return {"compared_fields": self.compared, "verdict": "OK"}

    def _replay_batch(self, records):
        # 阶段一：全部查找
        lookups = []
        for rec in records:
            node = self.orch.nodes[rec["node"]]
            rep = node.call(
                {"type": "request", "path": rec["path"], "tick_ms": rec["tick_ms"]},
                self.policy.pull_timeout_ms,
            )
            lookups.append((rec, rep))
        # 阶段二：未命中/过期按账本记录的内容版本写入
        for rec, rep in lookups:
            self._replay_one(rec, rep)

    def _replay_one(self, rec, rep):
        b, n, p = rec["batch"], rec["node"], rec["path"]
        node = self.orch.nodes[n]
        self._cmp(b, n, p, "result", rec["result"], rep["result"])
        replayed_evictions = []
        if rep.get("expired_evicted"):
            replayed_evictions.append(
                {"cache_key": rep["expired_evicted"]["cache_key"], "reason": "expired"}
            )
        if rec["result"] == "hit":
            self._cmp(b, n, p, "source", rec["source"], rep.get("source"))
            self._cmp(b, n, p, "etag", rec["etag"], rep.get("etag"))
            self._cmp(
                b, n, p, "ttl_remaining_ms",
                rec["ttl_remaining_ms"], rep.get("ttl_remaining_ms"),
            )
        else:
            stored = rec["stored"]
            rep_stored = node.call(
                {
                    "type": "store_replay",
                    "path": p,
                    "etag": stored["etag"],
                    "ttl_ms": stored["ttl_ms"],
                    "tick_ms": rec["tick_ms"],
                    "source": rec["source"],
                },
                self.policy.pull_timeout_ms,
            )
            self._cmp(b, n, p, "etag", rec["etag"], rep_stored["etag"])
            self._cmp(b, n, p, "stored_etag", stored["etag"], rep_stored["etag"])
            self._cmp(
                b, n, p, "entry_version",
                stored["entry_version"], rep_stored["entry_version"],
            )
            for ev in rep_stored["evictions"]:
                replayed_evictions.append(
                    {"cache_key": ev["cache_key"], "reason": "capacity"}
                )
        self._cmp(b, n, p, "evictions", rec["evictions"], replayed_evictions)

    def _replay_propagation(self, rec, contents):
        node = self.orch.nodes[rec["node"]]
        path = rec["affected_path"]
        content = contents[path]
        rep = node.call(
            {
                "type": "invalidate_replace",
                "path": path,
                "etag": rec["new_etag"],
                "body_base64": base64.b64encode(content.body).decode("ascii"),
                "ttl_ms": content.ttl_ms,
                "tick_ms": 0,
            },
            self.policy.pull_timeout_ms,
        )
        b = rec["propagation_id"]
        n = rec["node"]
        self._cmp(b, n, path, "present", rec["present"], rep["present"])
        self._cmp(
            b, n, path, "old_entry_version",
            rec["old_entry_version"], rep["old_entry_version"],
        )
        self._cmp(
            b, n, path, "new_entry_version",
            rec["new_entry_version"], rep["new_entry_version"],
        )
