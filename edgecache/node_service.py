"""Edge-node process implementation and JSONL command surface."""

from __future__ import annotations

import os
from typing import Any

from .cache import EdgeCache, monotonic_ms
from .errors import EdgeError
from .model import Policy, b64decode
from .protocol import PROTOCOL_VERSION, request
from .spec import parse_policy


class EdgeNode:
    def __init__(
        self,
        *,
        node_id: str,
        source_host: str,
        source_port: int,
        timeout: float = 2.0,
    ) -> None:
        self.node_id = node_id
        self.source_host = source_host
        self.source_port = source_port
        self.timeout = timeout
        self.policy: Policy | None = None
        self.cache: EdgeCache | None = None

    def hello_payload(self) -> dict[str, Any]:
        return {
            "service": "edge-node",
            "protocol": PROTOCOL_VERSION,
            "pid": os.getpid(),
            "node_id": self.node_id,
            "ledger_version": self.cache.ledger_version if self.cache else None,
        }

    def handle(self, message: dict[str, Any]) -> dict[str, Any]:
        op = message.get("op")
        if op == "hello":
            return {"ok": True, **self.hello_payload()}
        if op == "configure":
            return self._configure(message)
        if op == "sync":
            return self._sync(message)
        if op == "apply_propagation":
            return self._apply_propagation(message)
        if op == "restore":
            return self._restore(message)
        if op == "state":
            return self._state(message)
        if op == "get":
            return self._get(message)
        if op == "shutdown":
            return {"ok": True, **self.hello_payload(), "shutdown": True, "close_connection": True}
        raise EdgeError(
            code="E_PROTOCOL",
            message="unsupported edge-node operation",
            details={"op": op},
        )

    def _configure(self, message: dict[str, Any]) -> dict[str, Any]:
        policy = parse_policy(message.get("policy"))
        ledger_version = _ledger_version(message)
        node = policy.node(self.node_id)
        if self.cache is None:
            self.policy = policy
            self.cache = EdgeCache(
                node_id=self.node_id,
                capacity=node.capacity,
                eviction_rule=node.eviction_rule,
                policy=policy,
                ledger_version=ledger_version,
            )
        elif bool(message.get("preserve_entries", False)):
            self.cache.advance(ledger_version)
            self.cache.configure(policy, ledger_version=ledger_version)
            self.policy = policy
        else:
            self.cache.configure(policy, ledger_version=ledger_version)
            self.policy = policy
        return {
            "ok": True,
            **self.hello_payload(),
            "capacity": node.capacity,
            "eviction_rule": node.eviction_rule,
        }

    def _sync(self, message: dict[str, Any]) -> dict[str, Any]:
        cache = self._require_cache()
        ledger_version = _ledger_version(message)
        expected_version = str(message.get("content_version"))
        snapshot = self._fetch_snapshot()
        if snapshot.get("content_version") != expected_version:
            raise EdgeError(
                code="E_STALE",
                message="content source version changed before node synchronization",
                details={"expected": expected_version, "actual": snapshot.get("content_version")},
            )
        cache.clear(ledger_version=ledger_version, reason="sync_clear")
        for record in sorted(snapshot["content"], key=lambda item: str(item["path"])):
            path = str(record["path"])
            etag = str(record["etag"])
            content = b64decode(str(record["content_b64"]))
            cache.put(
                ledger_version=ledger_version,
                path=path,
                etag=etag,
                content_version=expected_version,
                ttl_ms=int(record["ttl_ms"]),
                content_size=len(content),
            )
        return self._state_payload(ledger_version)

    def _apply_propagation(self, message: dict[str, Any]) -> dict[str, Any]:
        cache = self._require_cache()
        ledger_version = _ledger_version(message)
        trigger_path = str(message.get("trigger_path"))
        expected_version = str(message.get("content_version"))
        expected_paths = message.get("affected_paths")
        if not isinstance(expected_paths, list):
            raise EdgeError(code="E_STATE", message="affected_paths must be an array")
        snapshot = self._fetch_snapshot()
        if snapshot.get("content_version") != expected_version:
            raise EdgeError(
                code="E_STALE",
                message="content source version changed before propagation",
                details={"expected": expected_version, "actual": snapshot.get("content_version")},
            )
        records = {str(record["path"]): record for record in snapshot["content"]}
        if trigger_path not in records:
            raise EdgeError(code="E_STATE", message="trigger path is missing from the source", path=trigger_path)

        from .model import ContentSnapshot

        parsed = ContentSnapshot(
            tuple(
                _record_from_raw(record)
                for record in sorted(snapshot["content"], key=lambda item: str(item["path"]))
            )
        )
        canonical_affected = parsed.descendant_order(trigger_path)
        supplied = [str(path) for path in expected_paths]
        if supplied != canonical_affected:
            raise EdgeError(
                code="E_STALE",
                message="orchestrator propagation order differs from the node's dependency graph",
                path=trigger_path,
                field_path="affected_paths",
                details={"expected": canonical_affected, "actual": supplied},
            )

        invalidated = [trigger_path] + canonical_affected
        for path in invalidated:
            cache.invalidate_path(ledger_version=ledger_version, path=path, reason="propagation_invalidate")
        for path in invalidated:
            record = records[path]
            content = b64decode(str(record["content_b64"]))
            cache.put(
                ledger_version=ledger_version,
                path=path,
                etag=str(record["etag"]),
                content_version=expected_version,
                ttl_ms=int(record["ttl_ms"]),
                content_size=len(content),
            )
        cache.rebase_content_version(
            ledger_version=ledger_version,
            content_version=expected_version,
            valid_etags={str(record["path"]): str(record["etag"]) for record in snapshot["content"]},
        )
        payload = self._state_payload(ledger_version)
        payload["trigger_path"] = trigger_path
        payload["affected_paths"] = canonical_affected
        return payload

    def _restore(self, message: dict[str, Any]) -> dict[str, Any]:
        ledger_version = _ledger_version(message)
        policy = parse_policy(message.get("policy"))
        node = policy.node(self.node_id)
        if self.cache is None:
            self.cache = EdgeCache(
                node_id=self.node_id,
                capacity=node.capacity,
                eviction_rule=node.eviction_rule,
                policy=policy,
                ledger_version=ledger_version,
            )
        self.policy = policy
        self.cache.configure(policy, ledger_version=max(ledger_version, self.cache.ledger_version))
        if self.cache.ledger_version != ledger_version:
            # configure may only clear when the ledger advances; a restore is
            # always against one exact ledger and must not silently retarget.
            if ledger_version < self.cache.ledger_version:
                raise EdgeError(
                    code="E_STALE",
                    message="snapshot ledger version is older than the node's active version",
                    details={"requested": ledger_version, "active": self.cache.ledger_version},
                )
            self.cache.bind(ledger_version)
        entries = message.get("entries")
        if not isinstance(entries, list):
            raise EdgeError(code="E_STATE", message="restore entries must be an array")
        self.cache.restore(
            ledger_version=ledger_version,
            entries=[dict(item) for item in entries],
            normalize_snapshot_ttl=bool(message.get("normalize_snapshot_ttl", False)),
        )
        return self._state_payload(ledger_version)

    def _state(self, message: dict[str, Any]) -> dict[str, Any]:
        ledger_version = _ledger_version(message)
        return self._state_payload(ledger_version, expire=bool(message.get("expire", True)))

    def _state_payload(self, ledger_version: int, *, expire: bool = True) -> dict[str, Any]:
        cache = self._require_cache()
        entries = cache.entry_state(ledger_version=ledger_version, expire=expire)
        return {
            "ok": True,
            "service": "edge-node",
            "node_id": self.node_id,
            "ledger_version": ledger_version,
            "content_version": entries[0]["content_version"] if entries else None,
            "entries": [{"node_id": self.node_id, **entry} for entry in entries],
            "evictions": cache.events(ledger_version=ledger_version),
            "monotonic_now_ms": monotonic_ms(),
        }

    def _get(self, message: dict[str, Any]) -> dict[str, Any]:
        cache = self._require_cache()
        ledger_version = _ledger_version(message)
        entry = cache.get(
            ledger_version=ledger_version,
            path=str(message.get("path")),
            etag=str(message.get("etag")),
        )
        return {
            "ok": True,
            "hit": entry is not None,
            "entry": {"node_id": self.node_id, **entry.to_dict()} if entry else None,
            "ledger_version": ledger_version,
        }

    def _fetch_snapshot(self) -> dict[str, Any]:
        return request(
            self.source_host,
            self.source_port,
            {"op": "snapshot", "protocol": PROTOCOL_VERSION},
            timeout=self.timeout,
        )

    def _require_cache(self) -> EdgeCache:
        if self.cache is None:
            raise EdgeError(code="E_STATE", message="node has not been configured")
        return self.cache


def _ledger_version(message: dict[str, Any]) -> int:
    value = message.get("ledger_version")
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise EdgeError(code="E_STALE", message="ledger_version must be a positive integer")
    return value


def _record_from_raw(raw: dict[str, Any]):
    from .model import ContentRecord, Derivation

    return ContentRecord(
        path=str(raw["path"]),
        etag=str(raw["etag"]),
        content=b64decode(str(raw["content_b64"])),
        ttl_ms=int(raw["ttl_ms"]),
        derived_from=tuple(str(item) for item in raw.get("derived_from", [])),
        derivation=(
            Derivation(
                rule=str(raw["derivation"]["rule"]),
                separator=str(raw["derivation"].get("separator", "")),
            )
            if raw.get("derivation")
            else None
        ),
    )


__all__ = ["EdgeNode"]
