"""会话快照: 版本识别、内容摘要校验与整体替换恢复(业务码 E_SNAPSHOT)。

快照持久化实例集合版本、故障域版本、事件流内容摘要、各域隔离状态、
实例健康状态与指令账本摘要, 并携带自身内容摘要。重新打开时:
- 摘要校验失败 -> E_SNAPSHOT;
- 版本或摘要同当前状态不一致 -> 整体替换(非增量合并), 旧快照保留为
  只读备份, 并稳定列出被替换的域与实例。
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

from .errors import DiagError
from .util import canonical, digest_obj

SNAPSHOT_NAME = "snapshot.json"
SNAPSHOT_FORMAT_VERSION = 1


def _payload_digest(payload: dict) -> str:
    return digest_obj({k: v for k, v in payload.items() if k != "digest"})


def build_payload(instances_version: str, domains_version: str, events_digest: str,
                  cursor: int, domain_states: dict, instance_states: dict,
                  ledger_digest: str) -> dict:
    payload = {"format_version": SNAPSHOT_FORMAT_VERSION,
               "instances_version": instances_version,
               "domains_version": domains_version,
               "events_digest": events_digest,
               "cursor": cursor,
               "domain_states": domain_states,
               "instance_states": instance_states,
               "ledger_digest": ledger_digest}
    payload["digest"] = _payload_digest(payload)
    return payload


def write_snapshot(workdir, payload: dict) -> Path:
    snapdir = Path(workdir) / "snapshots"
    snapdir.mkdir(parents=True, exist_ok=True)
    path = snapdir / SNAPSHOT_NAME
    path.write_text(canonical(payload), encoding="utf-8")
    return path


def read_snapshot(path) -> dict:
    p = Path(path)
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DiagError("E_SNAPSHOT", f"snapshot unreadable: {exc}",
                        {"path": str(p)})
    if not isinstance(payload, dict) or "digest" not in payload:
        raise DiagError("E_SNAPSHOT", "snapshot missing content digest",
                        {"path": str(p)})
    if _payload_digest(payload) != payload["digest"]:
        raise DiagError("E_SNAPSHOT", "snapshot content digest mismatch",
                        {"path": str(p), "stored_digest": payload["digest"],
                         "computed_digest": _payload_digest(payload)})
    return payload


def _make_readonly_backup(path: Path) -> Path:
    idx = 0
    while True:
        backup = path.with_suffix(f".bak{idx}.json" if idx else ".bak.json")
        if not backup.exists():
            break
        idx += 1
    backup.write_bytes(path.read_bytes())
    os.chmod(backup, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return backup


def open_or_replace(workdir, current_payload: dict) -> tuple[dict, dict | None]:
    """打开快照; 版本/摘要与当前状态不一致时整体替换。

    返回 (有效快照, 替换报告或 None)。
    """
    path = Path(workdir) / "snapshots" / SNAPSHOT_NAME
    if not path.exists():
        return write_snapshot(workdir, current_payload) and current_payload, None
    old = read_snapshot(path)  # 摘要损坏 -> E_SNAPSHOT
    keys = ("instances_version", "domains_version", "events_digest")
    if all(old.get(k) == current_payload[k] for k in keys):
        return old, None
    backup = _make_readonly_backup(path)
    replaced_domains = sorted(d for d in
                              set(old.get("domain_states", {}))
                              | set(current_payload["domain_states"])
                              if old.get("domain_states", {}).get(d)
                              != current_payload["domain_states"].get(d))
    replaced_instances = sorted(i for i in
                                set(old.get("instance_states", {}))
                                | set(current_payload["instance_states"])
                                if old.get("instance_states", {}).get(i)
                                != current_payload["instance_states"].get(i))
    write_snapshot(workdir, current_payload)
    report = {"backup": str(backup),
              "replaced_domains": replaced_domains,
              "replaced_instances": replaced_instances,
              "reason": "version or digest mismatch, whole-replace applied"}
    return current_payload, report


def list_snapshots(workdir) -> list[dict]:
    snapdir = Path(workdir) / "snapshots"
    if not snapdir.exists():
        return []
    out = []
    for p in sorted(snapdir.iterdir()):
        out.append({"name": p.name,
                    "readonly": not os.access(p, os.W_OK),
                    "size": p.stat().st_size})
    return out
