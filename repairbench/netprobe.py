"""127.0.0.1 回环端口探测：确认本机允许在声明端口启动监听。"""

from __future__ import annotations

import socket

from .spec import GroupSpec


def probe_ports(groups: list, timeout_s: float = 0.5) -> dict:
    """逐个绑定声明的回环端口并监听一次，确认本地环境可用。"""
    results = []
    for group in groups:
        for node in group.nodes:
            entry = {
                "group_id": group.group_id,
                "node_id": node.node_id,
                "address": f"{node.host}:{node.port}",
            }
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((node.host, node.port))
                sock.listen(1)
                sock.settimeout(timeout_s)
                entry["listenable"] = True
            except OSError as err:
                entry["listenable"] = False
                entry["error"] = str(err)
            finally:
                sock.close()
            results.append(entry)
    return {
        "ok": all(r["listenable"] for r in results),
        "ports": results,
    }
