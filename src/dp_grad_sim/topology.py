from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import SimError
from .util import hash_mapping


@dataclass(frozen=True)
class Link:
    src: int
    dst: int
    bandwidth_bytes: int

    @property
    def key(self) -> str:
        return f"{self.src}->{self.dst}"


@dataclass(frozen=True)
class Topology:
    kind: str
    workers: int
    links: dict[tuple[int, int], Link]
    topology_hash: str

    def next_link(self, src: int) -> Link:
        key = (src, (src + 1) % self.workers)
        try:
            return self.links[key]
        except KeyError as exc:
            raise SimError(
                "E_TOPOLOGY_INVALID", f"ring topology is missing next link {key[0]}->{key[1]}"
            ) from exc

    def link(self, src: int, dst: int) -> Link:
        try:
            return self.links[(src, dst)]
        except KeyError as exc:
            raise SimError("E_TOPOLOGY_INVALID", f"topology has no link {src}->{dst}") from exc

    def link_list(self) -> list[Link]:
        return [self.links[key] for key in sorted(self.links)]

    def budget_map(self, overrides: dict[tuple[int, int], int] | None = None) -> dict[str, int]:
        result = {link.key: link.bandwidth_bytes for link in self.link_list()}
        if overrides:
            for (src, dst), budget in overrides.items():
                link = self.link(src, dst)
                result[link.key] = min(result[link.key], max(0, int(budget)))
        return result


def _extract_budget(item: dict[str, Any], default_budget: int | None) -> int:
    value = item.get(
        "bandwidth_bytes",
        item.get("budget_bytes", item.get("bandwidth_budget_bytes", default_budget)),
    )
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SimError("E_TOPOLOGY_INVALID", "each link requires a positive bandwidth budget")
    return value


def load_topology(path: str | Path, workers: int) -> Topology:
    topology_path = Path(path)
    try:
        raw = json.loads(topology_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SimError("E_TOPOLOGY_INVALID", f"topology file not found: {topology_path}") from exc
    except json.JSONDecodeError as exc:
        raise SimError("E_TOPOLOGY_INVALID", f"invalid topology JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise SimError("E_TOPOLOGY_INVALID", "topology must be a JSON object")

    kind = str(raw.get("type", raw.get("topology", ""))).lower()
    if kind not in {"ring", "full"}:
        raise SimError("E_TOPOLOGY_INVALID", "topology type must be 'ring' or 'full'")
    declared_workers = raw.get("workers", raw.get("data_parallel_workers", workers))
    if declared_workers != workers:
        raise SimError(
            "E_TOPOLOGY_INVALID",
            f"topology declares {declared_workers} workers but config declares {workers}",
        )
    default_budget = raw.get(
        "bandwidth_bytes",
        raw.get("default_bandwidth_bytes", raw.get("link_bandwidth_bytes", raw.get("bandwidth_budget_bytes"))),
    )
    if default_budget is not None and (
        isinstance(default_budget, bool) or not isinstance(default_budget, int) or default_budget <= 0
    ):
        raise SimError("E_TOPOLOGY_INVALID", "default bandwidth must be a positive integer")

    links: dict[tuple[int, int], Link] = {}
    raw_links = raw.get("links")
    if raw_links is None:
        if default_budget is None:
            raise SimError(
                "E_TOPOLOGY_INVALID", "topology requires links or a default bandwidth budget"
            )
        if kind == "ring":
            for src in range(workers):
                for dst in ((src + 1) % workers, (src - 1) % workers):
                    if src == dst:
                        continue
                    links[(src, dst)] = Link(src, dst, default_budget)
        else:
            for src in range(workers):
                for dst in range(workers):
                    if src != dst:
                        links[(src, dst)] = Link(src, dst, default_budget)
    else:
        if not isinstance(raw_links, list):
            raise SimError("E_TOPOLOGY_INVALID", "topology.links must be a list")
        for index, item in enumerate(raw_links):
            if not isinstance(item, dict):
                raise SimError("E_TOPOLOGY_INVALID", f"topology.links[{index}] must be an object")
            src = item.get("src", item.get("from"))
            dst = item.get("dst", item.get("to"))
            if (
                isinstance(src, bool)
                or isinstance(dst, bool)
                or not isinstance(src, int)
                or not isinstance(dst, int)
                or not 0 <= src < workers
                or not 0 <= dst < workers
                or src == dst
            ):
                raise SimError("E_TOPOLOGY_INVALID", f"topology.links[{index}] has invalid endpoints")
            if kind == "ring" and (dst not in {(src + 1) % workers, (src - 1) % workers}):
                raise SimError(
                    "E_TOPOLOGY_INVALID",
                    f"ring link {src}->{dst} is not adjacent to worker {src}",
                )
            budget = _extract_budget(item, default_budget if isinstance(default_budget, int) else None)
            link = Link(src, dst, budget)
            if (src, dst) in links and links[(src, dst)] != link:
                raise SimError("E_TOPOLOGY_INVALID", f"duplicate conflicting link {src}->{dst}")
            links[(src, dst)] = link
            if bool(item.get("bidirectional", kind == "ring")) and (dst, src) not in links:
                links[(dst, src)] = Link(dst, src, budget)

    if kind == "ring":
        for src in range(workers):
            key = (src, (src + 1) % workers)
            if key not in links and workers > 1:
                raise SimError(
                    "E_TOPOLOGY_INVALID", f"ring topology is missing next link {key[0]}->{key[1]}"
                )
    elif workers > 1:
        for src in range(workers):
            for dst in range(workers):
                if src != dst and (src, dst) not in links:
                    raise SimError(
                        "E_TOPOLOGY_INVALID", f"full topology is missing link {src}->{dst}"
                    )

    link_payload = [
        {"src": link.src, "dst": link.dst, "bandwidth_bytes": link.bandwidth_bytes}
        for link in (links[key] for key in sorted(links))
    ]
    topology_hash = hash_mapping({"type": kind, "workers": workers, "links": link_payload})
    return Topology(kind=kind, workers=workers, links=links, topology_hash=topology_hash)


def topology_payload(topology: Topology) -> dict[str, Any]:
    return {
        "type": topology.kind,
        "workers": topology.workers,
        "links": [
            {"src": link.src, "dst": link.dst, "bandwidth_bytes": link.bandwidth_bytes}
            for link in topology.link_list()
        ],
        "topology_hash": topology.topology_hash,
    }

