"""Generate deterministic node/pod sample JSON files used by demo and tests.

The samples deliberately cover:
  * a tainted node that only one pod tolerates;
  * hard node affinity and hard pod anti-affinity;
  * soft affinity / anti-affinity / locality / resource-balance trade-offs;
  * resource exhaustion so at least one node is filtered on headroom;
  * near-tie nodes whose weighted totals fall inside the combined error bound.
"""
from __future__ import annotations

import json
from pathlib import Path

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "samples"


NODES = {
    "nodes": [
        {
            "name": "node-a",
            "labels": {
                "kubernetes.io/hostname": "node-a",
                "zone": "z1",
                "disktype": "ssd",
                "tier": "frontend",
            },
            "cpu": 16,
            "memory": 32,
            "taints": [],
        },
        {
            "name": "node-b",
            "labels": {
                "kubernetes.io/hostname": "node-b",
                "zone": "z1",
                "disktype": "hdd",
                "tier": "backend",
            },
            "cpu": 16,
            "memory": 32,
            "taints": [],
        },
        {
            "name": "node-c",
            "labels": {
                "kubernetes.io/hostname": "node-c",
                "zone": "z2",
                "disktype": "ssd",
                "tier": "backend",
            },
            "cpu": 8,
            "memory": 16,
            "taints": [
                {"key": "dedicated", "value": "gpu", "effect": "NoSchedule"}
            ],
        },
        {
            "name": "node-d",
            "labels": {
                "kubernetes.io/hostname": "node-d",
                "zone": "z2",
                "disktype": "ssd",
                "tier": "frontend",
            },
            "cpu": 8,
            "memory": 16,
            "taints": [],
        },
    ]
}

PODS = {
    "pods": [
        {
            "pod_id": "web-1",
            "cpu_request": 4,
            "memory_request": 8,
            "labels": {"app": "web", "tier": "frontend"},
            "node_affinity": {
                "required": [
                    {"key": "tier", "operator": "In", "values": ["frontend"]}
                ],
                "preferred": [
                    {
                        "weight": 70,
                        "requirement": {
                            "key": "disktype",
                            "operator": "In",
                            "values": ["ssd"],
                        },
                    }
                ],
            },
            "pod_anti_affinity": {
                "required": [],
                "preferred": [
                    {
                        "weight": 50,
                        "requirement": {"key": "app", "operator": "In", "values": ["web"]},
                    }
                ],
                "preferred_topology_key": "kubernetes.io/hostname",
            },
            "tolerations": [],
        },
        {
            "pod_id": "web-2",
            "cpu_request": 4,
            "memory_request": 8,
            "labels": {"app": "web", "tier": "frontend"},
            "node_affinity": {
                "required": [
                    {"key": "tier", "operator": "In", "values": ["frontend"]}
                ],
                "preferred": [
                    {
                        "weight": 70,
                        "requirement": {
                            "key": "disktype",
                            "operator": "In",
                            "values": ["ssd"],
                        },
                    }
                ],
            },
            "pod_anti_affinity": {
                "required": [
                    {"key": "app", "operator": "In", "values": ["web"]}
                ],
                "required_topology_key": "kubernetes.io/hostname",
                "preferred": [],
            },
            "tolerations": [],
        },
        {
            "pod_id": "api-1",
            "cpu_request": 2,
            "memory_request": 4,
            "labels": {"app": "api", "tier": "backend"},
            "node_affinity": {
                "required": [
                    {"key": "tier", "operator": "In", "values": ["backend"]}
                ],
                "preferred": [
                    {
                        "weight": 60,
                        "requirement": {"key": "zone", "operator": "In", "values": ["z1"]},
                    }
                ],
            },
            "pod_anti_affinity": {"required": [], "preferred": []},
            "tolerations": [],
        },
        {
            "pod_id": "gpu-job-1",
            "cpu_request": 6,
            "memory_request": 12,
            "labels": {"app": "train", "tier": "backend"},
            "node_affinity": {
                "required": [
                    {"key": "dedicated", "operator": "DoesNotExist"}
                ],
                "preferred": [],
            },
            "pod_anti_affinity": {"required": [], "preferred": []},
            "tolerations": [
                {
                    "key": "dedicated",
                    "operator": "Equal",
                    "value": "gpu",
                    "effect": "NoSchedule",
                }
            ],
        },
        {
            "pod_id": "batch-1",
            "cpu_request": 7,
            "memory_request": 14,
            "labels": {"app": "batch"},
            "node_affinity": {
                "required": [],
                "preferred": [
                    {
                        "weight": 40,
                        "requirement": {"key": "zone", "operator": "In", "values": ["z2"]},
                    }
                ],
            },
            "pod_anti_affinity": {"required": [], "preferred": []},
            "tolerations": [],
        },
        {
            "pod_id": "web-3",
            "cpu_request": 5,
            "memory_request": 10,
            "labels": {"app": "web", "tier": "frontend"},
            "node_affinity": {
                "required": [
                    {"key": "tier", "operator": "In", "values": ["frontend"]}
                ],
                "preferred": [
                    {
                        "weight": 30,
                        "requirement": {
                            "key": "disktype",
                            "operator": "In",
                            "values": ["hdd"],
                        },
                    }
                ],
            },
            "pod_anti_affinity": {
                "required": [
                    {"key": "app", "operator": "In", "values": ["web"]}
                ],
                "required_topology_key": "kubernetes.io/hostname",
                "preferred": [],
            },
            "tolerations": [],
        },
    ]
}


def main() -> None:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    (SAMPLE_DIR / "nodes.json").write_text(
        json.dumps(NODES, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (SAMPLE_DIR / "pods.json").write_text(
        json.dumps(PODS, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {SAMPLE_DIR / 'nodes.json'}")
    print(f"wrote {SAMPLE_DIR / 'pods.json'}")


if __name__ == "__main__":
    main()
