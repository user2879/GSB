"""测试共享工具：临时端口分配与声明构造。"""

import copy
import json
import socket

from tests.test_bootstrap import kill_orphan_workers

kill_orphan_workers()


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def make_spec(workers=2, deliveries=9, batch_size=3, seed=20261126,
              lease_ms=1000, global_max=3, start_port=None):
    if start_port is None:
        ports = [free_port() for _ in range(workers)]
    else:
        ports = [start_port + index for index in range(workers)]
    worker_list = [
        {"worker_id": f"worker-{chr(97 + index)}", "host": "127.0.0.1",
         "port": ports[index], "concurrency": 2, "param_version": 1}
        for index in range(workers)
    ]
    delivery_list = [
        {"request_id": f"req-{index:03d}", "arrival_seq": index + 1,
         "value": 1000 + index, "replaces": None}
        for index in range(deliveries)
    ]
    return {
        "schema_version": 1,
        "model": {"input_dim": 16, "hidden_dim": 8, "seed": seed,
                  "batch_size": batch_size},
        "limits": {"lease_ms": lease_ms,
                   "global_max_active_leases": global_max,
                   "connect_timeout_ms": 1000, "read_timeout_ms": 2000},
        "merge_rule": {"supersede_field": "replaces",
                       "policy": "exclude_superseded"},
        "workers": worker_list,
        "deliveries": delivery_list,
        "quota_events": [],
    }


def clone(spec_dict, **changes):
    copied = copy.deepcopy(spec_dict)
    for dotted, value in changes.items():
        section, key = dotted.split(".")
        copied[section][key] = value
    return copied


def dumps(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)
