"""生成确定性演示样本（合法声明 + 覆盖 E_SPEC/E_FIELD/E_REF 的非法样例）。"""

import copy
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def base_spec():
    deliveries = []
    for index in range(1, 13):
        event = {
            "request_id": f"req-{index:03d}",
            "arrival_seq": index,
            "value": 1000 + index,
            "replaces": None,
        }
        deliveries.append(event)
    deliveries.append({
        "request_id": "req-013",
        "arrival_seq": 13,
        "value": 9999,
        "replaces": "req-003",
    })
    return {
        "schema_version": 1,
        "model": {
            "input_dim": 16,
            "hidden_dim": 8,
            "seed": 20261126,
            "batch_size": 3,
        },
        "limits": {
            "lease_ms": 800,
            "global_max_active_leases": 3,
            "connect_timeout_ms": 2000,
            "read_timeout_ms": 3000,
        },
        "merge_rule": {"supersede_field": "replaces",
                       "policy": "exclude_superseded"},
        "workers": [
            {"worker_id": "worker-a", "host": "127.0.0.1", "port": 39001,
             "concurrency": 2, "param_version": 1},
            {"worker_id": "worker-b", "host": "127.0.0.1", "port": 39002,
             "concurrency": 1, "param_version": 1},
        ],
        "deliveries": deliveries,
        "quota_events": [
            {"event_id": "qe-1", "worker_id": "worker-b", "concurrency": 2},
        ],
    }


INVALID_SPECS = {
    "bad_duplicate_worker.json": ("workers/1", "worker_id",
                                  lambda s: s["workers"][1].__setitem__(
                                      "worker_id", "worker-a")),
    "bad_host_literal.json": ("workers/0", "host",
                              lambda s: s["workers"][0].__setitem__(
                                  "host", "localhost")),
    "bad_duplicate_port.json": ("workers/1", "port",
                                lambda s: s["workers"][1].__setitem__(
                                    "port", 39001)),
    "bad_dimension.json": ("model", "hidden_dim",
                           lambda s: s["model"].__setitem__("hidden_dim", 0)),
    "bad_lease_ms.json": ("limits", "lease_ms",
                          lambda s: s["limits"].__setitem__("lease_ms", 0)),
    "bad_concurrency.json": ("workers/0", "concurrency",
                             lambda s: s["workers"][0].__setitem__(
                                 "concurrency", -1)),
    "bad_duplicate_request.json": ("deliveries/2", "request_id",
                                   lambda s: s["deliveries"][2].__setitem__(
                                       "request_id", "req-001")),
}


def main():
    os.makedirs(HERE, exist_ok=True)
    valid = base_spec()
    with open(os.path.join(HERE, "spec_valid.json"), "w", encoding="utf-8") as handle:
        json.dump(valid, handle, ensure_ascii=False, indent=2, sort_keys=True)

    for filename, (_location, _field, mutate) in INVALID_SPECS.items():
        broken = copy.deepcopy(valid)
        mutate(broken)
        with open(os.path.join(HERE, filename), "w", encoding="utf-8") as handle:
            json.dump(broken, handle, ensure_ascii=False, indent=2, sort_keys=True)

    evolved = copy.deepcopy(valid)
    evolved["workers"][0]["param_version"] = 2
    with open(os.path.join(HERE, "spec_param_v2.json"), "w",
              encoding="utf-8") as handle:
        json.dump(evolved, handle, ensure_ascii=False, indent=2, sort_keys=True)

    repartitioned = copy.deepcopy(valid)
    repartitioned["model"]["batch_size"] = 4
    with open(os.path.join(HERE, "spec_batch_size_4.json"), "w",
              encoding="utf-8") as handle:
        json.dump(repartitioned, handle, ensure_ascii=False, indent=2,
                  sort_keys=True)

    print(f"wrote samples under {HERE}")


if __name__ == "__main__":
    main()
