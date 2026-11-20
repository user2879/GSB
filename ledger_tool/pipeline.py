"""确定性请求集构造、分组与 NumPy 前向推理.

同一配置多次构造必须得到逐位一致的输入摘要与参数摘要;
所有随机源均来自显式整数种子的 RandomState, 不依赖字典/哈希遍历顺序.
"""

from __future__ import annotations

import numpy as np

from .canon import sha256_hex


class Pipeline:
    """由本地 JSON 配置驱动的确定性推理管线."""

    def __init__(self, config: dict):
        self.config = config
        seed = int(config["seed"])
        n = int(config["num_requests"])
        d = int(config["input_dim"])
        h = int(config["hidden_dim"])
        rng = np.random.RandomState(seed)
        self.inputs = rng.standard_normal((n, d))
        self.w1 = rng.standard_normal((d, h))
        self.bias = rng.standard_normal(h)
        self.w2 = rng.standard_normal((h, d))
        self.params = np.asarray(config["backend_params"], dtype=np.float64)
        if self.params.shape != (h,):
            raise ValueError("backend_params length must equal hidden_dim")
        self.backend_version = str(config["backend_version"])
        self.supported_slots = [int(s) for s in config["supported_slots"]]
        self.latency_base = int(config.get("latency_ms_base", 5))

        self.request_ids = [f"req-{i:04d}" for i in range(n)]
        arrivals = rng.permutation(n)
        slot_choices = rng.randint(0, len(self.supported_slots), n)
        deliveries = []
        for i in range(n):
            deliveries.append({
                "request_id": self.request_ids[i],
                "arrival_seq": int(arrivals[i]),
                "batch_slot": self.supported_slots[int(slot_choices[i])],
                "row": i,
            })
        base_slot = {d_["request_id"]: d_["batch_slot"] for d_ in deliveries}
        row_of = {rid: i for i, rid in enumerate(self.request_ids)}
        for dup in config.get("duplicates", []):
            rid = str(dup["request_id"])
            for seq in dup["extra_arrivals"]:
                deliveries.append({
                    "request_id": rid,
                    "arrival_seq": int(seq),
                    "batch_slot": base_slot[rid],
                    "row": row_of[rid],
                })

        # 重复投递按最小到达序号保留, 被覆盖投递的到达序号作为可回溯引用.
        by_id = {}
        for d_ in deliveries:
            by_id.setdefault(d_["request_id"], []).append(d_)
        self.superseded = {}
        kept = []
        for rid, ds in by_id.items():
            ds.sort(key=lambda x: x["arrival_seq"])
            kept.append(ds[0])
            self.superseded[rid] = [x["arrival_seq"] for x in ds[1:]]
        kept.sort(key=lambda x: (x["arrival_seq"], x["request_id"]))
        self.kept = kept

        groups = {}
        for d_ in kept:
            groups.setdefault(d_["batch_slot"], []).append(d_)
        self.groups = {slot: groups[slot] for slot in sorted(groups)}

        self.input_digest = sha256_hex(b"input\x00" + self.inputs.tobytes())
        self.param_digest = sha256_hex(
            b"param\x00" + self.w1.tobytes() + self.bias.tobytes()
            + self.w2.tobytes() + self.params.tobytes()
        )
        self.backend_fingerprint = sha256_hex(
            b"backend\x00" + self.backend_version.encode("utf-8")
            + b"\x00" + bytes.fromhex(self.param_digest)
        )

        # 逐批前向推理并生成归属视图. 组内按到达序号排序,
        # 因此到达/处理顺序的变化不会改变输出摘要与组内名次.
        self.views = {}
        for slot, members in self.groups.items():
            rows = np.array([self.inputs[m["row"]] for m in members], dtype=np.float64)
            hidden = rows @ self.w1 + (self.bias + self.params)
            hidden = np.maximum(hidden, 0.0)
            out = hidden @ self.w2
            batch_digest = sha256_hex(
                "\x00".join(sorted(m["request_id"] for m in members)).encode("utf-8")
            )
            for rank, m in enumerate(members):
                rid = m["request_id"]
                latency = self.latency_base + (m["arrival_seq"] * 3 + slot) % 17
                output_digest = sha256_hex(
                    b"out\x00" + rid.encode("utf-8") + b"\x00"
                    + bytes.fromhex(self.backend_fingerprint)
                    + bytes.fromhex(self.param_digest)
                    + out[rank].tobytes()
                )
                self.views[rid] = {
                    "request_id": rid,
                    "batch_slot": int(slot),
                    "arrival_seq": int(m["arrival_seq"]),
                    "group_rank": int(rank),
                    "backend_fingerprint": self.backend_fingerprint,
                    "latency_ms": int(latency),
                    "output_digest": output_digest,
                    "superseded_refs": list(self.superseded.get(rid, [])),
                    "fingerprint_snapshot": {
                        "backend_fingerprint": self.backend_fingerprint,
                        "param_digest": self.param_digest,
                        "input_digest": self.input_digest,
                        "batch_digest": batch_digest,
                    },
                }

    def view_for(self, request_id):
        if request_id not in self.views:
            raise KeyError(request_id)
        return dict(self.views[request_id])

    def ordered_views(self):
        return [
            self.views[rid]
            for rid in sorted(self.views, key=lambda r: (self.views[r]["arrival_seq"], r))
        ]
