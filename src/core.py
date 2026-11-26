"""确定性定点推理核心：定点口径、参数生成、请求样本、批划分与前向计算。

定点口径（显式声明，随参数摘要进入回执，可被复核）：
- 字长：32 位有符号整数（int32），中间累加使用 int64；
- 舍入方向：向负无穷取整（floor），通过对 int64 累加值算术右移 frac_bits 实现；
- 溢出规则：饱和（saturate），截断到 int32 区间 [-2^31, 2^31 - 1]。

确定性保证：请求输入仅由 (样本种子, 请求标识) 派生；参数仅由
(工作进程种子, 维度, 定点口径) 派生；批划分按到达序号排序切分；输出摘要
对批内请求标识排序后计算。因此与派发顺序、字典遍历顺序、批执行顺序无关。
"""

import hashlib
import json
import time
from dataclasses import dataclass

import numpy as np

INT32_MIN = -(2 ** 31)
INT32_MAX = 2 ** 31 - 1


def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(data):
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def monotonic_ms():
    return time.monotonic_ns() // 1_000_000


@dataclass(frozen=True)
class FixedPointSpec:
    frac_bits: int = 8
    word_bits: int = 32
    rounding: str = "floor"
    saturation: str = "saturate"

    @staticmethod
    def from_config(d):
        spec = FixedPointSpec(
            frac_bits=int(d.get("frac_bits", 8)),
            word_bits=int(d.get("word_bits", 32)),
            rounding=str(d.get("rounding", "floor")),
            saturation=str(d.get("saturation", "saturate")),
        )
        spec.validate()
        return spec

    def validate(self):
        if self.word_bits != 32:
            raise ValueError("word_bits 仅支持 32")
        if not 0 <= self.frac_bits < self.word_bits:
            raise ValueError("frac_bits 越界")
        if self.rounding != "floor":
            raise ValueError("rounding 仅支持 floor（向负无穷取整）")
        if self.saturation != "saturate":
            raise ValueError("saturation 仅支持 saturate（饱和到 int32）")

    def to_dict(self):
        return {
            "frac_bits": self.frac_bits,
            "word_bits": self.word_bits,
            "rounding": self.rounding,
            "saturation": self.saturation,
        }


def floor_shift(values, frac_bits):
    """向负无穷取整的算术右移（显式舍入方向）。"""
    return np.floor_divide(values, np.int64(1) << frac_bits)


def saturate_int32(values):
    """饱和到 int32 区间（显式溢出规则）。"""
    return np.clip(values, INT32_MIN, INT32_MAX).astype(np.int32)


@dataclass(frozen=True)
class Request:
    request_id: str
    seq: int


@dataclass(frozen=True)
class PlannedBatch:
    batch_id: str
    worker_id: str
    param_version: int
    request_ids: tuple


def request_input(sample_seed, request_id, input_dim):
    """由 (样本种子, 请求标识) 派生确定性整数输入向量。"""
    digest = hashlib.sha256(f"{sample_seed}:{request_id}".encode("utf-8")).digest()
    sub_seed = int.from_bytes(digest[:8], "little") % (2 ** 32)
    rng = np.random.RandomState(sub_seed)
    return rng.randint(-256, 256, size=input_dim).astype(np.int32)


def gen_requests(sample_seed, count):
    return [Request(request_id=f"req-{i:04d}", seq=i) for i in range(count)]


def chunk_batches(requests, batch_size):
    """按到达序号稳定切批；与调用方传入顺序无关。"""
    ordered = sorted(requests, key=lambda r: (r.seq, r.request_id))
    batches = []
    for idx in range(0, len(ordered), batch_size):
        part = ordered[idx:idx + batch_size]
        batches.append((f"batch-{idx // batch_size:04d}", tuple(r.request_id for r in part)))
    return batches


def assign_workers(batch_ids, worker_ids):
    """确定性派工：批序号对排序后的工作进程数取模。"""
    ordered_workers = sorted(worker_ids)
    return {bid: ordered_workers[i % len(ordered_workers)] for i, bid in enumerate(batch_ids)}


def param_spec_dict(worker_id, worker_seed, cfg):
    return {
        "worker_id": worker_id,
        "worker_seed": int(worker_seed),
        "input_dim": int(cfg["input_dim"]),
        "hidden_dim": int(cfg["hidden_dim"]),
        "output_dim": int(cfg["output_dim"]),
        "fixed_point": FixedPointSpec.from_config(cfg["fixed_point"]).to_dict(),
    }


def spec_digest(spec_dict):
    return sha256_hex(canonical_json(spec_dict))


@dataclass(frozen=True)
class Params:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray


def generate_params(spec_dict):
    """按工作进程种子与维度生成确定性整数参数。"""
    rng = np.random.RandomState(int(spec_dict["worker_seed"]))
    input_dim = spec_dict["input_dim"]
    hidden_dim = spec_dict["hidden_dim"]
    output_dim = spec_dict["output_dim"]
    w1 = rng.randint(-64, 64, size=(input_dim, hidden_dim)).astype(np.int32)
    b1 = rng.randint(-4096, 4096, size=hidden_dim).astype(np.int32)
    w2 = rng.randint(-64, 64, size=(hidden_dim, output_dim)).astype(np.int32)
    b2 = rng.randint(-4096, 4096, size=output_dim).astype(np.int32)
    return Params(w1=w1, b1=b1, w2=w2, b2=b2)


def param_digest(spec_dict, params):
    """参数摘要：绑定参数口径声明与全部参数字节。"""
    h = hashlib.sha256()
    h.update(canonical_json(spec_dict).encode("utf-8"))
    for arr in (params.w1, params.b1, params.w2, params.b2):
        h.update(arr.astype("<i4").tobytes())
    return h.hexdigest()


def forward(params, fp_spec, x_int32):
    """单层隐藏层定点前向：floor 舍入 + int32 饱和。"""
    x = x_int32.astype(np.int64)
    acc1 = x @ params.w1.astype(np.int64) + params.b1.astype(np.int64)
    h = saturate_int32(floor_shift(acc1, fp_spec.frac_bits))
    h = np.maximum(h, np.int32(0))
    acc2 = h.astype(np.int64) @ params.w2.astype(np.int64) + params.b2.astype(np.int64)
    return saturate_int32(floor_shift(acc2, fp_spec.frac_bits))


def output_digest(request_ids, outputs, param_version, p_digest):
    """输出摘要：绑定参与请求标识集合、参数版本序号与参数摘要。"""
    lines = [
        f"{rid}:{outputs[rid].astype('<i4').tobytes().hex()}"
        for rid in sorted(request_ids)
    ]
    payload = canonical_json({
        "param_version": int(param_version),
        "param_digest": p_digest,
        "outputs": lines,
    })
    return sha256_hex(payload)


def compute_batch(cfg, worker_id, param_version, batch_id, request_ids):
    """计算一个批次的输出摘要（worker 子进程、校验与重放共用的唯一入口）。"""
    spec = param_spec_dict(worker_id, cfg["workers_by_id"][worker_id]["seed"], cfg)
    params = generate_params(spec)
    p_digest = param_digest(spec, params)
    fp_spec = FixedPointSpec.from_config(cfg["fixed_point"])
    outputs = {
        rid: forward(params, fp_spec, request_input(cfg["sample_seed"], rid, cfg["input_dim"]))
        for rid in request_ids
    }
    return output_digest(request_ids, outputs, param_version, p_digest), p_digest


def compute_receipt_lines(cfg, worker_id, param_version, batch_id, request_ids, delay_ms=0):
    """生成一个批次的 r3 回执行（名次为批内按请求标识排序的稳定名次）。"""
    start = monotonic_ms()
    digest, _ = compute_batch(cfg, worker_id, param_version, batch_id, request_ids)
    if delay_ms:
        time.sleep(delay_ms / 1000.0)
    elapsed = max(0, monotonic_ms() - start)
    lines = []
    for rank, rid in enumerate(sorted(request_ids)):
        lines.append(f"r3|{batch_id}|{rid}|{param_version}|{elapsed}|{worker_id}|{rank}||{digest}")
    return lines


def load_config(path):
    with open(path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    FixedPointSpec.from_config(cfg["fixed_point"])
    workers = cfg["workers"]
    if not workers:
        raise ValueError("workers 不能为空")
    cfg["workers_by_id"] = {w["worker_id"]: w for w in workers}
    if len(cfg["workers_by_id"]) != len(workers):
        raise ValueError("worker_id 重复")
    return cfg


def load_ports(path):
    with open(path, "r", encoding="utf-8") as fh:
        ports = json.load(fh)
    lo, hi = ports["port_range"]
    if lo < 1024 or hi > 65535 or lo > hi:
        raise ValueError("端口范围必须落在 1024..65535")
    if ports.get("host") != "127.0.0.1":
        raise ValueError("仅允许 127.0.0.1 回环地址")
    for wid, port in ports["workers"].items():
        if not lo <= int(port) <= hi:
            raise ValueError(f"工作进程 {wid} 端口 {port} 越界")
    return ports


def build_plan(cfg, param_versions):
    """由配置构建确定性执行计划：请求 -> 批 -> 工作进程 -> 参数版本。"""
    requests = gen_requests(cfg["sample_seed"], cfg["request_count"])
    batches = chunk_batches(requests, cfg["batch_size"])
    batch_ids = [bid for bid, _ in batches]
    assignment = assign_workers(batch_ids, cfg["workers_by_id"].keys())
    plan = []
    for bid, rids in batches:
        wid = assignment[bid]
        plan.append(PlannedBatch(
            batch_id=bid,
            worker_id=wid,
            param_version=param_versions[wid],
            request_ids=tuple(sorted(rids)),
        ))
    return plan
