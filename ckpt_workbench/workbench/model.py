"""Deterministic small feed-forward model, sharding and digest rules.

All randomness comes from a single ``numpy.random.RandomState`` created from an
integer seed. Repeated runs with the same declaration produce byte-identical
tensors, RNG states and digests.

Sharding partition modes (explicit, rule-based):
  * ``layer`` - one shard per dense layer, holding ``W_l`` and ``b_l`` plus
    their optimizer momenta.
  * ``unit``  - one shard per tensor, each holding a single parameter tensor
    and its momentum.

Tensor naming order is fixed: ``W1, b1, W2, b2, ...`` where layer ``l`` maps
dims[l-1] -> dims[l].
"""

import json
import struct

import numpy as np

from .hashing import canon_bytes, digest, digest_parts

BLOB_MAGIC = b"CKWBSH01"
PARTITION_MODES = ("layer", "unit")


def load_declaration(path):
    with open(path, "r", encoding="utf-8") as handle:
        decl = json.load(handle)
    validate_declaration(decl)
    return decl


def validate_declaration(decl):
    required = ("seed", "shard_count", "partition_mode", "layer_dims", "steps")
    missing = [key for key in required if key not in decl]
    if missing:
        raise ValueError(f"declaration missing keys: {missing}")
    if not isinstance(decl["seed"], int) or decl["seed"] < 0:
        raise ValueError("seed must be a non-negative integer")
    if decl["partition_mode"] not in PARTITION_MODES:
        raise ValueError(f"partition_mode must be one of {PARTITION_MODES}")
    dims = decl["layer_dims"]
    if not isinstance(dims, list) or len(dims) < 2 or any(d <= 0 for d in dims):
        raise ValueError("layer_dims must be a list of >=2 positive ints")
    if not isinstance(decl["steps"], int) or decl["steps"] < 0:
        raise ValueError("steps must be a non-negative int")
    if not isinstance(decl["shard_count"], int) or decl["shard_count"] <= 0:
        raise ValueError("shard_count must be a positive int")
    expected = len(plan_shards(decl["partition_mode"], dims))
    if decl["shard_count"] != expected:
        raise ValueError(
            f"shard_count={decl['shard_count']} does not match partition "
            f"mode {decl['partition_mode']!r} which yields {expected} shards"
        )


def tensor_names(layer_dims):
    names = []
    for layer in range(1, len(layer_dims)):
        names.append(f"W{layer}")
        names.append(f"b{layer}")
    return names


def layer_of_tensor(name):
    return int(name[1:])


def plan_shards(partition_mode, layer_dims):
    """Return ordered list of shard plans: [{"id", "tensors": [...]}]."""
    names = tensor_names(layer_dims)
    plans = []
    if partition_mode == "unit":
        for idx, name in enumerate(names):
            plans.append({"id": f"shard{idx:03d}", "tensors": [name]})
    elif partition_mode == "layer":
        idx = 0
        for layer in range(1, len(layer_dims)):
            plans.append({
                "id": f"shard{idx:03d}",
                "tensors": [f"W{layer}", f"b{layer}"],
            })
            idx += 1
    else:
        raise ValueError(f"unknown partition mode {partition_mode!r}")
    return plans


def new_rng(seed):
    return np.random.RandomState(int(seed))


def encode_rng_state(rng):
    tag, key, pos, has_gauss, cached = rng.get_state()
    payload = {
        "tag": str(tag),
        "key": np.asarray(key, dtype=np.uint32).tolist(),
        "pos": int(pos),
        "has_gauss": bool(has_gauss),
        "cached_gauss": float(cached),
    }
    return canon_bytes(payload)


def decode_rng_state(encoded):
    # encoded is canonical bytes; callers persist the decoded tuple instead.
    raise NotImplementedError("use rng_state_from_dict instead")


def rng_state_to_dict(rng):
    tag, key, pos, has_gauss, cached = rng.get_state()
    return {
        "tag": str(tag),
        "key": np.asarray(key, dtype=np.uint32).tolist(),
        "pos": int(pos),
        "has_gauss": bool(has_gauss),
        "cached_gauss": float(cached),
    }


def rng_from_state_dict(state_dict):
    rng = np.random.RandomState()
    rng.set_state((
        state_dict["tag"],
        np.asarray(state_dict["key"], dtype=np.uint32),
        int(state_dict["pos"]),
        bool(state_dict["has_gauss"]),
        float(state_dict["cached_gauss"]),
    ))
    return rng


def initial_state(decl):
    """Build initial parameters, zero momenta, RNG state and zero cursor."""
    rng = new_rng(decl["seed"])
    dims = decl["layer_dims"]
    tensors = {}
    for layer in range(1, len(dims)):
        tensors[f"W{layer}"] = rng.normal(size=(dims[layer], dims[layer - 1]))
        tensors[f"b{layer}"] = rng.normal(size=(dims[layer],))
    momenta = {name: np.zeros_like(value) for name, value in tensors.items()}
    return {
        "tensors": tensors,
        "momenta": momenta,
        "rng_state": rng_state_to_dict(rng),
        "cursor": 0,
    }


def train_step(state, step_index, decl):
    """Apply one deterministic SGD+momentum step in place on a state dict."""
    rng = rng_from_state_dict(state["rng_state"])
    lr = 0.01
    momentum = 0.9
    for name in tensor_names(decl["layer_dims"]):
        grad = rng.normal(scale=0.1, size=state["tensors"][name].shape)
        update = momentum * state["momenta"][name] - lr * grad
        state["momenta"][name] = update
        state["tensors"][name] = state["tensors"][name] + update
    batch_size = 4
    state["cursor"] = int(state["cursor"]) + batch_size
    state["rng_state"] = rng_state_to_dict(rng)
    return state


def advance_steps(state, count, decl):
    for index in range(count):
        train_step(state, index, decl)
    return state


def shard_digest(tensor_names_in_shard, state):
    """Stable digest of one shard's parameters and optimizer momenta."""
    mapping = {}
    for name in tensor_names_in_shard:
        mapping[name] = {
            "param": np.ascontiguousarray(state["tensors"][name]),
            "momentum": np.ascontiguousarray(state["momenta"][name]),
        }
    return digest(mapping)


def partition_digest(shard_digests):
    """Derived digest over the ordered shard digest set."""
    return digest_parts([f"part:{idx}:{value}" for idx, value in enumerate(shard_digests)])


def optimizer_digest(state, layer_dims):
    return digest_parts([
        digest(np.ascontiguousarray(state["momenta"][name]))
        for name in tensor_names(layer_dims)
    ])


def content_digest(entry):
    """Checkpoint content digest over every ledger field except itself."""
    material = {key: value for key, value in entry.items() if key != "content_digest"}
    return digest(material)


def build_shard_blob(shard_id, tensor_names_in_shard, state):
    """Deterministic byte container for one shard; arrays little-endian raw."""
    out = bytearray(BLOB_MAGIC)
    out += struct.pack(">I", len(tensor_names_in_shard))
    for name in tensor_names_in_shard:
        param = np.ascontiguousarray(state["tensors"][name], dtype=np.float64)
        momentum = np.ascontiguousarray(state["momenta"][name], dtype=np.float64)
        name_raw = name.encode("utf-8")
        out += struct.pack(">I", len(name_raw)) + name_raw
        shape = list(param.shape)
        shape_raw = json.dumps(shape, separators=(",", ":")).encode("ascii")
        out += struct.pack(">I", len(shape_raw)) + shape_raw
        out += struct.pack(">Q", param.nbytes) + param.tobytes(order="C")
        out += struct.pack(">Q", momentum.nbytes) + momentum.tobytes(order="C")
    return bytes(out)


def parse_shard_blob(blob):
    """Inverse of build_shard_blob, returning tensors and momenta dicts."""
    if not blob.startswith(BLOB_MAGIC):
        raise ValueError("bad shard blob magic")
    offset = len(BLOB_MAGIC)
    (count,) = struct.unpack_from(">I", blob, offset)
    offset += 4
    tensors, momenta = {}, {}
    for _ in range(count):
        (name_len,) = struct.unpack_from(">I", blob, offset)
        offset += 4
        name = blob[offset:offset + name_len].decode("utf-8")
        offset += name_len
        (shape_len,) = struct.unpack_from(">I", blob, offset)
        offset += 4
        shape = json.loads(blob[offset:offset + shape_len].decode("ascii"))
        offset += shape_len
        (nbytes,) = struct.unpack_from(">Q", blob, offset)
        offset += 8
        param = np.frombuffer(blob, dtype="<f8", count=nbytes // 8, offset=offset).reshape(shape).copy()
        offset += nbytes
        (mbytes,) = struct.unpack_from(">Q", blob, offset)
        offset += 8
        momentum = np.frombuffer(blob, dtype="<f8", count=mbytes // 8, offset=offset).reshape(shape).copy()
        offset += mbytes
        tensors[name] = param
        momenta[name] = momentum
    return tensors, momenta
