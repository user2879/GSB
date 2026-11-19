"""Deterministic query-request generation and batch solving.

A query set is one fixed camera query plus a seeded set of dynamic-object
queries.  Batch results are merged by query key (never by completion order),
so shuffling the request order cannot change per-query results or the
result-set hash.
"""
import numpy as np

from .hashing import floats_hex, sha256_json
from .strategies import ProbeField, evaluate

CAMERA_POSITION = (0.0, 0.0, 12.0)
CAMERA_NORMAL = (0.0, 0.0, 1.0)
BATCH_TOLERANCE = 1e-6  # declared batch-vs-single relative tolerance


def generate_queries(seed: int, n_objects: int = 24) -> list:
    rng = np.random.default_rng(seed + 7919)  # fixed stream offset
    queries = [{
        "key": "cam:000",
        "position": list(CAMERA_POSITION),
        "normal": list(CAMERA_NORMAL),
    }]
    positions = rng.uniform(-3.0, 3.0, size=(n_objects, 3))
    normals = rng.standard_normal((n_objects, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    for i in range(n_objects):
        queries.append({
            "key": f"obj:{i:03d}",
            "position": positions[i].tolist(),
            "normal": normals[i].tolist(),
        })
    return queries


def query_set_hash(queries: list) -> str:
    canon = [{"key": q["key"],
              "position": floats_hex(q["position"]),
              "normal": floats_hex(q["normal"])} for q in
             sorted(queries, key=lambda q: q["key"])]
    return sha256_json(canon)


def _solve(field: ProbeField, strategy: str, queries: list) -> dict:
    points = np.array([q["position"] for q in queries], dtype=np.float64)
    normals = np.array([q["normal"] for q in queries], dtype=np.float64)
    values = evaluate(field, strategy, points, normals)
    return {q["key"]: values[i] for i, q in enumerate(queries)}


def solve_batch(field: ProbeField, strategy: str, queries: list) -> dict:
    """Vectorised batch solve; returns {key: (3,) irradiance}."""
    return _solve(field, strategy, list(queries))


def solve_single(field: ProbeField, strategy: str, queries: list) -> dict:
    """Per-query solve of the same requests (reference for consistency)."""
    out = {}
    for q in queries:
        out.update(_solve(field, strategy, [q]))
    return out


def merge_by_key(results: dict) -> list:
    """Deterministic merge: sorted by query key, never completion order."""
    return [(k, results[k]) for k in sorted(results.keys())]


def result_set_hash(results: dict) -> str:
    """Order-independent hash of the full result set."""
    canon = {k: floats_hex(v) for k, v in merge_by_key(results)}
    return sha256_json(canon)


def check_batch_consistency(batch: dict, single: dict,
                            tol: float = BATCH_TOLERANCE) -> dict:
    worst = 0.0
    for key, value in batch.items():
        ref = single[key]
        denom = np.maximum(np.abs(ref), 1e-12)
        worst = max(worst, float(np.max(np.abs(value - ref) / denom)))
    return {"tolerance": tol, "max_relative_deviation": worst,
            "consistent": bool(worst <= tol)}
