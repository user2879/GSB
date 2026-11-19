"""Command-line interface and the fixed three-density / three-order run.

``python -m probegi run-all --workspace OUT --seed 1234`` executes the fixed
sequence: probe generation (8/16/32 cubed), SH projection (orders 1..3),
strategy selection under the declared per-tier budgets, batch + single
solves with consistency checks, error diagnostics, budget-degradation
planning, an offscreen preview, and one atomic snapshot commit.  Every step
prints a JSON line with its content hash.

Declared run budgets (part of the reproducibility contract):
  sparse  8^3: memory 4 MiB, batch time 250 ms  (accuracy first)
  medium 16^3: memory 4 MiB, batch time 150 ms  (balanced)
  dense  32^3: memory 4 MiB, batch time 200 ms  (resource aware)
Planned query load for time estimates: 100000 queries/batch.
Accuracy budget (soft): 0.05 mean relative error.
"""
import argparse
import json
import os
import sys

import numpy as np

from . import diagnostics, probes, queries as qmod, sh, snapshot, strategies
from .budget import plan_budget
from .errors import ProbeGIError
from .hashing import canonical_json, sha256_bytes, sha256_json
from .preview import render_preview
from .strategies import ProbeField

GRID_SIZES = (8, 16, 32)
ORDERS = (1, 2, 3)
DEFAULT_SEED = 1234
DEFAULT_DIRECTIONS = 256
N_OBJECTS = 24
PLANNED_QUERIES = 100000
ACCURACY_BUDGET = 0.05
MEMORY_BUDGET = 4 * 1024 * 1024
TIER_TIME_BUDGET_MS = {8: 250.0, 16: 150.0, 32: 200.0}
# Tight budgets used to demonstrate the degradation path deterministically.
DEMO_DEGRADE = {"grid_n": 32, "order": 3, "memory_bytes": 1024 * 1024,
                "time_ms": 100.0}


def _emit(record: dict) -> None:
    print(canonical_json(record), flush=True)


def _ensure_probes(workspace: str, grid_n: int, seed: int,
                   n_directions: int) -> dict:
    """Generate (or hash-verify) the staged probe file for a grid size."""
    pdir = os.path.join(workspace, "probes")
    os.makedirs(pdir, exist_ok=True)
    path = os.path.join(pdir, f"probes_{grid_n}.bin")
    mpath = path + ".manifest.json"
    if os.path.exists(path) and os.path.exists(mpath):
        with open(mpath, encoding="utf-8") as fh:
            manifest = json.load(fh)
        with open(path, "rb") as fh:
            digest = sha256_bytes(fh.read())
        if (digest == manifest["sha256"] and manifest["seed"] == seed
                and manifest["n_directions"] == n_directions):
            return manifest
    return probes.generate_to_file(path, grid_n, seed, n_directions)


def run_all(workspace: str, seed: int = DEFAULT_SEED,
            n_directions: int = DEFAULT_DIRECTIONS) -> int:
    os.makedirs(workspace, exist_ok=True)
    store = snapshot.SnapshotStore(os.path.join(workspace, "snapshot"))
    queries = qmod.generate_queries(seed, N_OBJECTS)
    commit_files = {}
    steps = []

    for grid_n in GRID_SIZES:
        manifest = _ensure_probes(workspace, grid_n, seed, n_directions)
        _emit({"step": "generate_probes", "grid_n": grid_n,
               "sha256": manifest["sha256"],
               "probe_count": manifest["probe_count"],
               "energy": manifest["energy"]})
        data = probes.load_probe_file(
            os.path.join(workspace, "probes", f"probes_{grid_n}.bin"))
        with open(os.path.join(workspace, "probes", f"probes_{grid_n}.bin"),
                  "rb") as fh:
            commit_files[f"probes/probes_{grid_n}.bin"] = fh.read()
        commit_files[f"probes/probes_{grid_n}.manifest.json"] = manifest

        for order in ORDERS:
            coeffs = sh.project(data["radiance"], data["directions"], order)
            field = ProbeField(grid_n, data["bbox"], coeffs)
            coeff_hash = sha256_bytes(
                np.ascontiguousarray(coeffs, dtype="<f8").tobytes())
            _emit({"step": "project_sh", "grid_n": grid_n, "order": order,
                   "coeff_sha256": coeff_hash})

            choice = strategies.select_strategy(
                grid_n, order, PLANNED_QUERIES, field.field_scale,
                MEMORY_BUDGET, TIER_TIME_BUDGET_MS[grid_n], ACCURACY_BUDGET)
            commit_files[f"strategy/select_{grid_n}_o{order}.json"] = choice
            _emit({"step": "select_strategy", "grid_n": grid_n,
                   "order": order, "strategy": choice["strategy"],
                   "est_error_bound": choice["est_error_bound"],
                   "est_memory_bytes": choice["est_memory_bytes"],
                   "sha256": sha256_json(choice)})

            strategy = choice["strategy"]
            batch = qmod.solve_batch(field, strategy, queries)
            single = qmod.solve_single(field, strategy, queries)
            consistency = qmod.check_batch_consistency(batch, single)
            rhash = qmod.result_set_hash(batch)
            _emit({"step": "solve_batch", "grid_n": grid_n, "order": order,
                   "strategy": strategy, "result_sha256": rhash,
                   "batch_consistent": consistency["consistent"]})
            if not consistency["consistent"]:
                raise ProbeGIError("batch/single mismatch", **consistency)

            report = diagnostics.diagnose(field, strategy, queries,
                                          n_directions, manifest["sha256"])
            ordering = diagnostics.check_ordering_consistency(report)
            commit_files[f"diagnostics/report_{grid_n}_o{order}.json"] = report
            _emit({"step": "diagnose", "grid_n": grid_n, "order": order,
                   "strategy": strategy,
                   "mean_relative_error":
                       report["aggregate"]["mean_relative_error"],
                   "ordering_consistent": ordering["consistent"],
                   "sha256": report["sha256"]})
            steps.append({"grid_n": grid_n, "order": order,
                          "strategy": strategy,
                          "report_sha256": report["sha256"]})

    # Deterministic budget-degradation demonstration (densest tier).
    degrade = DEMO_DEGRADE
    data32 = probes.load_probe_file(
        os.path.join(workspace, "probes", "probes_32.bin"))
    coeffs32 = sh.project(data32["radiance"], data32["directions"],
                          degrade["order"])
    field32 = ProbeField(32, data32["bbox"], coeffs32)
    plan = plan_budget(degrade["grid_n"], degrade["order"], n_directions,
                       PLANNED_QUERIES, field32.field_scale,
                       degrade["memory_bytes"], degrade["time_ms"])
    commit_files["budget/degradation_plan.json"] = plan
    _emit({"step": "budget_plan", "grid_n": degrade["grid_n"],
           "order": degrade["order"],
           "degradation_steps": [s["action"] for s in
                                 plan["degradation_steps"]],
           "final": plan["final"], "sha256": sha256_json(plan)})

    # Offscreen preview of one reconstructed frame (medium tier, order 2).
    data16 = probes.load_probe_file(
        os.path.join(workspace, "probes", "probes_16.bin"))
    field16 = ProbeField(16, data16["bbox"],
                         sh.project(data16["radiance"], data16["directions"],
                                    2))
    res = 96
    axis = np.linspace(-4.0, 4.0, res)
    px, py = np.meshgrid(axis, axis, indexing="xy")
    pts = np.stack([px.ravel(), py.ravel(), np.zeros(res * res)], axis=1)
    nrm = np.broadcast_to(np.array([0.0, 0.0, 1.0]), (res * res, 3))
    frame = strategies.evaluate(field16, "full", pts, nrm).reshape(res, res, 3)
    frame = np.clip(frame, 0.0, None)
    # Display exposure: normalise by mean irradiance and amplify the
    # deviation (fixed gain 4) so spatial variation from the seeded
    # emitters is visible in the preview.
    mean = max(float(frame.mean()), 1e-9)
    frame = 1.0 + (frame / mean - 1.0) * 4.0
    frame = np.clip(frame, 0.0, None)
    preview_path = os.path.join(workspace, "preview.png")
    preview = render_preview(frame, preview_path)
    with open(preview_path, "rb") as fh:
        commit_files["preview/preview.png"] = fh.read()
    commit_files["preview/preview.json"] = {
        "renderer": preview["renderer"], "status": preview["status"],
        "sha256": preview["sha256"]}
    _emit({"step": "preview", "renderer": preview["renderer"],
           "status": preview["status"], "sha256": preview["sha256"]})

    summary = {
        "seed": seed, "n_directions": n_directions,
        "query_set_sha256": qmod.query_set_hash(queries),
        "budgets": {"memory_bytes": MEMORY_BUDGET,
                    "tier_time_ms": TIER_TIME_BUDGET_MS,
                    "planned_queries": PLANNED_QUERIES,
                    "accuracy": ACCURACY_BUDGET},
        "steps": steps,
    }
    commit_files["run_summary.json"] = summary

    cur = store.current()
    new_hash = snapshot.content_hash_for(commit_files)
    if cur and cur["content_sha256"] == new_hash:
        _emit({"step": "snapshot", "changed": False,
               "version": cur["version"], "content_sha256": new_hash})
    else:
        committed = store.commit(commit_files,
                                 base_version=cur["version"] if cur else None)
        _emit({"step": "snapshot", "changed": True,
               "version": committed["version"],
               "content_sha256": committed["content_sha256"]})
    _emit({"step": "done", "status": "ok"})
    return 0


def verify(workspace: str, seed: int = DEFAULT_SEED) -> int:
    """Reload the committed snapshot and independently recompute every
    diagnostics report from the committed probe bytes."""
    store = snapshot.SnapshotStore(os.path.join(workspace, "snapshot"))
    state = store.load()
    queries = qmod.generate_queries(seed, N_OBJECTS)
    results = []
    for rel in sorted(state["files"]):
        if not rel.startswith("diagnostics/report_"):
            continue
        report = json.loads(state["files"][rel].decode("utf-8"))
        grid_n = report["grid_n"]
        probe_bytes = state["files"][f"probes/probes_{grid_n}.bin"]
        data = probes.decode(probe_bytes)
        check = diagnostics.verify_report(data, queries, report)
        check["report"] = rel
        results.append(check)
        _emit({"step": "verify", "report": rel,
               "verified": check["verified"]})
    ok = all(r["verified"] for r in results)
    _emit({"step": "verify_done", "verified": ok,
           "snapshot_version": state["version"],
           "content_sha256": state["content_sha256"]})
    return 0 if ok else 3


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="probegi")
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run-all", help="run the fixed full sequence")
    p_run.add_argument("--workspace", required=True)
    p_run.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p_run.add_argument("--directions", type=int, default=DEFAULT_DIRECTIONS)
    p_ver = sub.add_parser("verify",
                           help="recompute diagnostics from the snapshot")
    p_ver.add_argument("--workspace", required=True)
    p_ver.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    try:
        if args.command == "run-all":
            return run_all(args.workspace, args.seed, args.directions)
        if args.command == "verify":
            return verify(args.workspace, args.seed)
    except ProbeGIError as exc:
        print(canonical_json({"error": exc.to_dict()}), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
