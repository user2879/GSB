"""Acceptance suite for the qdemo quantization demo environment.

Run from the workspace root:  python3.10 tests/run_acceptance.py
Exits 0 when every check passes.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
ENV = dict(os.environ, PYTHONPATH=ROOT)

CHECKS = []


def check(name: str, ok: bool, detail: str = ""):
    CHECKS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" :: {detail}" if detail else ""))


def run(args, env=None, cwd=ROOT):
    return subprocess.run(
        [PY, "-m", "qdemo"] + args, capture_output=True, text=True, cwd=ROOT, env=env or ENV
    )


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main() -> int:
    work = tempfile.mkdtemp(prefix="qdemo_acc_")
    data = os.path.join(ROOT, "data")
    os.makedirs(data, exist_ok=True)

    # -- setup: deterministic local samples --------------------------------
    r = run(["gen-weights", "--out", "data/weights.npz", "--seed", "12345"])
    check("gen-weights", r.returncode == 0 and '"OK"' in r.stdout)
    r = run(["gen-calib", "--weights", "data/weights.npz", "--out", "data/calib.npz",
             "--seed", "777", "--samples", "32"])
    check("gen-calib", r.returncode == 0 and '"OK"' in r.stdout)

    base_cfg = load(os.path.join(ROOT, "config.sample.json"))

    def write_cfg(name, mutate=None):
        cfg = json.loads(json.dumps(base_cfg))
        if mutate:
            mutate(cfg)
        path = os.path.join(work, name)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh)
        return path

    # -- 1. full run + replay ------------------------------------------------
    cfg_base = write_cfg("base.json")
    r = run(["full", "--config", cfg_base, "--report", f"{work}/r_full.json",
             "--record", f"{work}/rec_full.json"])
    check("full-run", r.returncode == 0, r.stdout.strip()[:200])
    r = run(["replay", "--record", f"{work}/rec_full.json"])
    check("replay-identical", r.returncode == 0 and '"identical"' in r.stdout, r.stdout.strip())

    # -- 2. concurrency consistency (1 vs 4 workers) --------------------------
    cfg_c1 = write_cfg("c1.json", lambda c: c["interaction"].update(concurrency=1))
    run(["full", "--config", cfg_c1, "--report", f"{work}/r_c1.json", "--record", f"{work}/rec_c1.json"])
    rep4, rep1 = load(f"{work}/r_full.json"), load(f"{work}/r_c1.json")
    same = rep4["layers"] == rep1["layers"] and rep4["failures"] == rep1["failures"]
    check("concurrency-order-consistency", same)

    # -- 3. incremental interactive == full with final config -----------------
    final_cfg = write_cfg("final.json", lambda c: (
        c["layers"].setdefault("overrides", {}).update({
            "fc1": {"bits": 8, "group_size": 16},
            "fc2": {"scheme": "symmetric", "per_channel": False},
        }),
        c["distill"].update(temperature=3.0, soft_label_weight=0.7),
    ))
    script = os.path.join(work, "cmds.txt")
    with open(script, "w", encoding="utf-8") as fh:
        fh.write("\n".join([
            "set-bits fc1 8",
            "set-scheme fc2 symmetric",
            "set-group fc1 16",
            "set-per-channel fc2 off",
            "set-distill 3.0 0.7",
            "quit",
        ]))
    r = run(["interactive", "--config", cfg_base, "--workdir", f"{work}/isess", "--script", script])
    check("interactive-session", r.returncode == 0, r.stdout.strip()[:120])
    run(["full", "--config", final_cfg, "--report", f"{work}/r_final.json",
         "--record", f"{work}/rec_final.json"])
    inc = load(f"{work}/isess/report.json")
    full = load(f"{work}/r_final.json")
    fields = ["layers", "failures", "distill", "summary"]
    same = all(inc[f] == full[f] for f in fields)
    check("incremental-equals-full", same,
          "" if same else str([f for f in fields if inc[f] != full[f]]))

    # invalidation audit: last modifying command touched distill params only,
    # so every quant stage must be reused and every distill stage invalidated.
    log = [json.loads(l) for l in open(f"{work}/isess/interactive_log.jsonl", encoding="utf-8")]
    last_modify = [e for e in log if e["command"].startswith("set-distill")][-1]
    reused_quant = all(f"quant:{n}" in last_modify["reused"] for n in ("fc1", "fc2", "head"))
    invalidated_distill = all(f"distill:{n}" in last_modify["invalidated"] for n in ("fc1", "fc2", "head"))
    check("invalidation-audit", reused_quant and invalidated_distill)

    # -- 4. latency budget business code --------------------------------------
    cfg_slow = write_cfg("slow.json", lambda c: c["interaction"].update(
        budget_ms=5, compute_delay_ms_per_layer=60))
    script2 = os.path.join(work, "cmds2.txt")
    with open(script2, "w", encoding="utf-8") as fh:
        fh.write("set-bits fc1 8\nset-bits fc1 4\nquit\n")
    r = run(["interactive", "--config", cfg_slow, "--workdir", f"{work}/isess2", "--script", script2])
    log2 = [json.loads(l) for l in open(f"{work}/isess2/interactive_log.jsonl", encoding="utf-8")]
    pendings = [e for e in log2 if e.get("business_code") == "E_INTERACT_BUDGET_PENDING"]
    check("budget-pending-code", r.returncode == 0 and len(pendings) >= 1,
          f"pending={len(pendings)}")
    # pending responses must carry the most recent completed summary
    check("pending-carries-summary", all(e.get("summary") for e in pendings))
    # record must still be committed after background completion
    check("record-committed-after-pending", os.path.exists(f"{work}/isess2/experiment_record.json"))

    # -- 5. exception isolation ------------------------------------------------
    env_fail = dict(ENV, QDEMO_FAIL_LAYER="fc2", QDEMO_FAIL_STAGE="quant")
    r = run(["full", "--config", cfg_base, "--report", f"{work}/r_fail.json",
             "--record", f"{work}/rec_fail.json"], env=env_fail)
    rep = load(f"{work}/r_fail.json")
    kinds = {(f["layer"], f["stage"], f["code"]) for f in rep["failures"]}
    ok = (r.returncode == 0
          and ("fc2", "quant", "E_LAYER_FAILED") in kinds
          and ("fc2", "eval", "E_DEPENDENCY_FAILED") in kinds
          and ("fc2", "distill", "E_DEPENDENCY_FAILED") in kinds
          and rep["summary"]["completed_layers"] == 2)
    check("exception-isolation", ok, json.dumps(rep["failures"]))

    # -- 6. invalid config / weight mismatch -----------------------------------
    cfg_bad = write_cfg("bad.json", lambda c: c["layers"]["default"].update(bits=9))
    r = run(["validate", "--config", cfg_bad])
    check("config-invalid-code", r.returncode == 2 and "E_CONFIG_INVALID" in r.stdout)

    meta_path = os.path.join(data, "weights.npz.meta.json")
    meta = load(meta_path)
    meta["layers"][0]["shape"] = [999, 999]
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    r = run(["validate", "--config", cfg_base])
    check("weight-shape-mismatch-code", r.returncode == 3 and "E_WEIGHT_SHAPE_MISMATCH" in r.stdout)
    # restore
    r = run(["gen-weights", "--out", "data/weights.npz", "--seed", "12345"])
    r2 = run(["validate", "--config", cfg_base])
    check("weights-restored", r.returncode == 0 and r2.returncode == 0)

    # -- 7. distill instability marking ----------------------------------------
    cfg_unstable = write_cfg("unstable.json", lambda c: c["distill"].update(
        temperature=50.0, soft_label_weight=1.0))
    r = run(["full", "--config", cfg_unstable, "--report", f"{work}/r_unst.json",
             "--record", f"{work}/rec_unst.json"])
    rep = load(f"{work}/r_unst.json")
    unstable = rep["summary"]["distill_unstable_layers"]
    check("distill-instability-marked", r.returncode == 0 and len(unstable) >= 1,
          f"unstable={unstable}")

    shutil.rmtree(work, ignore_errors=True)
    failed = [n for n, ok, _ in CHECKS if not ok]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
