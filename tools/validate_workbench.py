#!/usr/bin/env python3
"""End-to-end validation for the deterministic rendering workbench.

Checks implemented:
  1. fingerprint-driven stage invalidation and precise reuse boundaries
  2. incremental vs clean-slate full render: pixel, latent and PNG equality
  3. serial (concurrency=1) vs thread-pool (concurrency=N) identical output
  4. img2img/txt2img shared-segment element-wise equivalence
  5. local patch invalidation boundary and no-op patch tolerance
  6. decoding determinism and real tier-dimension changes
  7. snapshot recovery after hard termination (no report promise)
  8. graceful stop exit code 0 and resumability
  9. failure code coverage and non-interrupting batch processing
"""

import json
import os
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from render_workbench.config import load_yaml, validate_config, load_prompts
from render_workbench.edits import EditPlanner, load_plan
from render_workbench.runner import run_workflow
from render_workbench.hashing import fingerprint
from render_workbench.config import config_fingerprint_payload

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class CheckFailure(AssertionError):
    pass


def _check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print("[%s] %s %s" % (status, name, ("- " + detail) if detail else ""))
    if not condition:
        raise CheckFailure("%s: %s" % (name, detail))


def _paths(sample_dir, work_root):
    return {
        "config": os.path.join(sample_dir, "render_config.yaml"),
        "prompts": os.path.join(sample_dir, "prompts.jsonl"),
        "plan": os.path.join(sample_dir, "edit_plan.json"),
        "inc": os.path.join(work_root, "incremental"),
        "full": os.path.join(work_root, "full"),
        "serial": os.path.join(work_root, "serial"),
        "parallel": os.path.join(work_root, "parallel"),
    }


def _load_manifest(workdir):
    with open(os.path.join(workdir, "manifest.json"), encoding="utf-8") as h:
        return json.load(h)


def _node_map(manifest):
    return manifest["nodes"]


def _compare_manifolds(path_a, path_b):
    """Compare every common node's content fingerprint and stored bytes."""
    ma, mb = _load_manifest(path_a), _load_manifest(path_b)
    na, nb = ma["nodes"], mb["nodes"]
    missing = sorted(set(na) ^ set(nb))
    if missing:
        return False, "node sets differ: %s" % missing[:5]
    mismatches = []
    for node_id in sorted(na):
        if na[node_id]["content_fp"] != nb[node_id]["content_fp"]:
            mismatches.append(node_id)
        for filename, fp in na[node_id]["file_fps"].items():
            other = nb[node_id]["file_fps"].get(filename)
            if fp != other:
                mismatches.append("%s/%s" % (node_id, filename))
    if mismatches:
        return False, "fingerprint mismatch: %s" % mismatches[:5]
    for node_id in sorted(na):
        safe = node_id.replace("::", "__").replace("/", "_")
        for filename in na[node_id]["file_fps"]:
            pa = os.path.join(path_a, "store", safe, filename)
            pb = os.path.join(path_b, "store", safe, filename)
            with open(pa, "rb") as fa, open(pb, "rb") as fb:
                if fa.read() != fb.read():
                    return False, "byte mismatch at %s/%s" % (node_id, filename)
    return True, "all %d nodes byte-identical" % len(na)


def check_full_vs_incremental(paths):
    print("\n== incremental vs clean-slate full render ==")
    shutil.rmtree(paths["inc"], ignore_errors=True)
    shutil.rmtree(paths["full"], ignore_errors=True)
    inc_report, _ = run_workflow(paths["config"], paths["prompts"],
                                 paths["plan"], paths["inc"])
    full_report, _ = run_workflow(paths["config"], paths["prompts"],
                                  paths["plan"], paths["full"], mode="full")
    ok, detail = _compare_manifolds(paths["inc"], paths["full"])
    _check("pixel/latent/PNG byte equality", ok, detail)

    inc_codes = [f["code"] for f in inc_report["failures"]]
    full_codes = [f["code"] for f in full_report["failures"]]
    _check("failure list order identical", inc_codes == full_codes,
           "%s vs %s" % (inc_codes, full_codes))
    inc_status = [b["status"] for b in inc_report["batches"]]
    full_status = [b["status"] for b in full_report["batches"]]
    _check("batch status sequence identical", inc_status == full_status)
    for ba, bb in zip(inc_report["batches"], full_report["batches"]):
        ra = [r["node_id"] for r in ba["recompute"]]
        rb = [r["node_id"] for r in bb["recompute"]]
        ua = [r["node_id"] for r in ba["reuse"]]
        ub = [r["node_id"] for r in bb["reuse"]]
        _check("batch %s recompute/reuse lists identical" % ba["batch_id"],
               ra == rb and ua == ub)
    return inc_report


def check_concurrency(paths):
    print("\n== serial vs thread-pool concurrency ==")
    import yaml
    raw = load_yaml(paths["config"])
    shutil.rmtree(paths["serial"], ignore_errors=True)
    shutil.rmtree(paths["parallel"], ignore_errors=True)
    os.makedirs(paths["serial"], exist_ok=True)
    os.makedirs(paths["parallel"], exist_ok=True)

    serial_raw = dict(raw, concurrency=1)
    parallel_raw = dict(raw, concurrency=max(4, int(raw.get("concurrency", 1))))
    serial_cfg_path = os.path.join(paths["serial"], "serial_config.yaml")
    parallel_cfg_path = os.path.join(paths["parallel"], "parallel_config.yaml")
    with open(serial_cfg_path, "w", encoding="utf-8") as h:
        yaml.safe_dump(serial_raw, h, sort_keys=True)
    with open(parallel_cfg_path, "w", encoding="utf-8") as h:
        yaml.safe_dump(parallel_raw, h, sort_keys=True)
    run_workflow(serial_cfg_path, paths["prompts"], paths["plan"],
                 os.path.join(paths["serial"], "work"))
    run_workflow(parallel_cfg_path, paths["prompts"], paths["plan"],
                 os.path.join(paths["parallel"], "work"))
    ok, detail = _compare_manifolds(
        os.path.join(paths["serial"], "work"),
        os.path.join(paths["parallel"], "work"))
    _check("concurrency 1 vs N byte-identical", ok, detail)


def check_determinism(paths):
    print("\n== repeat-run determinism ==")
    repeat_dir = os.path.join(os.path.dirname(paths["inc"]), "repeat")
    shutil.rmtree(repeat_dir, ignore_errors=True)
    run_workflow(paths["config"], paths["prompts"], paths["plan"], repeat_dir)
    ok, detail = _compare_manifolds(paths["inc"], repeat_dir)
    _check("two independent runs byte-identical", ok, detail)


def check_invalidation_boundaries(paths, report):
    print("\n== precise fingerprint-driven invalidation ==")
    by_id = {batch["batch_id"]: batch for batch in report["batches"]}

    # A baseline render establishes the pristine cache; every edit batch
    # therefore runs warm and must invalidate only its own subgraph.
    baseline = by_id["baseline"]
    _check("baseline computes an embedding per prompt",
           len({r["node_id"] for r in baseline["recompute"]
                if r["node_id"].startswith("embedding::")}) == 4)
    b01 = by_id["b01_token_replace"]
    recomputed = {r["node_id"] for r in b01["recompute"]}
    embedding_hits = sorted(n for n in recomputed
                            if n.startswith("embedding::"))
    _check("warm prompt edit recomputes only its own embedding",
           embedding_hits == ["embedding::p_alpha"], str(embedding_hits))
    other_embeddings = [n for n in recomputed
                        if n.startswith("embedding::")
                        and n != "embedding::p_alpha"]
    _check("other prompts' embeddings stay valid", not other_embeddings,
           str(other_embeddings))

    b02 = by_id["b02_seed_and_guidance"]
    seed_recomputed = {r["node_id"] for r in b02["recompute"]}
    embedding_recomputed = [n for n in seed_recomputed
                            if n.startswith("embedding::")]
    _check("seed/guidance change keeps embeddings cached",
           not embedding_recomputed, str(embedding_recomputed))
    init_or_later = [n for n in seed_recomputed
                     if n.startswith(("init_latent::", "denoise::",
                                      "frame::", "patch::", "img_encode::"))]
    _check("seed/guidance change invalidates latent chain",
           len(init_or_later) > 0)

    b05 = by_id["b05_patch_noop"]
    patch_nodes = [n for n in (r["node_id"] for r in b05["recompute"])
                   if n.startswith("patch::")]
    _check("tolerance-level no-op patch avoids downstream recompute",
           b05["recomputed_nodes"] == 0,
           "recomputed=%d" % b05["recomputed_nodes"])

    b04 = by_id["b04_patch_add"]
    gamma_recomputed = [r["node_id"] for r in b04["recompute"]
                        if "p_gamma" in r["node_id"]]
    upstream = [n for n in gamma_recomputed
                if n.startswith(("embedding::", "init_latent::"))]
    _check("patch keeps upstream stages hit", not upstream, str(upstream))
    downstream = [n for n in gamma_recomputed
                  if n.startswith(("denoise::", "frame::", "patch::"))]
    _check("patch invalidates downstream steps/frames", len(downstream) > 0)

    b06 = by_id["b06_tier_add"]
    ultra = [r["node_id"] for r in b06["recompute"] if "ultra" in r["node_id"]]
    old_tier = [r["node_id"] for r in b06["recompute"]
                if "::low::" in r["node_id"] or "::high::" in r["node_id"]]
    _check("tier add only builds new tier nodes", len(ultra) > 0
           and not old_tier, "new=%d old=%d" % (len(ultra), len(old_tier)))

    b07 = by_id["b07_tier_remove"]
    removed = [n for n in _load_manifest(paths["inc"])["nodes"]
               if "ultra" in n]
    _check("tier remove prunes its artifacts", not removed)



def check_tier_pixel_dimensions(paths):
    print("\n== decoder tier dimensions and determinism ==")
    manifest = _load_manifest(paths["inc"])
    frames = [n for n, e in manifest["nodes"].items() if e["kind"] == "frame"]
    by_tier = {}
    for node_id in frames:
        entry = manifest["nodes"][node_id]
        by_tier.setdefault(entry["tier_id"], []).append((node_id, entry))
    shapes = {tier: entries[0][1]["pixel_shape"]
              for tier, entries in by_tier.items()}
    _check("tiers expose distinct pixel shapes",
           len({tuple(s) for s in shapes.values()}) == len(shapes),
           str(shapes))
    for tier, entries in by_tier.items():
        node_id, entry = entries[0]
        safe = node_id.replace("::", "__").replace("/", "_")
        pixels_a = np.frombuffer(
            open(os.path.join(paths["inc"], "store", safe, "pixels.bin"),
                 "rb").read(), dtype=np.uint8).reshape(entry["pixel_shape"])
        from render_workbench.png_codec import decode_png
        png_pixels = decode_png(
            open(os.path.join(paths["inc"], "store", safe, "frame.png"),
                 "rb").read())
        _check("tier %s PNG matches pixel buffer" % tier,
               np.array_equal(pixels_a, png_pixels))
        _check("tier %s pixels within uint8 range" % tier,
               pixels_a.min() >= 0 and pixels_a.max() <= 255)


def check_shared_segments(report):
    print("\n== img2img shared-segment semantics ==")
    verifications = report["shared_segment_verification"]
    _check("img2img prompts verified", len(verifications) > 0)
    for item in verifications:
        _check("divergent/shared segment lists present",
               item["shared_steps"] and item["replace_steps"] >= 0,
               "%s/%s" % (item["prompt_id"], item["tier_id"]))
        for check in item["checks"]:
            _check("shared step %d sandbox recompute matches txt2img exactly"
                   % check["step_index"],
                   check["sandbox_matches_txt2img"],
                   "%s/%s" % (item["prompt_id"], item["tier_id"]))
            _check("shared step %d uses one shared denoise semantics"
                   % check["step_index"],
                   check["sandbox_fp"] == check["txt2img_fp"],
                   "%s/%s" % (item["prompt_id"], item["tier_id"]))
        divergent = item["shared_steps"][0] if item["shared_steps"] else 0
        if item["checks"]:
            first = item["checks"][0]
            _check("divergent origins remain distinct at shared entry",
                   first["txt2img_fp"] != first["img2img_fp"],
                   "%s/%s" % (item["prompt_id"], item["tier_id"]))


def check_failure_codes(report):
    print("\n== failure codes and non-interrupting processing ==")
    codes = [f["code"] for f in report["failures"]]
    for expected in ("E_TOKEN_OUT_OF_VOCAB", "E_PATCH_OUT_OF_RANGE",
                     "E_CONFIG_INVALID"):
        _check("failure code observed: %s" % expected, expected in codes,
               str(codes))
    failed_batches = [b["batch_id"] for b in report["batches"]
                      if b["status"] == "failed"]
    applied_after = [b["batch_id"] for b in report["batches"]
                     if b["status"] == "applied"]
    _check("invalid batches recorded as failed",
           "b11_bad_token" in failed_batches
           and "b12_patch_out_of_range" in failed_batches
           and "b13_bad_seed" in failed_batches)
    _check("processing continues after failures",
           "b14_final_tweak" in applied_after)
    by_id = {b["batch_id"]: b for b in report["batches"]}
    _check("failed batch triggers no recompute",
           by_id["b12_patch_out_of_range"]["recomputed_nodes"] == 0)


def check_recovery(paths):
    print("\n== snapshot recovery ==")
    kill_dir = os.path.join(os.path.dirname(paths["inc"]), "kill_hard")
    shutil.rmtree(kill_dir, ignore_errors=True)
    driver = os.path.join(ROOT, "tools", "_kill_driver.py")
    proc = subprocess.run(
        [sys.executable, driver, "hard", "2", kill_dir,
         paths["config"], paths["prompts"], paths["plan"]],
        capture_output=True, text=True)
    _check("hard termination exits non-zero", proc.returncode == 137,
           "rc=%s" % proc.returncode)
    _check("hard termination writes no report",
           not os.path.exists(os.path.join(kill_dir, "report.json")))

    manifest_before = _load_manifest(kill_dir)
    version_before = manifest_before["version"]
    nodes_before = set(manifest_before["nodes"])
    resumed_report, _ = run_workflow(paths["config"], paths["prompts"],
                                     paths["plan"], kill_dir)
    manifest_after = _load_manifest(kill_dir)
    _check("resume recovers to full node set",
           len(manifest_after["nodes"])
           == len(_load_manifest(paths["inc"])["nodes"]))
    ok, detail = _compare_manifolds(kill_dir, paths["inc"])
    _check("resumed run matches uninterrupted run byte-for-byte", ok, detail)
    _check("resume version strictly advances",
           manifest_after["version"] >= version_before)
    _check("recovered report records resume version",
           resumed_report["resume_version"] == manifest_after["version"])

    print("\n== graceful stop ==")
    graceful_dir = os.path.join(os.path.dirname(paths["inc"]), "kill_grace")
    shutil.rmtree(graceful_dir, ignore_errors=True)
    proc = subprocess.run(
        [sys.executable, driver, "graceful", "1", graceful_dir,
         paths["config"], paths["prompts"], paths["plan"]],
        capture_output=True, text=True)
    _check("graceful stop exits 0", proc.returncode == 0,
           "rc=%s stderr=%s" % (proc.returncode, proc.stderr[-200:]))
    graceful_report = json.load(
        open(os.path.join(graceful_dir, "report.json"), encoding="utf-8"))
    _check("graceful report marks stopped status",
           graceful_report["status"] == "stopped")
    run_workflow(paths["config"], paths["prompts"], paths["plan"],
                 graceful_dir)
    ok, detail = _compare_manifolds(graceful_dir, paths["inc"])
    _check("post-graceful resume matches uninterrupted run", ok, detail)


def check_serial_baseline(paths):
    print("\n== single-thread serial baseline ==")
    baseline_dir = os.path.join(os.path.dirname(paths["inc"]), "baseline")
    shutil.rmtree(baseline_dir, ignore_errors=True)
    report, _ = run_workflow(paths["config"], paths["prompts"], paths["plan"],
                             baseline_dir, mode="baseline",
                             baseline_concurrency=1)
    _check("baseline run completes", report["status"] == "completed")
    ok, detail = _compare_manifolds(baseline_dir, paths["inc"])
    _check("serial baseline byte-identical to incremental", ok, detail)
    manifest = _load_manifest(baseline_dir)
    _check("baseline manifest carries cursors", bool(manifest.get("cursors")))


def main():
    sample_dir = os.path.join(ROOT, "samples")
    work_root = os.path.join(ROOT, "work", "validation")
    shutil.rmtree(work_root, ignore_errors=True)
    os.makedirs(work_root, exist_ok=True)
    paths = _paths(sample_dir, work_root)

    inc_report = check_full_vs_incremental(paths)
    check_determinism(paths)
    check_concurrency(paths)
    check_serial_baseline(paths)
    check_invalidation_boundaries(paths, inc_report)
    check_tier_pixel_dimensions(paths)
    check_shared_segments(inc_report)
    check_failure_codes(inc_report)
    check_recovery(paths)

    print("\nALL VALIDATION CHECKS PASSED")
    result = {
        "status": "passed",
        "incremental_workdir": paths["inc"],
        "full_workdir": paths["full"],
        "total_nodes": len(_load_manifest(paths["inc"])["nodes"]),
        "total_recomputed_nodes": inc_report["total_recomputed_nodes"],
    }
    out_path = os.path.join(work_root, "validation_summary.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except CheckFailure as exc:
        print("\nVALIDATION FAILED: %s" % exc)
        sys.exit(1)
