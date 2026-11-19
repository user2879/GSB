"""Pipeline orchestration: synth -> encode -> decode -> inject -> metrics ->
audio alignment -> detection -> ledger commit. Every encode unit is isolated:
a failure is recorded with its retry count and never aborts the batch, and a
failed unit is never silently counted as passed."""
from __future__ import annotations

from pathlib import Path

from . import audio as audio_mod
from . import colors, detect, inject, ledger as ledger_mod, metrics as metrics_mod
from . import pipeline, report as report_mod, synth, util

METRIC_CONVENTIONS = {
    "pixel_range": "full [0,255] 8-bit",
    "matrix": "BT.601",
    "chroma_downsample": "box2x2",
    "psnr_cap_db": metrics_mod.PSNR_CAP,
    "ssim": "gaussian window, K1=0.01, K2=0.03, edge padding",
}


def process_encode(cfg, params, planes_cache, ref_pcm, truth, out_dir, injections):
    enc_path = Path(out_dir) / "encodes" / f"{params['id']}.mkv"
    if params["pix_fmt"] not in planes_cache:
        raise ValueError(f"unsupported pix_fmt: {params['pix_fmt']}")
    pipeline.encode_run(planes_cache[params["pix_fmt"]], ref_pcm, cfg, params, enc_path)
    stream, planes_seq, pcm, dec_errors = pipeline.decode_run(enc_path, cfg)
    stream["params"] = dict(params)
    original_srcs = [f["src"] for f in stream["frames"]]
    my_injections = [inj for inj in injections if inj.get("target") == params["id"]]
    pcm, applied = inject.apply_injections(stream, pcm, my_injections)
    surviving = {f["src"] for f in stream["frames"]}
    planes_seq = [p for p, s in zip(planes_seq, original_srcs) if s in surviving]
    pix = stream["actual_pix_fmt"]
    mcfg = cfg["metrics"]
    per_frame = []
    for f, dec_planes in zip(stream["frames"], planes_seq):
        ref_planes = planes_cache[pix][f["src"]]
        m = metrics_mod.frame_metrics(ref_planes, dec_planes, mcfg["weights"], mcfg["ssim"])
        m["src"] = f["src"]
        per_frame.append(m)
    segments = metrics_mod.segment_aggregate(
        per_frame, [f["src"] for f in stream["frames"]], truth["shots"]
    )
    audio_eval = audio_mod.evaluate_audio(
        ref_pcm, pcm, cfg, truth["shots"], cfg["tolerances"]["audio_max_lag_samples"]
    )
    findings = detect.detect_stream(stream, cfg, audio_eval)
    for err in dec_errors:
        findings.append({"code": "E_DECODE_PARTIAL", "location": err, "detail": err["error"]})
    return {
        "id": params["id"],
        "status": "ok",
        "params": stream["params"],
        "params_encoded": dict(params),
        "actual_pix_fmt": pix,
        "frame_count": len(stream["frames"]),
        "frame_order_hash": pipeline.frame_order_hash(stream["frames"]),
        "frames": stream["frames"],
        "audio": stream["audio"],
        "injections": applied,
        "metrics": {
            "conventions": METRIC_CONVENTIONS | {"weights": mcfg["weights"], "ssim_config": mcfg["ssim"]},
            "frames": per_frame,
            "segments": segments,
        },
        "audio_eval": audio_eval,
        "decode_errors": dec_errors,
        "findings": findings,
        "hashes": {
            "frames": util.hash_obj(stream["frames"]),
            "metrics": util.hash_obj(per_frame),
            "segments": util.hash_obj(segments),
            "audio_eval": util.hash_obj(audio_eval),
        },
    }


def run_pipeline(cfg, out_dir):
    out_dir = Path(out_dir)
    util.clean_stale_tmp(out_dir)
    frames_rgb, ref_pcm, truth = synth.generate_source(cfg, out_dir / "source")
    # Shot boundaries must equal the ground-truth list exactly; never infer
    # equality from matching counts.
    if list(cfg["shots"]) != [s["start"] for s in truth["shots"]]:
        raise ValueError("config shots do not match ground-truth shot boundaries")
    # Convert source RGB to YUV planes once per required pixel format and
    # reuse for encoding, the inspection copy and metric references. Encodes
    # with an unsupported pix_fmt are left to fail inside their isolated unit.
    needed = {"yuv420p"} | {
        p["pix_fmt"] for p in cfg["encodes"] if p["pix_fmt"] in colors.SUPPORTED_PIX_FMTS
    }
    planes_cache = {
        pix: [colors.rgb_to_planes(rgb, pix) for rgb in frames_rgb] for pix in sorted(needed)
    }
    del frames_rgb
    pipeline.mux_source_reference(
        planes_cache["yuv420p"], ref_pcm, cfg, out_dir / "source" / "source.mkv"
    )
    injections = cfg.get("injections", [])
    retries = int(cfg.get("retries", 1))
    encodes = []
    isolation = []
    for params in cfg["encodes"]:
        attempt = 0
        while True:
            try:
                rec = process_encode(cfg, params, planes_cache, ref_pcm, truth, out_dir, injections)
                break
            except Exception as exc:
                attempt += 1
                if attempt > retries:
                    isolation.append(
                        {"encode": params["id"], "stage": "encode_pipeline",
                         "retries": attempt - 1, "error": str(exc)}
                    )
                    rec = {
                        "id": params["id"],
                        "status": "failed",
                        "params": dict(params),
                        "retries": attempt - 1,
                        "error": str(exc),
                        "findings": [
                            {"code": "E_ENCODE_FAILED",
                             "location": {"encode": params["id"]},
                             "detail": str(exc)}
                        ],
                    }
                    break
        encodes.append(rec)
    ok = [e for e in encodes if e["status"] == "ok"]
    failed = [e for e in encodes if e["status"] != "ok"]
    summary = {
        "encodes_total": len(encodes),
        "encodes_ok": len(ok),
        "encodes_failed": len(failed),
        "findings_total": sum(len(e["findings"]) for e in encodes),
        "execution_complete": not failed and not any(e.get("decode_errors") for e in ok),
    }
    body = {
        "config": cfg,
        "source": truth,
        "encodes": encodes,
        "isolation": isolation,
        "summary": summary,
    }
    ledger = ledger_mod.finalize(body)
    commit_report = ledger_mod.commit(ledger, out_dir / "ledger")
    view = report_mod.render_report(out_dir / "ledger", out_dir / "report.html")
    return {"ledger": ledger, "commit": commit_report, "view": view}
