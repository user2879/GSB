"""Anomaly detectors. Each finding carries a business error code and a
location (frame number or segment/shot index) so issues can be pinpointed."""
from __future__ import annotations


def expected_pts_ms(src: int, fps: int) -> int:
    return int(round(src * 1000.0 / fps))


def detect_stream(stream: dict, cfg, audio_eval: dict | None) -> list:
    findings = []
    fps = cfg["video"]["fps"]
    n_expected = int(round(cfg["video"]["duration"] * fps))
    frames = stream["frames"]
    tol = cfg["tolerances"]

    srcs = [f["src"] for f in frames]
    missing = sorted(set(range(n_expected)) - set(srcs))
    if missing:
        findings.append(
            {
                "code": "E_FRAME_DROP",
                "location": {"frames": missing},
                "detail": f"{len(missing)} source frame(s) missing from decoded sequence",
            }
        )
    seen = set()
    dups = sorted({s for s in srcs if s in seen or seen.add(s)})
    if dups:
        findings.append(
            {
                "code": "E_FRAME_DUP",
                "location": {"frames": dups},
                "detail": "duplicate source frame(s) in decoded sequence",
            }
        )
    if len(frames) != n_expected and not missing:
        findings.append(
            {
                "code": "E_FRAME_COUNT",
                "location": {"expected": n_expected, "actual": len(frames)},
                "detail": "decoded frame count differs from source frame count",
            }
        )

    ts_bad = [
        {"src": f["src"], "pts_ms": f["pts_ms"], "expected_ms": expected_pts_ms(f["src"], fps)}
        for f in frames
        if f["pts_ms"] != expected_pts_ms(f["src"], fps)
    ]
    if ts_bad:
        findings.append(
            {
                "code": "E_TIMESTAMP_JUMP",
                "location": {"frame": ts_bad[0]["src"], "frames": [b["src"] for b in ts_bad]},
                "detail": (
                    f"pts {ts_bad[0]['pts_ms']}ms != expected {ts_bad[0]['expected_ms']}ms "
                    f"at frame {ts_bad[0]['src']}"
                ),
            }
        )

    declared = stream["params"].get("pix_fmt")
    actual = stream.get("actual_pix_fmt")
    if actual is not None and declared != actual:
        findings.append(
            {
                "code": "E_CHROMA_MISMATCH",
                "location": {"encode": stream["params"]["id"]},
                "detail": f"declared pix_fmt {declared} but decoded frames are {actual}",
            }
        )

    audio = stream["audio"]
    if audio["rate_declared"] != audio["rate_actual"]:
        declared_s = audio["samples"] / audio["rate_declared"]
        actual_s = audio["samples"] / audio["rate_actual"]
        findings.append(
            {
                "code": "E_AUDIO_RATE",
                "location": {"encode": stream["params"]["id"]},
                "detail": (
                    f"declared rate {audio['rate_declared']} != actual {audio['rate_actual']}; "
                    f"implied duration {declared_s:.6f}s vs actual {actual_s:.6f}s"
                ),
            }
        )

    if audio_eval is not None:
        limit = tol["audio_offset_samples"]
        for seg in audio_eval["segments"]:
            if abs(seg["offset_samples"]) > limit:
                findings.append(
                    {
                        "code": "E_AUDIO_DRIFT",
                        "location": {"segment": seg["shot"], "sample_start": seg["sample_start"]},
                        "detail": (
                            f"offset {seg['offset_samples']} samples exceeds tolerance "
                            f"{limit} in segment {seg['shot']} (first out-of-tolerance segment)"
                        ),
                    }
                )
                break
    return findings
