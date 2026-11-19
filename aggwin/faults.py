"""Fault injection helpers.  Injections only modify input files or
parameters, never engine code paths."""

import json
import os


def corrupt_snapshot_digest(path):
    """snapshot-digest: flip characters of the digest inside the manifest."""
    with open(path, "r", encoding="utf-8") as fh:
        body = fh.read()
    manifest = json.loads(body)
    digest = manifest["digest"]
    flipped = ("0" if digest[0] != "0" else "1") + digest[1:]
    manifest["digest"] = flipped
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, sort_keys=True,
                  separators=(",", ":"))
    return {"file": path, "original_digest": digest,
            "injected_digest": flipped}


def truncate_log_tail(log_dir, nbytes=7):
    """log-truncate: cut bytes off the tail of the newest segment."""
    segments = sorted(f for f in os.listdir(log_dir)
                      if f.startswith("segment-") and f.endswith(".log"))
    if not segments:
        raise FileNotFoundError(f"no segments in {log_dir}")
    path = os.path.join(log_dir, segments[-1])
    size = os.path.getsize(path)
    with open(path, "r+b") as fh:
        fh.truncate(size - nbytes)
    return {"file": path, "truncated_bytes": nbytes,
            "size_before": size, "size_after": size - nbytes}


def inject_batch_mid_window(session, batch_size=3):
    """batch-mid-window: strict alignment check + boundary-splitting size."""
    mutated = dict(session)
    mutated["window_aligned_batches"] = True
    mutated["batch_size"] = batch_size
    return mutated


def inject_out_of_order(session, shuffle_window=25, shuffle_seed=99,
                        max_ratio=0.05):
    """out-of-order: arrival disorder beyond the session lateness limit."""
    mutated = dict(session)
    mutated["arrival"] = {"shuffle_seed": shuffle_seed,
                          "shuffle_window": shuffle_window}
    mutated["max_out_of_order_ratio"] = max_ratio
    return mutated
