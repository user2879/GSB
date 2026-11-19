"""Integrate the public modules directly without the task orchestrator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audiofp.fingerprint import fingerprint_audio
from audiofp.index import ReferenceIndex
from audiofp.retrieval import fragment_to_dict, retrieve_similar_fragments
from audiofp.runtime import CancelToken, FrameBudget
from audiofp.samplegen import generate_demo
from audiofp.task import load_task
from audiofp.wavio import read_wav


def main() -> int:
    parser = argparse.ArgumentParser(description="Component-level fingerprint integration example")
    parser.add_argument("--workspace", default=".")
    args = parser.parse_args()
    workspace = Path(args.workspace).resolve()
    task_path = workspace / "task.json"
    if not task_path.exists():
        generate_demo(workspace)
    task = load_task(task_path)
    token = CancelToken()
    budget = FrameBudget(32)
    index = ReferenceIndex(task.retrieval.bucket_capacity)
    for order, track in enumerate(task.references):
        audio = read_wav(
            track.path,
            track_id=track.track_id,
            role="reference",
            allowed_sample_rates=task.allowed_sample_rates,
            declared_sample_rate=track.declared_sample_rate,
        )
        fingerprint = fingerprint_audio(
            audio, task.fingerprint, budget=budget, priority=order, token=token
        )
        index.add_fingerprint(fingerprint, reference_order=order)
    track = task.queries[0]
    audio = read_wav(
        track.path,
        track_id=track.track_id,
        role="query",
        allowed_sample_rates=task.allowed_sample_rates,
        declared_sample_rate=track.declared_sample_rate,
    )
    query = fingerprint_audio(
        audio, task.fingerprint, budget=budget, priority=len(task.references), token=token
    )
    fragments = retrieve_similar_fragments(
        index,
        query,
        task.retrieval,
        task.tolerance,
        task.fingerprint.base_sample_rate,
        task.fingerprint.hop_length,
    )
    for fragment in fragments:
        payload = fragment_to_dict(fragment)
        print(
            f"{fragment.reference_id}: ref={payload['reference_start_sample']}.."
            f"{payload['reference_end_sample']} query={payload['query_start_sample']}.."
            f"{payload['query_end_sample']} score={payload['score']} "
            f"offset_frames={payload['integer_frame_offset']} "
            f"drift={payload['drift']['direction']} {payload['drift']['drift_percent']:.3f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
