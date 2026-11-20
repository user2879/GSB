"""End-to-end acceptance check: incremental edits vs one-shot full decode."""

from __future__ import annotations

import dataclasses
import json
import os


from asr_stream_editor.audio import load_wav, plan_chunks
from asr_stream_editor.engine import StreamingEngine
from asr_stream_editor.full_decode import full_decode
from asr_stream_editor.runner import load_resources
from asr_stream_editor.snapshot import SnapshotStore

COMPARED = (
    "transcript", "text", "frame_labels", "frame_owner",
    "alignment_conflicts", "failures", "stable_prefix", "stable_prefix_text",
    "mutable_tail", "stable_until_frame", "boundary_runs",
)


def main(samples_dir: str, work_dir: str) -> None:
    pm, ac, sc, clips, audio_failures = load_resources(samples_dir)
    sc = dataclasses.replace(sc, redecode_budget_ms=100000)
    commands = json.load(open(os.path.join(samples_dir, "command_stream.json")))
    all_ok = True
    for clip in clips:
        store = SnapshotStore(os.path.join(work_dir, "snap", clip.source))
        engine = StreamingEngine(clip, pm, ac, sc, store, {})
        edited_samples = clip.samples.copy()
        final_overlap = sc.overlap_ms
        final_stable = sc.min_consistent_frames
        final_scale = 1.0
        for command in commands:
            replacement = None
            if command.get("command") == "replace_chunk":
                chunks = plan_chunks(edited_samples.shape[0], sc.chunk_ms,
                                     engine.overlap_ms)
                target = chunks[command["chunk_index"]]
                region_len = target.own_end_sample - target.own_start_sample
                rep_clip = load_wav(os.path.join(
                    samples_dir, command["replacement_wav"]))
                replacement = rep_clip.samples[:region_len]
                edited_samples[target.own_start_sample:target.own_end_sample] = replacement
            elif command.get("command") == "set_overlap":
                final_overlap = command["overlap_ms"]
            elif command.get("command") == "set_stable_frames":
                final_stable = command["min_consistent_frames"]
            elif command.get("command") == "set_backoff":
                final_scale = command["backoff_scale"]
            response = engine.process_command(command, replacement)
            while response.code is not None:
                response = engine.process_command({"command": "resume"})
        reference = full_decode(clip, edited_samples, pm, ac, sc,
                                overlap_ms=final_overlap,
                                min_consistent_frames=final_stable,
                                backoff_scale=final_scale)
        frame_shift_ms = ac.frame_shift_ms
        from asr_stream_editor.stability import split_tokens
        prefix, tail = split_tokens(engine.merged.tokens,
                                    engine.stability.stable_until_frame)
        engine_state = {
            "transcript": [t.to_dict(frame_shift_ms) for t in engine.merged.tokens],
            "text": "".join(t.word for t in engine.merged.tokens),
            "frame_labels": list(engine.merged.frame_labels),
            "frame_owner": list(engine.merged.frame_owner),
            "alignment_conflicts": list(engine.merged.conflicts),
            "failures": [f.to_dict() for f in engine.failures],
            "stable_prefix": [t.to_dict(frame_shift_ms) for t in prefix],
            "stable_prefix_text": "".join(t.word for t in prefix),
            "mutable_tail": [t.to_dict(frame_shift_ms) for t in tail],
            "stable_until_frame": engine.stability.stable_until_frame,
            "boundary_runs": engine.stability.boundary_runs,
        }
        mismatches = [field for field in COMPARED
                      if engine_state[field] != reference[field]]
        status = "OK" if not mismatches else f"MISMATCH {mismatches}"
        print(f"{clip.source}: words={len(engine_state['transcript'])} {status}")
        all_ok = all_ok and not mismatches
    print("audio format failures:", [f["source"] for f in audio_failures])
    if not all_ok:
        raise SystemExit(1)
    print("ALL CLIPS MATCH ONE-SHOT FULL DECODE FIELD-BY-FIELD")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "samples",
         sys.argv[2] if len(sys.argv) > 2 else "artifacts/verify")
