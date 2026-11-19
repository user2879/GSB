"""Online editing commands and deterministic audio materialization."""

import numpy as np

from .audioio import SAMPLE_RATE, read_wav, write_wav
from .errors import CommandError, OutOfRangeError


VALID_TYPES = {"trim", "replace", "merge_clusters", "split_cluster", "set_threshold"}


def load_command_stream(path):
    import json

    with open(path, "r", encoding="utf-8") as handle:
        stream = json.load(handle)
    commands = stream["commands"] if isinstance(stream, dict) else stream
    for ordinal, command in enumerate(commands):
        if command.get("type") not in VALID_TYPES:
            raise CommandError(
                f"unknown command type at index {ordinal}: {command.get('type')}",
                stage="command",
            )
        command.setdefault("sequence", ordinal)
    return commands


def _ms_to_samples(ms):
    return int(round(float(ms) * SAMPLE_RATE / 1000.0))


def _validate_range(start_ms, end_ms, n_samples, file_name,
                    injection_id=None):
    start_sample = _ms_to_samples(start_ms)
    end_sample = _ms_to_samples(end_ms)
    if start_sample < 0 or end_sample < start_sample:
        raise OutOfRangeError(
            f"invalid range [{start_ms}, {end_ms}) in {file_name}",
            stage="command",
            injection_id=injection_id,
        )
    if end_sample > n_samples:
        raise OutOfRangeError(
            f"range [{start_ms}, {end_ms}) exceeds audio length "
            f"{n_samples * 1000.0 / SAMPLE_RATE} ms in {file_name}",
            stage="command",
            injection_id=injection_id,
        )
    return start_sample, end_sample


def _replacement_samples(length_samples, command):
    """Deterministic replacement signal derived from command fields."""
    seed = int(
        command.get(
            "replace_seed",
            abs(
                hash(
                    "|".join(
                        str(command.get(key, ""))
                        for key in ("file", "start_ms", "end_ms", "sequence")
                    )
                )
            )
            % (2 ** 32),
        )
    )
    rng = np.random.default_rng(seed)
    frequency = float(command.get("frequency_hz", 220.0))
    amplitude = float(command.get("amplitude", 0.35))
    time_axis = np.arange(length_samples, dtype=np.float64) / SAMPLE_RATE
    tone = np.sin(2.0 * np.pi * frequency * time_axis)
    noise = rng.normal(0.0, 0.01, size=length_samples)
    return amplitude * tone + noise


def apply_audio_command(command, work_dir):
    """Materialize a trim/replace command by rewriting the working WAV."""
    from os.path import join

    file_name = command["file"]
    path = join(work_dir, file_name)
    samples, _ = read_wav(path)
    n_samples = len(samples)

    if command["type"] == "trim":
        start_sample, end_sample = _validate_range(
            command["start_ms"], command["end_ms"], n_samples, file_name,
            command.get("injection_id"),
        )
        new_samples = np.concatenate(
            [samples[:start_sample], samples[end_sample:]]
        )
    elif command["type"] == "replace":
        start_sample, end_sample = _validate_range(
            command["start_ms"], command["end_ms"], n_samples, file_name,
            command.get("injection_id"),
        )
        replacement = _replacement_samples(
            end_sample - start_sample, command
        )
        new_samples = np.concatenate(
            [
                samples[:start_sample],
                replacement,
                samples[end_sample:],
            ]
        )
    else:
        raise CommandError(
            f"not an audio command: {command['type']}", stage="command"
        )
    write_wav(path, new_samples)
    return {
        "file": file_name,
        "type": command["type"],
        "start_ms": command["start_ms"],
        "end_ms": command["end_ms"],
        "new_length_samples": int(len(new_samples)),
    }


def update_override_ledger(ledger, command, current_clusters,
                           interval_fingerprints):
    """Translate a merge/split command into a stable ledger entry."""
    assignments = {}
    for cluster in current_clusters:
        for fp in cluster["interval_fingerprints"]:
            assignments[fp] = cluster["cluster_id"]

    if command["type"] == "merge_clusters":
        cluster_ids = list(command["cluster_ids"])
        ledger.append(
            {
                "op": "merge",
                "sequence": command["sequence"],
                "cluster_ids": cluster_ids,
            }
        )
    elif command["type"] == "split_cluster":
        source = command["source_cluster_id"]
        member_fps = [
            fp
            for fp in interval_fingerprints
            if assignments.get(fp) == source
        ]
        count = int(command.get("member_count", 1))
        chosen = member_fps[:count] if member_fps else list(
            command.get("interval_fingerprints", [])
        )
        ledger.append(
            {
                "op": "split",
                "sequence": command["sequence"],
                "source_cluster_id": source,
                "interval_fingerprints": chosen,
            }
        )
    return ledger
