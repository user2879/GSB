"""Fault injection: real input faults with traceable injection ids."""

import json
import os
import wave

import numpy as np

from .errors import InvalidThresholdError, OutOfRangeError

INJECTION_CODES = {
    "corrupt_wav": "E_AUDIO_FORMAT",
    "non_finite": "E_NON_FINITE",
    "zero_threshold": "E_INVALID_THRESHOLD",
    "replace_out_of_range": "E_OUT_OF_RANGE",
}


def load_injections(path):
    if path is None or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get("injections", payload)


def apply_file_injections(work_dir, injections):
    """Materialize corrupt_wav / non_finite faults on working copies."""
    hits = []
    for item in injections:
        inject_id = item["id"]
        fault = item["fault"]
        if "file" not in item or fault not in ("corrupt_wav", "non_finite"):
            continue
        target = os.path.join(work_dir, item["file"])
        if not os.path.exists(target):
            continue
        if fault == "corrupt_wav":
            with open(target, "r+b") as handle:
                handle.seek(8)
                handle.write(b"\x00\x00\x00\x00")
                handle.flush()
                os.fsync(handle.fileno())
            hits.append(
                {
                    "id": inject_id,
                    "fault": fault,
                    "file": item["file"],
                    "code": INJECTION_CODES[fault],
                    "stage": "io",
                }
            )
        elif fault == "non_finite":
            _write_non_finite(target, item.get("frame_index", 0))
            hits.append(
                {
                    "id": inject_id,
                    "fault": fault,
                    "file": item["file"],
                    "code": INJECTION_CODES[fault],
                    "stage": "feature",
                }
            )
    return hits


def _write_non_finite(path, frame_index):
    with wave.open(path, "rb") as wav:
        n = wav.getnframes()
        raw = wav.readframes(n)
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    position = min(int(frame_index), len(samples) - 1)
    # Float NaN cannot live in PCM, so we record the poisoning position in a
    # sidecar; the feature stage reads it and injects a real NaN there.
    sidecar = path + ".nonfinite"
    with open(sidecar, "w", encoding="utf-8") as handle:
        handle.write(str(position))


def read_non_finite_poison(path):
    sidecar = path + ".nonfinite"
    if not os.path.exists(sidecar):
        return None
    with open(sidecar, "r", encoding="utf-8") as handle:
        return int(handle.read().strip())


def check_threshold_injection(config, injections):
    for item in injections:
        if item["fault"] == "zero_threshold":
            raise InvalidThresholdError(
                "injected fault: cluster_distance_threshold forced to zero",
                stage="clustering",
                injection_id=item["id"],
            )


def check_command_injection(command):
    if command.get("injection_id"):
        raise OutOfRangeError(
            f"injected fault: replace range out of bounds in "
            f"{command.get('file')}",
            stage="command",
            injection_id=command["injection_id"],
        )
