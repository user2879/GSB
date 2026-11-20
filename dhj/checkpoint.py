"""Per-bucket checkpoints for generation, upgrade and join phases."""

import os

from .jsonio import read_json, write_json

PHASE_GENERATE = "generate"
PHASE_UPGRADE = "upgrade"
PHASE_JOIN = "join"


def checkpoint_path(directory, phase):
    return os.path.join(directory, f"checkpoint-{phase}.json")


def load_checkpoint(directory, phase):
    path = checkpoint_path(directory, phase)
    if not os.path.exists(path):
        return empty_checkpoint(phase)
    data = read_json(path)
    if data.get("phase") != phase:
        data = empty_checkpoint(phase)
    data.setdefault("entries", {})
    return data


def empty_checkpoint(phase):
    return {"phase": phase, "entries": {}}


def save_checkpoint(directory, phase, checkpoint, atomic=True):
    write_json(
        checkpoint_path(directory, phase), checkpoint, atomic=atomic
    )


def mark_entry(checkpoint, partition, side, **fields):
    key = f"{side}:{partition}"
    entry = dict(checkpoint["entries"].get(key, {}))
    entry.update({"partition": partition, "side": side})
    entry.update(fields)
    checkpoint["entries"][key] = entry
    return entry


def is_done(checkpoint, partition, side):
    return f"{side}:{partition}" in checkpoint["entries"]


def done_partitions(checkpoint, side):
    out = []
    for key, entry in checkpoint["entries"].items():
        if entry.get("side") == side:
            out.append(entry["partition"])
    return sorted(out)
