"""Ledger data structure: committed states, stage hashes, op log, metrics."""

import os

from .config import LEDGER_SCHEMA_VERSION
from .hashing import hash_json
from .persistence import (
    atomic_write, atomic_write_json, read_ledger, write_ledger,
)


def new_ledger(state, frame_count, truth_hash):
    return {
        "ledger_schema_version": LEDGER_SCHEMA_VERSION,
        "frame_count": frame_count,
        "truth_hash": truth_hash,
        "committed_seq": 0,
        "states": [],            # list of {seq, state, state_hash}
        "current_seq": 0,
        "outputs": {},           # frame -> committed output record
        "metrics": {},           # frame -> metric dict (current seq)
        "interval_metrics": {},  # name -> pooled metrics (current seq)
        "op_log": [],
        "injection_files": [],
        "ledger_hash": "",
    }


def state_record(seq, state):
    return {
        "seq": seq,
        "state": state.to_dict(),
        "state_hash": hash_json(state.to_dict()),
    }


def append_state(ledger, state):
    seq = ledger["committed_seq"] + 1
    record = state_record(seq, state)
    ledger["states"].append(record)
    ledger["current_seq"] = seq
    ledger["committed_seq"] = seq
    return record


def log_op(ledger, op):
    """Append a deterministic operation record (seq assigned by caller)."""
    ledger["op_log"].append(op)


def commit_frame_output(work_dir, ledger, frame_index, uint8_frame,
                        stage_hashes, seq, recomputed_stages):
    rel = os.path.join("frames_out", "frame_%04d.u8" % frame_index)
    path = os.path.join(work_dir, *rel.split("/"))
    atomic_write(path, uint8_frame.tobytes(order="C"), binary=True)
    ledger["outputs"][str(frame_index)] = {
        "frame": frame_index,
        "seq": seq,
        "file": rel,
        "output_hash": stage_hashes["quantize"],
        "stage_hashes": dict(stage_hashes),
        "recomputed_stages": list(recomputed_stages),
    }


def save(work_dir, ledger):
    write_ledger(work_dir, ledger)


def load(work_dir):
    return read_ledger(work_dir)


def committed_frames(ledger, seq=None):
    """Frames whose committed output belongs to seq (default current)."""
    seq = seq or ledger["current_seq"]
    return sorted(
        rec["frame"] for rec in ledger["outputs"].values() if rec["seq"] == seq
    )
