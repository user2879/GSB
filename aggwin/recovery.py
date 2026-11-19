"""Recovery paths: state-log replay and snapshot rebuild, plus comparison."""

from .canon import digest_obj
from .engine import Engine
from .errors import PipelineError
from .persistence import REC_BATCH, REC_CHECKPOINT, read_log, read_snapshot
from .state import EngineState


def state_digest(state):
    return digest_obj(state.to_dict())


def replay_log(log_dir, tables, query_set, param_version, logger=None):
    """Rebuild state from the latest checkpoint plus subsequent batches."""
    records = read_log(log_dir)
    engine = Engine(tables, query_set, param_version, logger)
    start = 0
    for pos in range(len(records) - 1, -1, -1):
        if records[pos].record_type == REC_CHECKPOINT:
            engine.state = EngineState.from_dict(
                records[pos].payload["state"])
            start = pos + 1
            break
    for record in records[start:]:
        if record.record_type != REC_BATCH:
            continue
        engine.apply_rows(record.payload["table"], record.payload["rows"])
        if "cursor_after" in record.payload:
            engine.state.cursors[record.payload["table"]] = \
                record.payload["cursor_after"]
    # Fallback: derive cursors from applied intervals (idempotent replay).
    for table_name, intervals in engine.state.applied.items():
        if intervals and engine.state.cursors.get(table_name, 0) == 0:
            engine.state.cursors[table_name] = max(
                engine.state.cursors.get(table_name, 0),
                max(end for _s, end in intervals))
    engine.finalize()
    return engine


def rebuild_from_snapshot(path):
    """Load a snapshot manifest wholesale (no incremental merge)."""
    record = read_snapshot(path)
    return record, EngineState.from_dict(record.state)


def _first_diff(a, b, path):
    """Return the first differing path between two canonical structures."""
    if type(a) is not type(b):
        return path
    if isinstance(a, dict):
        if set(a) != set(b):
            return path
        for key in sorted(a):
            found = _first_diff(a[key], b[key], path + "/" + str(key))
            if found:
                return found
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return path
        for idx, (x, y) in enumerate(zip(a, b)):
            found = _first_diff(x, y, path + f"[{idx}]")
            if found:
                return found
        return None
    return path if a != b else None


def compare_states(state_a, state_b):
    """Bitwise-compare two engine states; E_RECOVER on first divergence."""
    dict_a, dict_b = state_a.to_dict(), state_b.to_dict()
    if dict_a == dict_b:
        return True
    path = _first_diff(dict_a, dict_b, "")
    details = _locate(path, dict_a, dict_b)
    raise PipelineError("E_RECOVER",
                        "log replay and snapshot rebuild diverge",
                        first_diff_path=path, **details)


def _locate(path, dict_a, dict_b):
    """Extract group key / partition key / field context from a diff path."""
    details = {}
    parts = [p for p in path.split("/") if p]
    if "aggs" in parts:
        idx = parts.index("aggs")
        if len(parts) > idx + 2:
            details["group_key"] = parts[idx + 2]
        if len(parts) > idx + 3:
            details["field"] = parts[idx + 3].split("[")[0]
    if "windows" in parts:
        idx = parts.index("windows")
        if len(parts) > idx + 2:
            details["partition_key"] = parts[idx + 2]
        if len(parts) > idx + 3:
            details["field"] = parts[idx + 3].split("[")[0]
    if "cursors" in parts:
        details["field"] = "cursor"
    node_a, node_b = dict_a, dict_b
    for part in parts:
        key, _, index = part.partition("[")
        node_a = node_a.get(key) if isinstance(node_a, dict) else None
        node_b = node_b.get(key) if isinstance(node_b, dict) else None
        if index and isinstance(node_a, list) and isinstance(node_b, list):
            pos = int(index.rstrip("]"))
            node_a = node_a[pos] if pos < len(node_a) else None
            node_b = node_b[pos] if pos < len(node_b) else None
    details["log_replay_value"] = node_a
    details["snapshot_value"] = node_b
    return details
