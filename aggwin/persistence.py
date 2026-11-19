"""Versioned persistence: state-log segments and snapshot manifests.

State log record layout (little endian):
  v1: magic(4) fmt(u32) seq(u64) type(u8) payload_len(u32) payload crc32(u32)
      sha256(32)
  v2: magic(4) fmt(u32) seq(u64) checkpoint_seq(u64) type(u8)
      payload_len(u32) payload crc32(u32) parent_digest(32) sha256(32)

Every record is written with a single O_APPEND os.write + fsync so a
record is atomically visible to concurrent readers.  Decoding validates
magic, version, length self-consistency, crc and content digest; any
violation raises E_LOG with the segment file and byte offset.

Snapshot manifests are JSON documents (format_version 1 or 2) carrying a
content digest and a parent reference; a digest mismatch raises
E_SNAPSHOT.  Both versions normalise to the same in-memory record, so
cross-version normalisation never changes the output.
"""

import json
import os
import struct
import zlib

from .canon import canonical_bytes, digest_obj, sha256_hex
from .errors import PipelineError

LOG_MAGIC_V1 = b"AWL1"
LOG_MAGIC_V2 = b"AWL2"
LOG_FORMAT_VERSIONS = (1, 2)
SNAPSHOT_FORMAT_VERSIONS = (1, 2)

REC_BATCH = 1
REC_CHECKPOINT = 2

_ZERO_DIGEST = b"\x00" * 32


class LogRecord:
    def __init__(self, log_seq, record_type, payload, checkpoint_seq=0,
                 parent_digest=_ZERO_DIGEST, format_version=2):
        self.log_seq = log_seq
        self.record_type = record_type
        self.payload = payload            # dict
        self.checkpoint_seq = checkpoint_seq
        self.parent_digest = parent_digest
        self.format_version = format_version


def _encode_v1(rec):
    payload = canonical_bytes(rec.payload)
    head = LOG_MAGIC_V1 + struct.pack("<IQBI", 1, rec.log_seq,
                                      rec.record_type, len(payload))
    crc = zlib.crc32(head + payload) & 0xFFFFFFFF
    return head + payload + struct.pack("<I", crc) + \
        bytes.fromhex(sha256_hex(payload))


def _encode_v2(rec):
    payload = canonical_bytes(rec.payload)
    head = LOG_MAGIC_V2 + struct.pack("<IQQBI", 2, rec.log_seq,
                                      rec.checkpoint_seq, rec.record_type,
                                      len(payload))
    crc = zlib.crc32(head + payload) & 0xFFFFFFFF
    return head + payload + struct.pack("<I", crc) + rec.parent_digest + \
        bytes.fromhex(sha256_hex(payload))


def encode_record(rec):
    if rec.format_version == 1:
        return _encode_v1(rec)
    if rec.format_version == 2:
        return _encode_v2(rec)
    raise PipelineError("E_LOG", "unsupported log format version",
                        format_version=rec.format_version)


def _decode_one(buf, offset, segment):
    """Decode one record; return (LogRecord, next_offset)."""

    def fail(message):
        raise PipelineError("E_LOG", message, segment=segment, offset=offset)

    if len(buf) - offset < 4:
        fail("truncated record: missing magic")
    magic = buf[offset:offset + 4]
    if magic == LOG_MAGIC_V1:
        fixed = struct.calcsize("<IQBI")
        if len(buf) - offset < 4 + fixed:
            fail("truncated v1 record header")
        fmt, seq, rtype, plen = struct.unpack_from("<IQBI", buf, offset + 4)
        checkpoint_seq, parent_digest = 0, _ZERO_DIGEST
        digest_len = 32
    elif magic == LOG_MAGIC_V2:
        fixed = struct.calcsize("<IQQBI")
        if len(buf) - offset < 4 + fixed:
            fail("truncated v2 record header")
        fmt, seq, checkpoint_seq, rtype, plen = struct.unpack_from(
            "<IQQBI", buf, offset + 4)
        digest_len = 64
    else:
        fail(f"bad magic {magic!r}")
    if fmt not in LOG_FORMAT_VERSIONS:
        fail(f"unsupported log format version {fmt}")
    body_start = offset + 4 + fixed
    body_end = body_start + plen
    tail_end = body_end + 4 + digest_len
    if len(buf) < tail_end:
        fail(f"truncated record: payload length {plen} exceeds segment")
    payload = buf[body_start:body_end]
    (crc,) = struct.unpack_from("<I", buf, body_end)
    if crc != (zlib.crc32(buf[offset:body_end]) & 0xFFFFFFFF):
        fail("crc mismatch")
    cursor = body_end + 4
    if magic == LOG_MAGIC_V2:
        parent_digest = buf[cursor:cursor + 32]
        cursor += 32
    digest = buf[cursor:cursor + 32]
    if digest.hex() != sha256_hex(payload):
        fail("content digest mismatch")
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        fail(f"payload is not valid JSON: {exc}")
    record = LogRecord(seq, rtype, decoded, checkpoint_seq, parent_digest,
                       fmt)
    return record, tail_end


class StateLogWriter:
    """Appends records to segment files with atomic single-write publish."""

    def __init__(self, log_dir, format_version=2):
        self.log_dir = log_dir
        self.format_version = format_version
        os.makedirs(log_dir, exist_ok=True)
        self._seq = 0
        self._checkpoint_seq = 0
        self._last_digest = _ZERO_DIGEST
        self._segment_index = self._next_segment_index()
        self._fh = None

    def _next_segment_index(self):
        existing = [f for f in os.listdir(self.log_dir)
                    if f.startswith("segment-") and f.endswith(".log")]
        if not existing:
            return 1
        return max(int(f[len("segment-"):-len(".log")]) for f in existing) + 1

    def _segment_path(self, index=None):
        return os.path.join(self.log_dir,
                            f"segment-{(index or self._segment_index):06d}.log")

    def _write(self, record_type, payload):
        record = LogRecord(self._seq + 1, record_type, payload,
                           self._checkpoint_seq, self._last_digest,
                           self.format_version)
        blob = encode_record(record)
        path = self._segment_path()
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, blob)  # single atomic append
            os.fsync(fd)
        finally:
            os.close(fd)
        self._seq = record.log_seq
        self._last_digest = bytes.fromhex(sha256_hex(canonical_bytes(payload)))
        if record_type == REC_CHECKPOINT:
            self._checkpoint_seq = record.log_seq
            self._segment_index += 1  # new segment after each checkpoint
        return record

    def write_batch(self, table, rows, cursor_after=None):
        payload = {"table": table, "rows": list(rows)}
        if cursor_after is not None:
            payload["cursor_after"] = cursor_after
        return self._write(REC_BATCH, payload)

    def write_checkpoint(self, state_dict):
        return self._write(REC_CHECKPOINT, {"state": state_dict})


def read_segment(path):
    """Decode every record of one segment; E_LOG on any violation."""
    with open(path, "rb") as fh:
        buf = fh.read()
    records, offset = [], 0
    while offset < len(buf):
        record, offset = _decode_one(buf, offset, path)
        records.append(record)
    return records


def read_log(log_dir):
    """Read all segments in order; validate monotonic log sequence."""
    segment_files = sorted(
        f for f in os.listdir(log_dir)
        if f.startswith("segment-") and f.endswith(".log"))
    records = []
    last_seq = 0
    for name in segment_files:
        path = os.path.join(log_dir, name)
        for record in read_segment(path):
            if record.log_seq <= last_seq:
                raise PipelineError(
                    "E_LOG", "log sequence is not monotonic",
                    segment=path, log_seq=record.log_seq)
            last_seq = record.log_seq
            records.append(record)
    return records


# --------------------------------------------------------------------------
# Snapshot manifests
# --------------------------------------------------------------------------

def write_snapshot(path, state_dict, query_param_version, snapshot_id,
                   parent=None, format_version=2):
    """Write a snapshot manifest atomically (temp file + rename)."""
    if format_version == 1:
        body = {"format_version": 1, "snapshot_id": snapshot_id,
                "query_param_version": query_param_version,
                "parent_snapshot_id": parent["snapshot_id"] if parent else None,
                "state": state_dict}
    elif format_version == 2:
        body = {"format_version": 2, "snapshot_id": snapshot_id,
                "query_param_version": query_param_version,
                "parent": {"snapshot_id": parent["snapshot_id"],
                           "digest": parent["digest"]} if parent else None,
                "state": state_dict}
    else:
        raise PipelineError("E_SNAPSHOT", "unsupported snapshot format",
                            format_version=format_version)
    body["digest"] = digest_obj(body)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(body, fh, ensure_ascii=False, sort_keys=True,
                  separators=(",", ":"))
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_path, path)
    return body


class SnapshotRecord:
    def __init__(self, snapshot_id, query_param_version, state, parent_id,
                 digest, format_version):
        self.snapshot_id = snapshot_id
        self.query_param_version = query_param_version
        self.state = state
        self.parent_id = parent_id
        self.digest = digest
        self.format_version = format_version


def read_snapshot(path):
    """Decode and validate a snapshot manifest of any supported version."""
    with open(path, "r", encoding="utf-8") as fh:
        try:
            body = json.load(fh)
        except json.JSONDecodeError as exc:
            raise PipelineError("E_SNAPSHOT", f"manifest is not JSON: {exc}",
                                snapshot=path)
    if not isinstance(body, dict):
        raise PipelineError("E_SNAPSHOT", "manifest must be an object",
                            snapshot=path)
    fmt = body.get("format_version")
    if fmt not in SNAPSHOT_FORMAT_VERSIONS:
        raise PipelineError("E_SNAPSHOT",
                            f"unsupported snapshot format version {fmt!r}",
                            snapshot=path)
    digest = body.get("digest")
    if not isinstance(digest, str):
        raise PipelineError("E_SNAPSHOT", "missing content digest",
                            snapshot=path)
    unsigned = {k: v for k, v in body.items() if k != "digest"}
    if digest_obj(unsigned) != digest:
        raise PipelineError("E_SNAPSHOT", "snapshot digest mismatch",
                            snapshot=path)
    if fmt == 1:
        parent_id = body.get("parent_snapshot_id")
    else:
        parent = body.get("parent")
        parent_id = parent["snapshot_id"] if parent else None
    return SnapshotRecord(body["snapshot_id"], body["query_param_version"],
                          body["state"], parent_id, digest, fmt)
