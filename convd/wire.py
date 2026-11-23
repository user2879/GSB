"""Deterministic UDP datagram protocol for convergence pulses.

All integers are big-endian. Strings are length-prefixed UTF-8.
Every decode validates length self-consistency and reports the byte
offset of the first offending field via E_WIRE.
"""

import struct

from .errors import WireError

MAGIC_PULSE = b"CP1P"
MAGIC_RECEIPT = b"CP1R"
MAGIC_ERROR = b"CP1E"
MAGIC_PING = b"CP1N"
MAGIC_PONG = b"CP1O"
MAX_STR = 128


class _Reader:
    def __init__(self, buf):
        self.buf = buf
        self.pos = 0

    def take(self, size, what):
        if self.pos + size > len(self.buf):
            raise WireError(self.pos, "truncated %s: need %d bytes, have %d"
                            % (what, size, len(self.buf) - self.pos))
        chunk = self.buf[self.pos:self.pos + size]
        self.pos += size
        return chunk

    def u8(self, what):
        return self.take(1, what)[0]

    def u32(self, what):
        return struct.unpack(">I", self.take(4, what))[0]

    def i64(self, what):
        return struct.unpack(">q", self.take(8, what))[0]

    def text(self, what):
        offset = self.pos
        length = self.u8(what + ".len")
        if length == 0 or length > MAX_STR:
            raise WireError(offset, "field %s length %d out of range"
                            % (what, length))
        raw = self.take(length, what)
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            raise WireError(self.pos - length, "field %s is not utf-8" % what)

    def finish(self):
        if self.pos != len(self.buf):
            raise WireError(self.pos, "trailing %d bytes after datagram"
                            % (len(self.buf) - self.pos))


def _pack_text(value):
    raw = value.encode("utf-8")
    if not raw or len(raw) > MAX_STR:
        raise ValueError("text field out of range: %r" % value)
    return struct.pack(">B", len(raw)) + raw


def encode_pulse(scenario_id, round_no, lease_ms, target_digest, send_mono_ms):
    return (MAGIC_PULSE + _pack_text(scenario_id)
            + struct.pack(">IIq", round_no, lease_ms, send_mono_ms)
            + _pack_text(target_digest))


def encode_receipt(executor_id, round_no, obs_seq, obs_mono_ms, state_digest):
    return (MAGIC_RECEIPT + _pack_text(executor_id)
            + struct.pack(">IIq", round_no, obs_seq, obs_mono_ms)
            + _pack_text(state_digest))


def encode_error(executor_id, round_no, code):
    return (MAGIC_ERROR + _pack_text(executor_id)
            + struct.pack(">I", round_no) + _pack_text(code))


def encode_ping(scenario_id):
    return MAGIC_PING + _pack_text(scenario_id)


def encode_pong(executor_id):
    return MAGIC_PONG + _pack_text(executor_id)


def decode_datagram(buf):
    """Decode any datagram; returns ('pulse'|'receipt'|'error', dict)."""
    if not isinstance(buf, (bytes, bytearray)):
        raise WireError(0, "datagram must be bytes")
    reader = _Reader(bytes(buf))
    magic = reader.take(4, "magic")
    if magic == MAGIC_PULSE:
        msg = _decode_pulse(reader)
        kind = "pulse"
    elif magic == MAGIC_RECEIPT:
        msg = _decode_receipt(reader)
        kind = "receipt"
    elif magic == MAGIC_ERROR:
        msg = _decode_error(reader)
        kind = "error"
    elif magic == MAGIC_PING:
        msg = {"scenario_id": reader.text("scenario_id")}
        kind = "ping"
    elif magic == MAGIC_PONG:
        msg = {"executor_id": reader.text("executor_id")}
        kind = "pong"
    else:
        raise WireError(0, "unknown magic %r" % bytes(magic))
    reader.finish()
    return kind, msg


def _decode_pulse(reader):
    scenario_id = reader.text("scenario_id")
    round_offset = reader.pos
    round_no = reader.u32("round")
    lease_ms = reader.u32("lease_ms")
    send_mono_ms = reader.i64("send_mono_ms")
    target_digest = reader.text("target_digest")
    if round_no < 1:
        raise WireError(round_offset, "round %d out of range" % round_no)
    return {"scenario_id": scenario_id, "round": round_no,
            "lease_ms": lease_ms, "send_mono_ms": send_mono_ms,
            "target_digest": target_digest}


def _decode_receipt(reader):
    executor_id = reader.text("executor_id")
    round_offset = reader.pos
    round_no = reader.u32("round")
    obs_offset = reader.pos
    obs_seq = reader.u32("obs_seq")
    obs_mono_ms = reader.i64("obs_mono_ms")
    state_digest = reader.text("state_digest")
    if round_no < 1:
        raise WireError(round_offset, "round %d out of range" % round_no)
    if obs_seq < 1:
        raise WireError(obs_offset, "obs_seq %d out of range" % obs_seq)
    return {"executor_id": executor_id, "round": round_no,
            "obs_seq": obs_seq, "obs_mono_ms": obs_mono_ms,
            "state_digest": state_digest}


def _decode_error(reader):
    executor_id = reader.text("executor_id")
    round_no = reader.u32("round")
    code = reader.text("code")
    return {"executor_id": executor_id, "round": round_no, "code": code}


def validate_receipt(msg, scenario_id, round_no, last_obs_seq):
    """Semantic checks on a decoded receipt against orchestrator state.

    ``last_obs_seq`` maps executor_id -> last accepted observation seq.
    Raises WireError pointing at the offending field offset.
    """
    executor_id = msg["executor_id"]
    if msg["round"] != round_no:
        raise WireError(5 + 1 + len(executor_id.encode("utf-8")),
                        "receipt round %d does not match pulse round %d"
                        % (msg["round"], round_no))
    prev = last_obs_seq.get(executor_id, 0)
    if msg["obs_seq"] < prev:
        raise WireError(5 + 1 + len(executor_id.encode("utf-8")) + 4,
                        "obs_seq %d regressed (last %d)"
                        % (msg["obs_seq"], prev))
    return True
