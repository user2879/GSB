from __future__ import annotations

import struct
import zlib

import pytest

from haptic_diag.codec import (
    HEADER,
    ProtocolFormatError,
    WireChannel,
    WireFrame,
    decode_frame,
    downgrade_to_v1,
    encode_frame,
    migrate_to_v2,
)
from haptic_diag.constants import (
    E_PROTO_FORMAT,
    OLD_CHANNEL_BITS,
    OLD_PROTOCOL_FREQUENCY_HZ,
    PROTOCOL_FLAG_MIGRATED,
    PROTOCOL_MAGIC,
    PROTOCOL_V1,
    PROTOCOL_V2,
)
from haptic_diag.generator import load_config, generate_sample
from haptic_diag.compiler import compile_graph
from haptic_diag.quantization import DEFAULT_QUANT_PROFILE
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _sample_graph():
    config = load_config(ROOT / "configs" / "samples.json")
    return generate_sample(config["samples"][2])


def test_v2_binary_roundtrip_is_exact() -> None:
    frame = WireFrame(
        seq=42,
        timestamp_us=5_000_000,
        protocol_version=PROTOCOL_V2,
        channels=(
            WireChannel("left_grip", 1234, 5678),
            WireChannel("right_trigger", 65535, 0),
        ),
    )
    encoded = encode_frame(frame)
    assert decode_frame(encoded) == frame
    assert encode_frame(decode_frame(encoded)) == encoded


def test_v1_binary_roundtrip_uses_old_mask_bits() -> None:
    frame = WireFrame(
        seq=3,
        timestamp_us=2500,
        protocol_version=PROTOCOL_V1,
        channels=(
            WireChannel("left_grip", 100, None),
            WireChannel("right_grip", 200, None),
            WireChannel("left_trigger", 300, None),
            WireChannel("right_trigger", 400, None),
        ),
    )
    encoded = encode_frame(frame)
    mask = encoded[16]
    expected_mask = sum(OLD_CHANNEL_BITS.values())
    assert mask == expected_mask
    assert decode_frame(encoded) == frame


def test_v1_to_v2_migration_is_idempotent_and_uses_fixed_table() -> None:
    frame = WireFrame(
        seq=5,
        timestamp_us=100,
        protocol_version=PROTOCOL_V1,
        channels=(WireChannel("right_grip", 777, None),),
    )
    first = migrate_to_v2(frame)
    second = migrate_to_v2(first)
    assert first == second
    assert encode_frame(first) == encode_frame(second)
    assert first.flags & PROTOCOL_FLAG_MIGRATED
    assert first.channels[0].frequency_code == DEFAULT_QUANT_PROFILE.encode_frequency(
        OLD_PROTOCOL_FREQUENCY_HZ["right_grip"]
    )


def test_v2_to_v1_roundtrip_preserves_timestamps_and_amplitudes() -> None:
    result = compile_graph(_sample_graph(), PROTOCOL_V2)
    for record in result.frames[:50]:
        wire = record.wire_frame()
        migrated = migrate_to_v2(decode_frame(encode_frame(downgrade_to_v1(wire))))
        assert migrated.timestamp_us == wire.timestamp_us
        assert [item.amplitude_code for item in migrated.channels] == [
            item.amplitude_code for item in wire.channels
        ]
        assert all(item.frequency_code is not None for item in migrated.channels)


def test_unmigratable_frame_returns_e_proto_format_with_seq_and_field() -> None:
    frame = WireFrame(
        seq=19,
        timestamp_us=10,
        protocol_version=PROTOCOL_V2,
        channels=(WireChannel("left_grip", 1, 2),),
    )
    encoded = bytearray(encode_frame(frame))
    encoded[16] = 0x20  # invalid v2 mask bit
    encoded[-4:] = struct.pack(">I", zlib.crc32(encoded[:-4]) & 0xFFFFFFFF)
    with pytest.raises(ProtocolFormatError) as caught:
        decode_frame(bytes(encoded))
    assert caught.value.code == E_PROTO_FORMAT
    assert caught.value.frame_seq == 19
    assert caught.value.field == "channel_mask"


def test_truncated_frequency_payload_and_crc_report_precise_fields() -> None:
    frame = WireFrame(
        seq=21,
        timestamp_us=10,
        protocol_version=PROTOCOL_V2,
        channels=(WireChannel("left_grip", 1, 2),),
    )
    encoded = encode_frame(frame)
    with pytest.raises(ProtocolFormatError) as truncated:
        decode_frame(encoded[:-1])
    assert truncated.value.frame_seq == 21
    assert truncated.value.field == "frequency"
    with pytest.raises(ProtocolFormatError) as amplitude_truncated:
        decode_frame(encoded[:-3])
    assert amplitude_truncated.value.frame_seq == 21
    assert amplitude_truncated.value.field == "amplitude"
    corrupted = bytearray(encoded)
    corrupted[-1] ^= 0xFF
    with pytest.raises(ProtocolFormatError) as crc:
        decode_frame(bytes(corrupted))
    assert crc.value.frame_seq == 21
    assert crc.value.field == "crc"


def test_invalid_magic_reports_protocol_format() -> None:
    frame = WireFrame(
        seq=2,
        timestamp_us=0,
        protocol_version=PROTOCOL_V2,
        channels=(),
    )
    encoded = bytearray(encode_frame(frame))
    encoded[0:2] = b"XX"
    encoded[-4:] = struct.pack(">I", zlib.crc32(encoded[:-4]) & 0xFFFFFFFF)
    with pytest.raises(ProtocolFormatError) as caught:
        decode_frame(bytes(encoded))
    assert caught.value.field == "magic"
    assert caught.value.frame_seq == 2
