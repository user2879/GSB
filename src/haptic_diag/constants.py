"""Shared protocol and channel constants.

The new (v2) protocol uses a dense channel-mask layout.  The legacy (v1)
protocol used a sparse layout and had no per-channel frequency field.
"""

from __future__ import annotations

CHANNELS: tuple[str, ...] = (
    "left_grip",
    "right_grip",
    "left_trigger",
    "right_trigger",
)

CHANNEL_INDEX = {name: index for index, name in enumerate(CHANNELS)}

NEW_CHANNEL_BITS = {
    "left_grip": 0x01,
    "right_grip": 0x02,
    "left_trigger": 0x04,
    "right_trigger": 0x08,
}

OLD_CHANNEL_BITS = {
    "left_grip": 0x01,
    "right_grip": 0x04,
    "left_trigger": 0x10,
    "right_trigger": 0x40,
}

OLD_TO_NEW_MASK = {
    OLD_CHANNEL_BITS[name]: NEW_CHANNEL_BITS[name] for name in CHANNELS
}
NEW_TO_OLD_MASK = {
    NEW_CHANNEL_BITS[name]: OLD_CHANNEL_BITS[name] for name in CHANNELS
}

# Fixed migration table for the field that does not exist in v1.  Values are
# expressed in hertz and are deliberately part of the versioned protocol
# compatibility contract.
OLD_PROTOCOL_FREQUENCY_HZ = {
    "left_grip": 100.0,
    "right_grip": 120.0,
    "left_trigger": 140.0,
    "right_trigger": 160.0,
}

AMPLITUDE_CODE_MAX = 65535
FREQUENCY_CODE_MAX = 65535
MAX_FREQUENCY_HZ = 2000.0
AMPLITUDE_STEP = 1.0 / AMPLITUDE_CODE_MAX
FREQUENCY_STEP = MAX_FREQUENCY_HZ / FREQUENCY_CODE_MAX

PROTOCOL_V1 = 1
PROTOCOL_V2 = 2
SUPPORTED_PROTOCOLS = (PROTOCOL_V1, PROTOCOL_V2)

PROTOCOL_MAGIC = b"HF"
PROTOCOL_FLAG_MIGRATED = 0x01
PROTOCOL_FLAG_LOSSY_FREQUENCY = 0x02

E_PROTO_FORMAT = "E_PROTO_FORMAT"
E_BUDGET = "E_BUDGET"
E_SEQUENCE = "E_SEQUENCE"

