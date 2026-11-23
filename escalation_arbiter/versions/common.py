"""Shared helpers for the three exchange versions."""

from ..fixedpoint import float_to_permille, permille_to_float

V1 = "v1"
V2 = "v2"
V3 = "v3"
VERSIONS = (V1, V2, V3)

ROLE_ALIASES = {
    "customer": "CUS",
    "agent": "AGT",
    "system": "SYS",
    "auditor": "AUD",
    "partner": "PTR",
}
ROLE_ALIASES_REV = {v: k for k, v in ROLE_ALIASES.items()}
ROLE_FALLBACK = "UNK"

# v1 can only display a reason-code index from a fixed table of six slots.
V1_REASON_SLOTS = ("MX1", "MX2", "MX3", "SP1", "SP2", "SP3")


def alias_role(role: str) -> str:
    return ROLE_ALIASES.get(role, ROLE_FALLBACK)


def unalias_role(token: str) -> str:
    return ROLE_ALIASES_REV.get(token, token)


def to_display_float(permille: int) -> float:
    return permille_to_float(permille)


def from_display_float(value: float) -> int:
    return float_to_permille(value)

