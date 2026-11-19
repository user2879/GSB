"""Deterministic pseudo-random generation.

The solver must never rely on the global :mod:`random` state, NumPy's global
RNG or the system clock.  A self-contained SplitMix64-based generator is used
instead; its byte-level behaviour is stable across Python/NumPy versions and
platforms, which keeps grid outputs bit-for-bit reproducible.
"""

from __future__ import annotations

import typing

_MASK = (1 << 64) - 1


def _mix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & _MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK
    return value ^ (value >> 31)


def derive_seed(material: typing.Any) -> int:
    """Derive a 64-bit seed deterministically from JSON-serialisable material.

    Integers are folded arithmetically; any other value is first turned into
    its canonical JSON text.  The result only depends on the input value.
    """
    if isinstance(material, bool) or not isinstance(material, int):
        import json

        text = json.dumps(material, sort_keys=True, separators=(",", ":"))
        material = 0
        for byte in text.encode("utf-8"):
            material = _mix64(material ^ byte)
    return material & _MASK


class DeterministicRNG:
    """SplitMix64 stream supporting seeding and independent sub-streams."""

    def __init__(self, seed: int) -> None:
        self._state = int(seed) & _MASK

    def next64(self) -> int:
        self._state = (self._state + 0x9E3779B97F4A7C15) & _MASK
        return _mix64(self._state)

    def randrange(self, stop: int) -> int:
        if stop <= 0:
            raise ValueError("randrange stop must be positive")
        return self.next64() % stop

    def randint(self, low: int, high: int) -> int:
        """Inclusive uniform integer in ``[low, high]``."""
        if high < low:
            raise ValueError("randint: empty range")
        return low + self.randrange(high - low + 1)

    def choice(self, sequence: typing.Sequence):
        if not sequence:
            raise IndexError("choice from empty sequence")
        return sequence[self.randrange(len(sequence))]

    def shuffled(self, sequence: typing.Sequence) -> list:
        """Return a deterministically Fisher-Yates shuffled copy."""
        result = list(sequence)
        for index in range(len(result) - 1, 0, -1):
            swap = self.randrange(index + 1)
            result[index], result[swap] = result[swap], result[index]
        return result

    def substream(self, label: int) -> "DeterministicRNG":
        """Derive an independent child stream without consuming this one."""
        return DeterministicRNG(_mix64(self._state ^ _mix64(int(label))))

    @property
    def state(self) -> int:
        return self._state

    @state.setter
    def state(self, value: int) -> None:
        self._state = int(value) & _MASK
