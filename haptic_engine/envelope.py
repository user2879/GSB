"""Device sample-domain envelope synthesis and deterministic quantization."""

from __future__ import annotations

import numpy as np

from .models import Device, Event


def ms_to_samples(time_ms: int, sample_rate_hz: int) -> int:
    """Convert a millisecond offset to an integer sample index.

    Rounding is deterministic (round half away from zero; all inputs here are
    non-negative) so the same declaration and device parameters always produce
    the same sample indices regardless of synthesis order.
    """
    exact = time_ms * sample_rate_hz / 1000.0
    return int(np.floor(exact + 0.5))


def event_sample_span(event: Event, device: Device) -> tuple[int, int]:
    """Return the half-open ``[start, end)`` integer sample interval of an event."""
    start = ms_to_samples(event.start_ms, device.sample_rate_hz)
    end = ms_to_samples(event.end_ms, device.sample_rate_hz)
    return start, max(end, start + 1)


def quantize_intensity(intensity: np.ndarray, device: Device) -> np.ndarray:
    """Map continuous intensities to integer levels in ``[0, levels-1]``.

    Uses round-half-up instead of NumPy's banker's rounding, keeping the result
    independent of platform float formatting.
    """
    clipped = np.clip(intensity, device.min_intensity, device.max_intensity)
    normalized = (clipped - device.min_intensity) / (device.max_intensity - device.min_intensity)
    scaled = normalized * (device.levels - 1)
    quantized = np.floor(scaled + 0.5).astype(np.int64)
    return np.clip(quantized, 0, device.levels - 1)


def level_to_intensity(level: int, device: Device) -> float:
    return device.min_intensity + (level / (device.levels - 1)) * (
        device.max_intensity - device.min_intensity
    )


def event_sample_intensities(event: Event, device: Device) -> tuple[int, int, np.ndarray]:
    """Piecewise-linear interpolation of one event on the device sample grid."""
    start, end = event_sample_span(event, device)
    sample_count = end - start
    sample_times = (np.arange(start, end, dtype=np.float64) * 1000.0) / device.sample_rate_hz
    curve_times = np.array([point.time_ms for point in event.curve], dtype=np.float64)
    curve_values = np.array([point.intensity for point in event.curve], dtype=np.float64)
    # np.interp clamps outside the curve span; parser guarantees full coverage.
    intensities = np.interp(sample_times, curve_times, curve_values)
    return start, end, intensities


def synthesize_device_levels(device: Device, events: list[Event]) -> np.ndarray:
    """Build the full per-sample quantized level array for one device.

    Overlapping events are summed as continuous intensities and clamped to the
    device range before quantization, which is independent of event ordering.
    Idle samples stay at level 0 (the device minimum intensity).
    """
    device_events = [event for event in events if event.target_device == device.device_id]
    total_samples = 1
    spans: list[tuple[int, int, np.ndarray]] = []
    for event in device_events:
        start, end, intensities = event_sample_intensities(event, device)
        spans.append((start, end, intensities))
        total_samples = max(total_samples, end)

    continuous = np.full(total_samples, device.min_intensity, dtype=np.float64)
    for start, end, intensities in spans:
        continuous[start:end] += intensities - device.min_intensity
    return quantize_intensity(continuous, device)
