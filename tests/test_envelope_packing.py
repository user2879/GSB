import tempfile
import unittest
from pathlib import Path

import numpy as np

from _helpers import parsed

from haptic_engine.envelope import (
    event_sample_span,
    ms_to_samples,
    synthesize_device_levels,
)
from haptic_engine.packing import build_blocks, expand_blocks, validate_device_packing


class EnvelopeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.manifest = parsed(Path(self.tmp.name))
        self.devices = {item.device_id: item for item in self.manifest.devices}

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_ms_to_sample_mapping_and_alignment(self):
        self.assertEqual(ms_to_samples(1, 1000), 1)
        self.assertEqual(ms_to_samples(1, 800), 1)
        self.assertEqual(ms_to_samples(10, 800), 8)
        self.assertEqual(ms_to_samples(1, 333), 0)
        self.assertEqual(ms_to_samples(2, 333), 1)

    def test_event_span_half_open_and_order_independent(self):
        device = self.devices["d-a"]
        event = self.manifest.events[0]
        span_a = event_sample_span(event, device)
        reversed_events = list(reversed(self.manifest.events))
        event_reversed = next(item for item in reversed_events if item.event_id == event.event_id)
        span_b = event_sample_span(event_reversed, device)
        self.assertEqual(span_a, span_b)
        self.assertEqual(span_a, (0, 100))

    def test_quantization_bounds_and_determinism(self):
        device = self.devices["d-a"]
        levels_a = synthesize_device_levels(device, list(self.manifest.events))
        levels_b = synthesize_device_levels(device, list(reversed(self.manifest.events)))
        np.testing.assert_array_equal(levels_a, levels_b)
        self.assertGreaterEqual(levels_a.min(), 0)
        self.assertLessEqual(levels_a.max(), device.levels - 1)

    def test_multi_device_alignment_within_one_sample(self):
        time_ms = 37
        mapped = {
            device.device_id: ms_to_samples(time_ms, device.sample_rate_hz)
            for device in self.manifest.devices
        }
        for device in self.manifest.devices:
            exact = time_ms * device.sample_rate_hz / 1000.0
            self.assertLessEqual(abs(mapped[device.device_id] - exact), 0.5)


class PackingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.manifest = parsed(Path(self.tmp.name))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_packing_equivalence_and_frame_cap(self):
        for device in self.manifest.devices:
            levels = synthesize_device_levels(device, list(self.manifest.events))
            blocks, overflow = build_blocks(levels, device)
            validate_device_packing(blocks, levels, device)
            for block in blocks:
                self.assertLessEqual(block.command_count, device.max_commands_per_frame)
            expanded = expand_blocks(blocks, levels.size)
            np.testing.assert_array_equal(expanded, levels)

    def test_overflow_splits_commands_without_timing_change(self):
        device = self.manifest.devices[0]
        levels = synthesize_device_levels(device, list(self.manifest.events))
        object.__setattr__(device, "max_commands_per_frame", 1)
        blocks, overflow = build_blocks(levels, device)
        self.assertGreater(overflow, 0)
        self.assertTrue(any(block.is_overflow for block in blocks))
        for block in blocks:
            self.assertLessEqual(block.command_count, 1)
        np.testing.assert_array_equal(expand_blocks(blocks, levels.size), levels)
        self.assertEqual([block.block_index for block in blocks], list(range(len(blocks))))

    def test_commands_carry_absolute_durations(self):
        device = self.manifest.devices[1]
        levels = synthesize_device_levels(device, list(self.manifest.events))
        blocks, _ = build_blocks(levels, device)
        total_duration = sum(
            command.duration_samples
            for block in blocks
            for command in block.commands
        )
        self.assertEqual(total_duration, levels.size)


if __name__ == "__main__":
    unittest.main()
