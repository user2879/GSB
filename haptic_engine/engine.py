"""End-to-end synthesis orchestration including recovery and export."""

from __future__ import annotations

import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .envelope import level_to_intensity, synthesize_device_levels
from .errors import E_DIGEST, HapticError
from .models import Device, Manifest
from .packing import Block, build_blocks, validate_device_packing
from .scheduler import ScheduledBlock, Scheduler
from .state import StateStore, atomic_write, block_digest
from .sync import build_anchor_rows, evaluate_anchors

SIGNAL_EXIT_CODES = {signal.SIGINT: 130, signal.SIGTERM: 143}


class SignalController:
    """Handled termination signals are honored only at frame-block boundaries."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self.pending_signal: signal.Signals | None = None

    def install(self) -> None:
        for signum in SIGNAL_EXIT_CODES:
            signal.signal(signum, self._handler)

    def _handler(self, signum: int, _frame: object) -> None:
        self.pending_signal = signal.Signals(signum)
        self._event.set()

    def requested(self) -> bool:
        return self._event.is_set()


@dataclass
class DeviceResult:
    device: Device
    levels: object
    blocks: list[Block]
    total_samples: int
    overflow_count: int
    recomputed_blocks: int
    digest_failures: int
    digest_events: list
    pending_repair: set


class HapticEngine:
    def __init__(
        self,
        manifest: Manifest,
        serial: bool = False,
        block_delay: float = 0.0,
        signals: SignalController | None = None,
    ) -> None:
        self.manifest = manifest
        self.serial = serial
        self.block_delay = block_delay
        self.signals = signals or SignalController()
        self.store = StateStore(manifest)
        self.device_map = manifest.device_map()
        self.plans: dict[str, list[ScheduledBlock]] = {}
        self._results: dict[str, DeviceResult] = {}
        self._locks = {device.device_id: threading.Lock() for device in manifest.devices}
        self.resumed_after_interrupt = False
        self.peak_inflight = 0
        self.waited_events = 0

    def run(self) -> dict:
        self.store.initialize()
        self.resumed_after_interrupt = self.store.has_pending_resume()
        if self.resumed_after_interrupt:
            self.store.clear_pending_resume()
        self._prepare_plans()
        self._run_scheduler()
        self._final_verify_and_repair()
        block_streams = {device_id: result.blocks for device_id, result in self._results.items()}
        sync_rows, sync_violations = self._safe_evaluate_sync(block_streams)
        report = self._export(block_streams, sync_rows, sync_violations)
        return report

    def handle_interrupt_and_exit(self) -> int:
        assert self.signals.pending_signal is not None
        signum = self.signals.pending_signal
        self.store.record_interrupt(signum.name, int(signum))
        return SIGNAL_EXIT_CODES[signum]

    def _prepare_plans(self) -> None:
        for device in self.manifest.devices:
            levels = synthesize_device_levels(device, list(self.manifest.events))
            blocks, overflow_count = build_blocks(levels, device)
            validate_device_packing(blocks, levels, device)
            stored: dict[int, Block] = {}
            digest_failures = 0
            digest_events: list[dict] = []
            pending_repair: set[int] = set()
            for block in blocks:
                path = self.store.block_path(device.device_id, block.block_index)
                snapshot_existed = path.exists()
                verified = self.store.verify_or_discard(device.device_id, block.block_index)
                if verified is not None:
                    stored[block.block_index] = verified
                elif snapshot_existed:
                    digest_failures += 1
                    pending_repair.add(block.block_index)
                    digest_events.append(
                        {
                            "code": E_DIGEST,
                            "device_id": device.device_id,
                            "block_index": block.block_index,
                            "action": "recompute",
                        }
                    )
            self._results[device.device_id] = DeviceResult(
                device=device,
                levels=levels,
                blocks=blocks,
                total_samples=int(levels.size),
                overflow_count=overflow_count,
                recomputed_blocks=0,
                digest_failures=digest_failures,
                digest_events=digest_events,
                pending_repair=pending_repair,
            )
            self.plans[device.device_id] = [
                ScheduledBlock(device.device_id, block.block_index, block.command_count)
                for block in blocks
                if block.block_index not in stored
            ]

    def _run_scheduler(self) -> None:
        scheduler = Scheduler(
            [device.device_id for device in self.manifest.devices],
            self.plans,
            concurrency=1 if self.serial else self.manifest.global_config.concurrency,
            inflight_limit=self.manifest.global_config.inflight_limit,
            signal_flag=self.signals.requested,
        )
        worker = self._synthesize_block
        if self.serial or self.manifest.global_config.concurrency == 1:
            scheduler.run_serial(worker)
        else:
            scheduler.run_parallel(worker)
        self.peak_inflight = scheduler.peak_inflight
        self.waited_events = scheduler.waited_events

    def _synthesize_block(self, scheduled: ScheduledBlock) -> None:
        if self.block_delay:
            time.sleep(self.block_delay)
        result = self._results[scheduled.device_id]
        block = next(item for item in result.blocks if item.block_index == scheduled.block_index)
        with self._locks[scheduled.device_id]:
            self.store.write_block(scheduled.device_id, block, result.total_samples)
        result.recomputed_blocks += 1

    def _final_verify_and_repair(self) -> None:
        """Re-check every persisted block; rebuild invalid ones and re-verify."""
        for device in self.manifest.devices:
            result = self._results[device.device_id]
            repaired_blocks: list[Block] = []
            for expected in result.blocks:
                loaded = self.store.load_block(device.device_id, expected.block_index)
                if loaded is None or block_digest(loaded["block"]) != block_digest(expected):
                    if expected.block_index not in result.pending_repair:
                        result.digest_failures += 1
                        result.digest_events.append(
                            {
                                "code": E_DIGEST,
                                "device_id": device.device_id,
                                "block_index": expected.block_index,
                                "action": "recompute",
                            }
                        )
                    path = self.store.block_path(device.device_id, expected.block_index)
                    if path.exists():
                        path.unlink()
                    self.store.write_block(device.device_id, expected, result.total_samples)
                    repaired = self.store.verify_or_discard(device.device_id, expected.block_index)
                    if repaired is None:
                        raise HapticError(
                            E_DIGEST,
                            "repaired frame block still fails digest verification",
                            device_id=device.device_id,
                            block_index=expected.block_index,
                        )
                    repaired_blocks.append(repaired)
                    result.pending_repair.discard(expected.block_index)
                else:
                    repaired_blocks.append(loaded["block"])
            repaired_blocks.sort(key=lambda item: item.block_index)
            validate_device_packing(repaired_blocks, result.levels, device)
            result.blocks = repaired_blocks

    def _safe_evaluate_sync(self, block_streams: dict[str, list[Block]]) -> tuple[list[dict], list[dict]]:
        rows = build_anchor_rows(self.manifest, self.device_map, block_streams)
        try:
            return evaluate_anchors(self.manifest, self.device_map, block_streams)
        except HapticError as exc:
            if exc.code != "E_SYNC":
                raise
            # Valid command streams are still exported alongside diagnostics.
            return rows, exc.details["violations"]

    def _export(
        self,
        block_streams: dict[str, list[Block]],
        sync_rows: list[dict],
        sync_violations: list[dict],
    ) -> dict:
        output_path = Path(self.manifest.output_file)
        streams: dict[str, list[dict]] = {}
        device_reports = {}
        for device_id in sorted(block_streams):
            result = self._results[device_id]
            commands: list[dict] = []
            for block in block_streams[device_id]:
                for command in block.commands:
                    commands.append(
                        {
                            **command.to_dict(),
                            "intensity": level_to_intensity(command.level, result.device),
                        }
                    )
            streams[device_id] = commands
            device_reports[device_id] = {
                "command_count": len(commands),
                "block_count": len(block_streams[device_id]),
                "overflow_count": result.overflow_count,
                "total_samples": result.total_samples,
                "recomputed_blocks": result.recomputed_blocks,
                "digest_failures": result.digest_failures,
                "digest_events": result.digest_events,
            }
        payload = {
            "format": "haptic-commands-v1",
            "version_id": self.store.version_id,
            "streams": streams,
        }
        atomic_write(output_path, payload)
        report = {
            "ok": not sync_violations,
            "version_id": self.store.version_id,
            "output_file": str(output_path),
            "state_dir": self.manifest.state_dir,
            "interrupt_count": self.store.read_interrupt_count(),
            "peak_inflight_commands": getattr(self, "peak_inflight", 0),
            "budget_wait_events": getattr(self, "waited_events", 0),
            "inflight_limit": self.manifest.global_config.inflight_limit,
            "concurrency": self.manifest.global_config.concurrency,
            "devices": device_reports,
            "sync": {"anchors": sync_rows, "violations": sync_violations},
        }
        atomic_write(output_path.with_name(f"{output_path.stem}.report.json"), report)
        return report
