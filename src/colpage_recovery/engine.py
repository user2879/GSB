from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .codec import (
    apply_patch,
    content_digest,
    generate_patch,
    initial_content,
    patch_byte_count,
    patch_digest,
    version_digest,
    PageContent,
)
from .errors import BudgetExceededError
from .models import Bundle, PageSpec, WalEntry
from .scheduler import ScheduledTask, run_quota_schedule


@dataclass
class PatchWorkerResult:
    key: int
    patch_digest: str
    byte_count: int
    patches: dict[str, Any]
    thread_name: str


@dataclass
class PageVersionRecord:
    lsn: int
    tx_id: str
    page_id: str
    version: int
    row_start: int
    row_end: int
    seed: int
    commit_flag: bool
    persist_flag: bool
    visible: bool
    after_crash: bool
    byte_count: int
    byte_length: int
    patch_digest: str
    content_digest: str
    version_digest: str
    content_bytes: bytes = field(repr=False)


@dataclass
class CheckpointPage:
    page_id: str
    version: int
    lsn: int
    content_digest: str
    version_digest: str


@dataclass
class CheckpointPlan:
    checkpoint_no: int
    at_lsn: int
    pages: list[CheckpointPage]


@dataclass
class BatchReplay:
    round_no: int
    entries: list[WalEntry]
    records: list[PageVersionRecord]
    timeline: list[dict[str, Any]]
    slot_peak: int
    byte_consumed: int
    page_bytes: dict[str, int]
    checkpoint: CheckpointPlan | None = None


@dataclass
class ReplayResult:
    records: list[PageVersionRecord]
    batches: list[BatchReplay]
    checkpoints: list[CheckpointPlan]
    final_content: dict[str, bytes]
    final_versions: dict[str, int]
    committed_txids: set[str]
    invisible_versions: list[dict[str, Any]]
    crash_point: dict[str, Any]
    timeline: list[dict[str, Any]]

    def digest_map(self) -> dict[tuple[str, int], str]:
        return {(record.page_id, record.version): record.content_digest for record in self.records}

    def lsn_record_map(self) -> dict[int, PageVersionRecord]:
        return {record.lsn: record for record in self.records}


def decide_crash_point(entries: list[WalEntry]) -> dict[str, Any]:
    durable = [entry.lsn for entry in entries if entry.commit_flag and entry.persist_flag]
    crash_lsn = max(durable, default=0)
    committed = sorted({entry.tx_id for entry in entries if entry.commit_flag})
    uncommitted = sorted({entry.tx_id for entry in entries if not entry.commit_flag})
    committed_pending_persist = sorted(
        entry.lsn
        for entry in entries
        if entry.commit_flag and not entry.persist_flag and entry.lsn > crash_lsn
    )
    return {
        "crash_after_lsn": crash_lsn,
        "criterion": "greatest_lsn_with_commit_flag_and_persist_flag_true",
        "durable_marker_lsns": durable,
        "committed_txids": committed,
        "uncommitted_txids": uncommitted,
        "committed_redo_lsns_after_crash": committed_pending_persist,
        "invisible_rule": "page version is invisible when tx commit_flag is false",
    }


class ReplayEngine:
    def __init__(self, bundle: Bundle) -> None:
        self.bundle = bundle
        self.pages = {page.page_id: page for page in bundle.pages}
        self.entries = sorted(bundle.entries, key=lambda entry: entry.lsn)
        self.target_versions = self._build_target_versions()
        self.commit_by_tx = {
            entry.tx_id: entry.commit_flag for entry in self.entries
        }
        self.crash_point = decide_crash_point(self.entries)
        self.checkpoint_lsns = {
            self.entries[index].lsn
            for index in range(
                bundle.budget.checkpoint_interval - 1,
                len(self.entries),
                bundle.budget.checkpoint_interval,
            )
        }

    def _build_target_versions(self) -> dict[int, int]:
        counts: dict[str, int] = {}
        result: dict[int, int] = {}
        for entry in self.entries:
            counts[entry.page_id] = counts.get(entry.page_id, 0) + 1
            result[entry.lsn] = counts[entry.page_id]
        return result

    def initial_states(self) -> dict[str, PageContent]:
        return {page.page_id: initial_content(page) for page in self.bundle.pages}

    def replay_all(self, batch_size: int | None = None) -> ReplayResult:
        groups = self.bundle.batches(batch_size)
        return self.replay_groups(groups)

    def replay_groups(
        self,
        groups: list[list[WalEntry]],
        *,
        initial_states: dict[str, PageContent] | None = None,
        start_round: int = 1,
        initial_page_usage: dict[str, int] | None = None,
    ) -> ReplayResult:
        states = initial_states or self.initial_states()
        page_usage = dict(initial_page_usage or {page.page_id: 0 for page in self.bundle.pages})
        records: list[PageVersionRecord] = []
        batch_results: list[BatchReplay] = []
        checkpoints: list[CheckpointPlan] = []
        timeline: list[dict[str, Any]] = []

        for offset, group in enumerate(groups):
            round_no = start_round + offset
            ordered = sorted(group, key=lambda entry: entry.lsn)
            page_batch_bytes: dict[str, int] = {}
            tasks = []
            for entry in ordered:
                page = self.pages[entry.page_id]
                byte_count = patch_byte_count(page, entry)
                page_batch_bytes[entry.page_id] = (
                    page_batch_bytes.get(entry.page_id, 0) + byte_count
                )
                tasks.append(
                    ScheduledTask(
                        key=entry.lsn,
                        tx_id=entry.tx_id,
                        label=entry.page_id,
                        byte_count=byte_count,
                        context=(page, entry, self.target_versions[entry.lsn]),
                    )
                )

            for page_id, added_bytes in page_batch_bytes.items():
                limit = self.bundle.budget.page_byte_budget
                if page_usage.get(page_id, 0) + added_bytes > limit:
                    raise BudgetExceededError(
                        "single-page replay byte budget exceeded",
                        field="single_page_replay_byte_budget",
                        details={
                            "page_id": page_id,
                            "already_consumed": page_usage.get(page_id, 0),
                            "batch_bytes": added_bytes,
                            "budget": limit,
                        },
                    )

            schedule_result = run_quota_schedule(tasks, self._worker, self.bundle.budget)
            batch_records: list[PageVersionRecord] = []
            checkpoint: CheckpointPlan | None = None

            for entry in ordered:
                page = self.pages[entry.page_id]
                worker_result: PatchWorkerResult = schedule_result.outcomes[entry.lsn]
                state = states[entry.page_id]
                target_version = self.target_versions[entry.lsn]
                apply_patch(state, page, entry, target_version, worker_result.patches)
                encoded = state.encode(page)
                digest = content_digest(encoded)
                vdigest = version_digest(
                    entry.page_id, target_version, entry.lsn, digest
                )
                visible = self.commit_by_tx[entry.tx_id]
                record = PageVersionRecord(
                    lsn=entry.lsn,
                    tx_id=entry.tx_id,
                    page_id=entry.page_id,
                    version=target_version,
                    row_start=entry.row_start,
                    row_end=entry.row_end,
                    seed=entry.seed,
                    commit_flag=entry.commit_flag,
                    persist_flag=entry.persist_flag,
                    visible=visible,
                    after_crash=entry.lsn > self.crash_point["crash_after_lsn"],
                    byte_count=worker_result.byte_count,
                    byte_length=len(encoded),
                    patch_digest=worker_result.patch_digest,
                    content_digest=digest,
                    version_digest=vdigest,
                    content_bytes=encoded,
                )
                batch_records.append(record)
                records.append(record)
                page_usage[entry.page_id] = page_usage.get(entry.page_id, 0) + record.byte_count

                if entry.lsn in self.checkpoint_lsns:
                    checkpoint = self._make_checkpoint(entry.lsn, len(checkpoints) + 1, states)
                    checkpoints.append(checkpoint)

            batch_page_bytes = {
                page_id: sum(record.byte_count for record in batch_records if record.page_id == page_id)
                for page_id in {record.page_id for record in batch_records}
            }
            batch_result = BatchReplay(
                round_no=round_no,
                entries=ordered,
                records=batch_records,
                timeline=schedule_result.timeline_dicts(),
                slot_peak=schedule_result.slot_peak,
                byte_consumed=sum(record.byte_count for record in batch_records),
                page_bytes=batch_page_bytes,
                checkpoint=checkpoint,
            )
            batch_results.append(batch_result)
            timeline.extend(batch_result.timeline)

        final_content: dict[str, bytes] = {}
        final_versions: dict[str, int] = {}
        for page in self.bundle.pages:
            state = states[page.page_id]
            encoded = state.encode(page)
            final_content[page.page_id] = encoded
            final_versions[page.page_id] = state.version

        invisible = [
            {
                "page_id": record.page_id,
                "version": record.version,
                "lsn": record.lsn,
                "tx_id": record.tx_id,
                "after_crash": record.after_crash,
                "content_digest": record.content_digest,
            }
            for record in records
            if not record.visible
        ]
        return ReplayResult(
            records=records,
            batches=batch_results,
            checkpoints=checkpoints,
            final_content=final_content,
            final_versions=final_versions,
            committed_txids={tx for tx, committed in self.commit_by_tx.items() if committed},
            invisible_versions=invisible,
            crash_point=self.crash_point,
            timeline=sorted(timeline, key=lambda item: (item["lsn"], item["page_id"])),
        )

    def _worker(self, task: ScheduledTask) -> PatchWorkerResult:
        page, entry, target_version = task.context
        patches = generate_patch(page, entry, target_version)
        digest = patch_digest(page, entry, target_version, patches)
        delay_ms = self.bundle.budget.work_delay_ms
        if delay_ms > 0:
            time.sleep(delay_ms / 1000.0)
        import threading

        return PatchWorkerResult(
            key=entry.lsn,
            patch_digest=digest,
            byte_count=task.byte_count,
            patches=patches,
            thread_name=threading.current_thread().name,
        )

    def _make_checkpoint(
        self,
        at_lsn: int,
        checkpoint_no: int,
        states: dict[str, PageContent],
    ) -> CheckpointPlan:
        checkpoint_number = sorted(self.checkpoint_lsns).index(at_lsn) + 1
        pages: list[CheckpointPage] = []
        for page in self.bundle.pages:
            state = states[page.page_id]
            encoded = state.encode(page)
            digest = content_digest(encoded)
            vdigest = version_digest(page.page_id, state.version, state.lsn, digest)
            pages.append(
                CheckpointPage(
                    page_id=page.page_id,
                    version=state.version,
                    lsn=state.lsn,
                    content_digest=digest,
                    version_digest=vdigest,
                )
            )
        return CheckpointPlan(checkpoint_no=checkpoint_number, at_lsn=at_lsn, pages=pages)
