"""Budgeted parallel preparation with canonical single-threaded commit."""

from __future__ import annotations

import os
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Any, Callable

from .content import deterministic_page_bytes, digest_bytes
from .errors import BusinessError, monotonic_ms
from .models import PageDecl, PreparedEntry, ReplayBudget, ReplayTaskSpec
from .storage import Storage


@dataclass(frozen=True)
class PreparedEffect:
    entry: PreparedEntry
    data: bytes
    content_path: str


def build_replay_tasks(
    run_id: str,
    entries: list[PreparedEntry],
    pages: list[PageDecl],
    budget: ReplayBudget,
) -> list[ReplayTaskSpec]:
    """Partition by transaction and page, then by byte/checkpoint limits.

    Splitting by page lets different pages in the same transaction execute in
    parallel.  The per-transaction quota is meaningful because several such
    tasks can coexist.  Actual page writes are nevertheless committed by the
    canonical LSN sequencer below.
    """

    page_by_id = {page.page_id: page for page in pages}
    transaction_groups: list[list[PreparedEntry]] = []
    current: list[PreparedEntry] = []
    for entry in entries:
        if current and current[-1].txid != entry.txid:
            transaction_groups.append(current)
            current = []
        current.append(entry)
    if current:
        transaction_groups.append(current)

    specs: list[ReplayTaskSpec] = []
    for tx_entries in transaction_groups:
        by_page: dict[str, list[PreparedEntry]] = {}
        page_order: list[str] = []
        for entry in tx_entries:
            if entry.page_id not in by_page:
                by_page[entry.page_id] = []
                page_order.append(entry.page_id)
            by_page[entry.page_id].append(entry)
        for page_id in sorted(page_order):
            page = page_by_id[page_id]
            max_entries = max(1, budget.page_byte_budget // page.size_bytes)
            max_entries = min(max_entries, budget.checkpoint_interval)
            page_entries = by_page[page_id]
            for chunk_index, offset in enumerate(range(0, len(page_entries), max_entries)):
                chunk = tuple(page_entries[offset : offset + max_entries])
                task_key = f"{run_id}:{tx_entries[0].txid}:{page_id}:{chunk_index:03d}"
                specs.append(
                    ReplayTaskSpec(
                        task_key=task_key,
                        txid=tx_entries[0].txid,
                        page_id=page_id,
                        chunk_index=chunk_index,
                        entries=chunk,
                        planned_bytes=len(chunk) * page.size_bytes,
                    )
                )
    return sorted(specs, key=lambda task: (task.lsn_start, task.task_key))


class ReplayEngine:
    def __init__(self, storage: Storage):
        self.storage = storage

    def replay(
        self,
        *,
        delay_ms: int = 0,
        crash_after_batches: int | None = None,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        if not self.storage.is_initialized():
            raise BusinessError(
                "E_STATE",
                "workspace is not initialized; load a sample first",
                {"workspace": str(self.storage.workspace)},
            )
        if crash_after_batches is not None and crash_after_batches < 1:
            raise BusinessError(
                "E_STATE",
                "crash_after_batches must be at least 1",
                {"field": "crash_after_batches", "value": crash_after_batches},
            )
        recovered = self.storage.recover_orphan_tasks()
        cursor = self.storage.ledger_cursor()
        entries = self.storage.prepared_entries(cursor)
        pages = self.storage.page_declarations()
        budget, budget_source = self.storage.ledger_budget()
        run_id = self.storage.next_run_id()
        self.storage.start_run(run_id, cursor, budget)
        if not entries:
            self.storage.complete_empty_run(run_id, cursor)
            return {
                "ok": True,
                "run_id": run_id,
                "cursor": cursor,
                "recovered_orphan_tasks": recovered,
                "budget_source": budget_source,
                "replayed_pages": 0,
                "batches": [],
                "note": "no entries remained after the ledger cursor",
            }

        tasks = build_replay_tasks(run_id, entries, pages, budget)
        self.storage.insert_tasks(
            [
                {
                    "task_key": task.task_key,
                    "run_id": run_id,
                    "txid": task.txid,
                    "page_id": task.page_id,
                    "chunk_index": task.chunk_index,
                    "lsn_start": task.lsn_start,
                    "lsn_end": task.lsn_end,
                    "entry_count": len(task.entries),
                    "planned_bytes": task.planned_bytes,
                    "detail": {
                        "lsn_list": [entry.lsn for entry in task.entries],
                        "planned_bytes": task.planned_bytes,
                    },
                }
                for task in tasks
            ]
        )
        try:
            result = self._execute(
                run_id=run_id,
                cursor=cursor,
                entries=entries,
                tasks=tasks,
                budget=budget,
                delay_ms=max(0, delay_ms),
                crash_after_batches=crash_after_batches,
                recovered_orphan_tasks=recovered,
                budget_source=budget_source,
                progress=progress,
            )
            return result
        except BusinessError as error:
            try:
                self.storage.fail_run(run_id, error)
            except Exception:
                pass
            raise

    def _prepare_one(
        self,
        task: ReplayTaskSpec,
        pages_by_id: dict[str, PageDecl],
        delay_ms: int,
    ) -> list[PreparedEffect]:
        if delay_ms:
            time.sleep(delay_ms / 1000.0)
        effects: list[PreparedEffect] = []
        for entry in task.entries:
            page = pages_by_id[entry.page_id]
            data = deterministic_page_bytes(
                entry.page_id, page.size_bytes, entry.target_version, entry.lsn
            )
            actual_digest = digest_bytes(data)
            if actual_digest != entry.generated_digest or actual_digest != entry.content_digest:
                raise BusinessError(
                    "E_REPLAY",
                    "worker produced a digest inconsistent with the validated WAL",
                    {
                        "lsn": entry.lsn,
                        "page_id": entry.page_id,
                        "field": "content_digest",
                        "expected": entry.generated_digest,
                        "actual": actual_digest,
                    },
                )
            effects.append(
                PreparedEffect(
                    entry=entry,
                    data=data,
                    content_path=self.storage.content_path_for(entry.page_id, entry.target_version),
                )
            )
        return effects

    def _execute(
        self,
        *,
        run_id: str,
        cursor: int,
        entries: list[PreparedEntry],
        tasks: list[ReplayTaskSpec],
        budget: ReplayBudget,
        delay_ms: int,
        crash_after_batches: int | None,
        recovered_orphan_tasks: int,
        budget_source: str,
        progress: Callable[[dict[str, Any]], None] | None,
    ) -> dict[str, Any]:
        pages_by_id = {page.page_id: page for page in self.storage.page_declarations()}
        task_by_key = {task.task_key: task for task in tasks}
        task_total = {task.task_key: len(task.entries) for task in tasks}
        task_done = {task.task_key: 0 for task in tasks}
        page_size_by_id = {page.page_id: page.size_bytes for page in pages_by_id.values()}
        final_lsn = entries[-1].lsn

        pending = list(tasks)
        active: dict[Future[list[PreparedEffect]], str] = {}
        active_by_tx: dict[str, int] = {}
        ready_effects: dict[int, tuple[str, PreparedEffect]] = {}
        wait_started: dict[str, int] = {}
        task_batch_ids: dict[str, int] = {}
        batches: list[dict[str, Any]] = []
        next_lsn = cursor + 1
        batch_effects: list[PreparedEffect] = []
        batch_owners: set[str] = set()
        batch_size = max(1, budget.checkpoint_interval)

        executor = ThreadPoolExecutor(max_workers=budget.global_slots, thread_name_prefix="wal-replay")
        try:
            while next_lsn <= final_lsn or active or pending:
                submitted_any = False
                for task in list(pending):
                    if len(active) >= budget.global_slots:
                        if task.task_key not in wait_started:
                            wait_started[task.task_key] = monotonic_ms()
                            self.storage.task_waiting(task.task_key, wait_started[task.task_key])
                        continue
                    tx_active = active_by_tx.get(task.txid, 0)
                    if tx_active >= budget.per_tx_quota:
                        if task.task_key not in wait_started:
                            wait_started[task.task_key] = monotonic_ms()
                            self.storage.task_waiting(task.task_key, wait_started[task.task_key])
                        continue
                    future = executor.submit(self._prepare_one, task, pages_by_id, delay_ms)
                    active[future] = task.task_key
                    active_by_tx[task.txid] = tx_active + 1
                    pending.remove(task)
                    submitted_any = True
                    self.storage.task_started(
                        task.task_key,
                        wait_started.get(task.task_key),
                        monotonic_ms(),
                    )

                done_futures = [future for future in active if future.done()]
                for future in done_futures:
                    task_key = active.pop(future)
                    spec = task_by_key[task_key]
                    active_by_tx[spec.txid] -= 1
                    prepared = future.result()
                    self.storage.task_staged(task_key)
                    for effect in prepared:
                        ready_effects[effect.entry.lsn] = (task_key, effect)

                made_progress = False
                while next_lsn in ready_effects:
                    task_key, effect = ready_effects.pop(next_lsn)
                    batch_effects.append(effect)
                    batch_owners.add(task_key)
                    task_done[task_key] += 1
                    next_lsn += 1
                    made_progress = True
                    if len(batch_effects) >= batch_size or next_lsn > final_lsn:
                        is_final = next_lsn > final_lsn
                        task_updates = []
                        for owner in sorted(batch_owners):
                            spec = task_by_key[owner]
                            consumed = task_done[owner] * page_size_by_id[spec.page_id]
                            state = "committed" if task_done[owner] == task_total[owner] else "partial"
                            task_updates.append(
                                {
                                    "task_id": owner,
                                    "state": state,
                                    "consumed_bytes": consumed,
                                    "entry_count_committed": task_done[owner],
                                }
                            )
                        committed = self.storage.commit_batch(
                            run_id,
                            batch_effects,
                            task_updates,
                            is_final=is_final,
                            checkpoint_interval=budget.checkpoint_interval,
                        )
                        for update in task_updates:
                            task_batch_ids[update["task_id"]] = committed["batch_id"]
                        batches.append(committed)
                        if progress:
                            progress({"kind": "batch_committed", **committed})
                        batch_effects = []
                        batch_owners = set()
                        if crash_after_batches is not None and len(batches) >= crash_after_batches:
                            # Deliberate hard stop: no Python cleanup, no exit report.
                            os._exit(91)

                if next_lsn > final_lsn:
                    break
                if not made_progress and not done_futures and not submitted_any:
                    if not active:
                        raise BusinessError(
                            "E_REPLAY",
                            "scheduler stalled before reaching the final LSN",
                            {
                                "expected_lsn": next_lsn,
                                "final_lsn": final_lsn,
                                "pending_tasks": len(pending),
                            },
                        )
                    wait(active.keys(), return_when=FIRST_COMPLETED)
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

        return {
            "ok": True,
            "run_id": run_id,
            "cursor_start": cursor,
            "cursor_end": final_lsn,
            "recovered_orphan_tasks": recovered_orphan_tasks,
            "budget_source": budget_source,
            "task_count": len(tasks),
            "batch_count": len(batches),
            "batch_size": batch_size,
            "batches": batches,
            "budget": budget.as_dict(),
        }
