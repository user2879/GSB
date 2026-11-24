from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Iterable


def stable_batch_key(batch_id: str, stage: str) -> str:
    return f"stage={stage}|batch={batch_id}"


def map_batch_tasks(
    *,
    stage: str,
    tasks: Iterable[dict[str, Any]],
    worker: Callable[[dict[str, Any]], Any],
    parallel: bool = True,
    max_workers: int = 4,
) -> list[dict[str, Any]]:
    """Run batch tasks and return envelopes ordered by derived stable keys."""

    task_list = list(tasks)
    prepared = []
    for task in task_list:
        batch_id = str(task["batch_id"])
        sort_key = task.get("sort_key") or stable_batch_key(batch_id, stage)
        prepared.append((sort_key, task))

    results: dict[str, Any] = {}
    errors: dict[str, BaseException] = {}
    if parallel and len(prepared) > 1:
        with ThreadPoolExecutor(max_workers=min(max_workers, len(prepared))) as executor:
            future_to_key = {
                executor.submit(worker, task): sort_key for sort_key, task in prepared
            }
            for future in as_completed(future_to_key):
                sort_key = future_to_key[future]
                try:
                    results[sort_key] = future.result()
                except BaseException as exc:  # deterministic re-raise after key ordering
                    errors[sort_key] = exc
    else:
        for sort_key, task in prepared:
            try:
                results[sort_key] = worker(task)
            except BaseException as exc:
                errors[sort_key] = exc

    ordered: list[dict[str, Any]] = []
    for sort_key, _ in sorted(prepared, key=lambda item: item[0]):
        if sort_key in errors:
            raise errors[sort_key]
        value = results[sort_key]
        ordered.append({"sort_key": sort_key, "value": value})
    return ordered
