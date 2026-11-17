"""ThreadPoolExecutor runner with deterministic ordering and isolation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from .errors import PipelineError, StageFailure
from .models import stable_sort_key


def run_entity_tasks(
    tasks: list[tuple[str, str, callable]],
    max_workers: int,
) -> tuple[list, list[StageFailure]]:
    """Run ``(entity_id, stage, callable)`` tasks concurrently.

    Every task carries the stable key ``(entity_id, stage)``. Results and
    failures are returned sorted by that key, so the final ordering is identical
    to a single-threaded run over entities sorted by id. A raising task records
    its business code and never aborts the remaining entities.
    """
    ordered = sorted(tasks, key=lambda item: stable_sort_key(item[0], item[1]))
    failures: list[StageFailure] = []
    indexed_results: list[tuple[tuple, object]] = []

    def _invoke(entity_id: str, stage: str, func):
        try:
            return entity_id, stage, func(), None
        except PipelineError as exc:
            return entity_id, stage, None, exc
        except Exception as exc:  # defensive: keep per-entity isolation
            wrapped = PipelineError(str(exc))
            wrapped.code = "E_PIPELINE_ERROR"
            wrapped.entity_id = entity_id
            wrapped.stage = stage
            return entity_id, stage, None, wrapped

    if max_workers <= 1:
        for entity_id, stage, func in ordered:
            entity_id, stage, result, error = _invoke(entity_id, stage, func)
            if error is not None:
                failures.append(
                    StageFailure(
                        entity_id=entity_id,
                        stage=stage,
                        code=error.code,
                        message=str(error),
                        sort_key=stable_sort_key(entity_id, stage),
                    )
                )
            else:
                indexed_results.append(
                    (stable_sort_key(entity_id, stage), result)
                )
        return [item[1] for item in indexed_results], failures

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(_invoke, entity_id, stage, func)
            for entity_id, stage, func in ordered
        ]
        for future in futures:
            entity_id, stage, result, error = future.result()
            if error is not None:
                failures.append(
                    StageFailure(
                        entity_id=entity_id,
                        stage=stage,
                        code=error.code,
                        message=str(error),
                        sort_key=stable_sort_key(entity_id, stage),
                    )
                )
            else:
                indexed_results.append(
                    (stable_sort_key(entity_id, stage), result)
                )

    indexed_results.sort(key=lambda item: item[0])
    failures.sort(key=lambda item: item.sort_key)
    return [item[1] for item in indexed_results], failures
