from __future__ import annotations

from decimal import Decimal

from .models import (
    CoverageEvent,
    Document,
    Sentence,
    SentenceScore,
    SelectionResult,
    WorkbenchConfig,
)


def select_summary(
    document: Document,
    sentences: list[Sentence],
    scores: dict[int, SentenceScore],
    config: WorkbenchConfig,
    *,
    reference_path: bool = False,
) -> SelectionResult:
    if reference_path:
        ordered = sorted(
            sentences,
            key=lambda sentence: (
                -scores[sentence.sentence_index].reference,
                sentence.start,
                sentence.sentence_index,
            ),
        )
    else:
        ordered = sorted(
            sentences,
            key=lambda sentence: (
                -scores[sentence.sentence_index].fixed_integer,
                sentence.start,
                sentence.sentence_index,
            ),
        )

    selectable = ordered[: config.max_summary_sentences]
    selected: list[int] = []
    covered: set[str] = set()
    events: list[CoverageEvent] = []
    total_entities = len(document.entities)
    threshold = config.entity_coverage_min
    satisfied = False
    for sentence in selectable:
        selected.append(sentence.sentence_index)
        covered.update(sentence.entity_ids)
        coverage = Decimal(len(covered)) / Decimal(total_entities)
        if coverage == threshold:
            events.append(
                CoverageEvent(
                    sentence_index=sentence.sentence_index,
                    covered_entities=len(covered),
                    total_entities=total_entities,
                    coverage=coverage,
                    threshold=threshold,
                    rule="coverage_equal_to_threshold_stop",
                )
            )
        if coverage >= threshold:
            satisfied = True
            break

    return SelectionResult(
        selected_indices=tuple(selected),
        ordered_indices=tuple(sentence.sentence_index for sentence in ordered),
        covered_entity_ids=tuple(sorted(covered)),
        coverage_events=tuple(events),
        coverage_satisfied=satisfied,
    )
