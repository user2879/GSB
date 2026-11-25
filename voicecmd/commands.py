"""命令汇总：将逐窗后验归并为带引用窗集合的命令结论。"""

from dataclasses import dataclass

from .classifier import WindowPosterior
from .config import PipelineConfig
from .fingerprint import PREFIX_COMMAND, cache_key, content_fingerprint


@dataclass(frozen=True)
class CommandConclusion:
    conclusion_id: int
    code: int
    word: str
    window_indices: tuple[int, ...]
    sample_span: tuple[int, int]
    boundary: bool
    cache_key: str
    posterior_keys: tuple[str, ...]


def summarize_commands(
    posteriors: list[WindowPosterior],
    window_spans: dict[int, tuple[int, int]],
    cfg: PipelineConfig,
) -> list[CommandConclusion]:
    ordered = sorted(posteriors, key=lambda item: item.window_index)
    conclusions: list[CommandConclusion] = []
    group: list[WindowPosterior] = []

    def flush(group_items: list[WindowPosterior]) -> None:
        if not group_items:
            return
        indices = tuple(item.window_index for item in group_items)
        code = group_items[0].predicted_code
        boundary = any(item.boundary for item in group_items)
        span_start = min(window_spans[index][0] for index in indices)
        span_end = max(window_spans[index][1] for index in indices)
        posterior_keys = tuple(item.cache_key for item in group_items)
        upstream = content_fingerprint(
            sorted((item.window_index, item.cache_key) for item in group_items)
        )
        param_fp = content_fingerprint(
            {"vocab": cfg.vocab_fingerprint(), "stage": "command_summary"}
        )
        key = cache_key(PREFIX_COMMAND, upstream, param_fp)
        conclusions.append(
            CommandConclusion(
                conclusion_id=len(conclusions),
                code=code,
                word=cfg.vocab[code],
                window_indices=indices,
                sample_span=(span_start, span_end),
                boundary=boundary,
                cache_key=key,
                posterior_keys=posterior_keys,
            )
        )

    for posterior in ordered:
        if group and posterior.predicted_code != group[-1].predicted_code:
            flush(group)
            group = []
        group.append(posterior)
    flush(group)
    return conclusions
