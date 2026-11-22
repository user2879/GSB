"""未登录词连续单字片段处理阶段。

两种规则都会先按固定上限做硬分段，再在每个硬分段内部按规则切分，
因此规则切换会真实改变候选边界（等均切分 vs 固定最大词长切分）。
"""
from __future__ import annotations

from .fingerprint import combine, fingerprint
from .models import Config, Token


def _hard_chunks(start: int, text: str, max_length: int) -> list[tuple[str, int, int, str]]:
    chunks: list[tuple[str, int, int, str]] = []
    for offset in range(0, len(text), max_length):
        piece = text[offset : offset + max_length]
        chunks.append(
            (
                piece,
                start + offset,
                start + offset + len(piece),
                f"hard_cap:{max_length}",
            )
        )
    return chunks


def _equal_pieces(piece: str) -> list[str]:
    n = len(piece)
    count = max(1, round(n ** 0.5))
    base, extra = divmod(n, count)
    if base == 0:
        return [piece]
    sizes = [base + (1 if i < extra else 0) for i in range(count)]
    out: list[str] = []
    cursor = 0
    for size in sizes:
        out.append(piece[cursor : cursor + size])
        cursor += size
    return out


def _fixed_pieces(piece: str, max_length: int) -> list[str]:
    return [piece[i : i + max_length] for i in range(0, len(piece), max_length)]


def process_oov(tokens: list[Token], seg_meta: dict, config: Config) -> tuple[list[Token], dict]:
    out: list[Token] = []
    candidate_fps: list[str] = []
    fragment_fps: list[str] = []
    oov_count = 0
    i = 0
    total = len(tokens)
    while i < total:
        token = tokens[i]
        if token.source != "single":
            out.append(token)
            i += 1
            continue
        j = i
        while j < total and tokens[j].source == "single":
            j += 1
        run = tokens[i:j]
        run_text = "".join(t.text for t in run)
        run_start = run[0].start
        fragment_fps.append(fingerprint(run_text, prefix="oov-fragment"))
        for piece_text, chunk_start, chunk_end, basis in _hard_chunks(run_start, run_text, config.oov_max_length):
            if len(piece_text) <= 1:
                pieces = [piece_text]
                inner_basis = "single-char"
            elif config.oov_rule == "equal":
                pieces = _equal_pieces(piece_text)
                inner_basis = "equal:sqrt"
            else:
                pieces = _fixed_pieces(piece_text, config.oov_max_length)
                inner_basis = "fixed_max:max_len"
            cursor = chunk_start
            for piece in pieces:
                fp = fingerprint(
                    {
                        "text": piece,
                        "start": cursor,
                        "end": cursor + len(piece),
                        "rule": config.oov_rule,
                        "max_length": config.oov_max_length,
                    },
                    prefix="oov-token",
                )
                out.append(
                    Token(
                        text=piece,
                        start=cursor,
                        end=cursor + len(piece),
                        source="oov",
                        segment_basis=f"{basis}|{inner_basis}",
                        fingerprint=fp,
                    )
                )
                candidate_fps.append(fp)
                oov_count += 1
                cursor += len(piece)
        i = j

    rule_fp = config.oov_param_fingerprint()
    content_fp = combine("oov-content", seg_meta["content_fingerprint"], fragment_fps, rule_fp)
    meta = {
        "content_fingerprint": content_fp,
        "rule": config.oov_rule,
        "oov_max_length": config.oov_max_length,
        "oov_token_count": oov_count,
        "candidate_fingerprints": candidate_fps,
        "token_count": len(out),
    }
    return out, meta
