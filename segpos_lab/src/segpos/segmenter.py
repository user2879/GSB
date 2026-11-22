"""正向最大匹配（FMM）分词阶段。"""
from __future__ import annotations

from .fingerprint import combine, fingerprint
from .models import Lexicon, Sentence, Token


class FmmSegmenter:
    def __init__(self, lexicon: Lexicon):
        self.lexicon = lexicon
        self.max_word_len = max((len(word) for word in lexicon.entries), default=1)

    def segment(self, sentence: Sentence) -> tuple[list[Token], dict]:
        text = sentence.text
        tokens: list[Token] = []
        hits: list[str] = []
        pos = 0
        total = len(text)
        while pos < total:
            upper = min(total, pos + self.max_word_len)
            best: tuple | None = None
            for end in range(upper, pos, -1):
                word = text[pos:end]
                entry = self.lexicon.entries.get(word)
                if entry is None:
                    continue
                candidate = (len(word), entry.freq, word, entry)
                if best is None:
                    best = candidate
                else:
                    if len(word) > best[0]:
                        best = candidate
                    elif len(word) == best[0] and (entry.freq, word) > (best[1], best[2]):
                        # 同长：词频降序，再按词形字典序（较大者优先）
                        best = candidate
            if best is not None:
                length, freq, word, entry = best
                end = pos + length
                token = Token(
                    text=word,
                    start=pos,
                    end=end,
                    source="lexicon",
                    freq=freq,
                    match_len=length,
                )
                token.fingerprint = fingerprint(
                    {"text": word, "start": pos, "end": end, "freq": freq, "source": "lexicon"},
                    prefix="seg-token",
                )
                tokens.append(token)
                hits.append(word)
                pos = end
            else:
                ch = text[pos]
                token = Token(
                    text=ch,
                    start=pos,
                    end=pos + 1,
                    source="single",
                    match_len=1,
                )
                token.fingerprint = fingerprint(
                    {"text": ch, "start": pos, "end": pos + 1, "source": "single"},
                    prefix="seg-token",
                )
                tokens.append(token)
                pos += 1

        text_fp = fingerprint(text, prefix="sentence-text")
        hit_fp = fingerprint(sorted(hits), prefix="hit-set")
        content_fp = combine("segment-content", text_fp, hit_fp)
        meta = {
            "content_fingerprint": content_fp,
            "text_fingerprint": text_fp,
            "hit_set_fingerprint": hit_fp,
            "hit_entries": sorted(set(hits)),
            "token_count": len(tokens),
        }
        return tokens, meta
