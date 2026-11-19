"""路径规范化比较模块。

处理点段、父目录段、长名段、Unicode 规范化变体与大小写变体，
按策略生成规范比较键，并逐步量化比较误差传播来源。
链接声明仅参与虚拟路径解析，不访问真实文件系统。
"""

from __future__ import annotations

import posixpath
import unicodedata
from typing import Optional

from .errors import E_LINK_ESCAPE, E_TRAVERSAL
from .models import LinkDecl, NormalizationPolicy, PathNormResult

_MAX_LINK_HOPS = 32


def _split_virtual(path: str) -> tuple[bool, list[str]]:
    """拆分虚拟路径，兼容 POSIX 与 Windows 反斜杠 / 盘符形式。"""
    text = path.replace("\\", "/")
    is_absolute = text.startswith("/") or (len(text) >= 2 and text[1] == ":")
    if len(text) >= 2 and text[1] == ":":
        text = text[2:]
        if text.startswith("/"):
            text = text[1:]
    segments = [seg for seg in text.split("/") if seg not in ("", ".")]
    return is_absolute, segments


def resolve_segments(segments: list[str], root_depth: int) -> tuple[list[str], int]:
    """点段 / 父目录段消解。

    返回 (栈内剩余段, 越界深度)。越界深度 = 根深度之上的父目录回退次数，
    越界深度 = 在根深度之上继续回退的父目录段数量。
    """
    stack: list[str] = []
    escape_depth = 0
    for seg in segments:
        if seg == "..":
            if len(stack) > root_depth:
                stack.pop()
            elif len(stack) == root_depth:
                escape_depth += 1
        else:
            stack.append(seg)
    return stack, escape_depth


def _apply_text_policy(segment: str, policy: NormalizationPolicy) -> tuple[str, int, int]:
    encoding_error = 0
    casefold_error = 0
    result = segment
    if policy.unicode_nfc:
        nfc = unicodedata.normalize("NFC", result)
        if nfc != result:
            encoding_error += abs(len(result) - len(nfc)) or sum(
                1 for a, b in zip(result, nfc) if a != b
            )
        result = nfc
    if policy.fold_case:
        folded = result.casefold()
        if folded != result:
            casefold_error += abs(len(result) - len(folded)) or sum(
                1 for a, b in zip(result, folded) if a != b
            )
        result = folded
    return result, encoding_error, casefold_error


def _normalize_key(parts: list[str], policy: NormalizationPolicy) -> tuple[str, int, int]:
    encoding_total = 0
    casefold_total = 0
    normalized: list[str] = []
    for seg in parts:
        new_seg, enc_err, cf_err = _apply_text_policy(seg, policy)
        encoding_total += enc_err
        casefold_total += cf_err
        normalized.append(new_seg)
    return posixpath.join("/", *normalized) if normalized else "/", encoding_total, casefold_total


def _link_index(links: tuple[LinkDecl, ...], policy: NormalizationPolicy) -> dict[str, LinkDecl]:
    index: dict[str, LinkDecl] = {}
    for link in links:
        _, segments = _split_virtual(link.at)
        key, _, _ = _normalize_key(segments, policy)
        index[key] = link
    return index


def normalize_path(
    target_path: str,
    root: str,
    policy: NormalizationPolicy,
    links: tuple[LinkDecl, ...] = (),
) -> PathNormResult:
    """生成规范比较键并量化分步误差。"""
    _, root_segments = _split_virtual(root)
    is_absolute, raw_segments = _split_virtual(target_path)

    full_parts = list(raw_segments) if is_absolute else root_segments + list(raw_segments)

    resolved_parts, escape_depth = resolve_segments(full_parts, len(root_segments))
    blocked: Optional[str] = E_TRAVERSAL if escape_depth > 0 else None

    canonical, enc_err, cf_err = _normalize_key(resolved_parts, policy)
    result = PathNormResult(
        canonical_key=canonical,
        encoding_error=enc_err,
        casefold_error=cf_err,
        traversal_depth=escape_depth,
        blocked_code=blocked,
        reachable_path=canonical,
    )

    if policy.resolve_links and links and escape_depth == 0:
        link_result = _resolve_links(resolved_parts, root_segments, policy, links)
        result.link_steps = link_result.link_steps
        result.reachable_path = link_result.canonical_key
        if link_result.link_escape:
            result.link_escape = True
            result.blocked_code = E_LINK_ESCAPE
    return result


def _resolve_links(
    parts: list[str],
    root_segments: list[str],
    policy: NormalizationPolicy,
    links: tuple[LinkDecl, ...],
) -> PathNormResult:
    """按链接声明重建实际可达路径，检测链接跳转逃逸与链接环。"""
    link_map = _link_index(links, policy)
    current = list(parts)
    hops = 0
    seen: set[tuple[str, ...]] = set()
    escape = False

    while hops < _MAX_LINK_HOPS:
        match = _longest_prefix_link(current, link_map, policy)
        if match is None:
            break
        link, matched_len = match

        fingerprint = tuple(current)
        if fingerprint in seen:
            break
        seen.add(fingerprint)

        target_abs, target_segments = _split_virtual(link.target)
        if target_abs:
            replacement = target_segments
        else:
            base = current[: matched_len - 1]
            replacement = base + target_segments
        current = replacement + current[matched_len:]
        hops += link.steps

        resolved, escape_depth = resolve_segments(current, len(root_segments))
        current = resolved
        if escape_depth > 0 or not _is_within(current, root_segments):
            escape = True
            break

    canonical, _, _ = _normalize_key(current, policy)
    if not _is_within(current, root_segments):
        escape = True
    return PathNormResult(
        canonical_key=canonical,
        link_steps=hops,
        link_escape=escape,
        reachable_path=canonical,
    )


def _longest_prefix_link(
    parts: list[str],
    link_map: dict[str, LinkDecl],
    policy: NormalizationPolicy,
) -> Optional[tuple[LinkDecl, int]]:
    best: Optional[tuple[LinkDecl, int]] = None
    for size in range(1, len(parts) + 1):
        key, _, _ = _normalize_key(parts[:size], policy)
        link = link_map.get(key)
        if link is not None:
            best = (link, size)
    return best


def _is_within(parts: list[str], root_segments: list[str]) -> bool:
    if len(parts) < len(root_segments):
        return False
    return parts[: len(root_segments)] == root_segments
