"""路径规范化、穿越深度与链接逃逸测试。"""

from __future__ import annotations

import pytest

from audit_engine.errors import E_LINK_ESCAPE, E_TRAVERSAL
from audit_engine.models import LinkDecl, NormalizationPolicy
from audit_engine.pathnorm import normalize_path, resolve_segments

ROOT = "/srv/audit-root"
POLICY = NormalizationPolicy()


def test_dot_segments_are_collapsed():
    result = normalize_path(f"{ROOT}/docs/./report.txt", ROOT, POLICY)
    assert result.canonical_key == f"{ROOT}/docs/report.txt"
    assert result.traversal_depth == 0
    assert result.blocked_code is None


def test_parent_traversal_within_root_is_safe():
    result = normalize_path(f"{ROOT}/docs/../secret/key.txt", ROOT, POLICY)
    assert result.canonical_key == f"{ROOT}/secret/key.txt"
    assert result.blocked_code is None


@pytest.mark.parametrize(
    "target,expected_depth",
    [
        (f"{ROOT}/a/b/../../../../etc/passwd", 2),
        (f"{ROOT}/../../etc/x", 2),
        (f"{ROOT}/../etc/x", 1),
        ("rel/../../etc/x", 1),
    ],
)
def test_traversal_depth_quantified(target, expected_depth):
    result = normalize_path(target, ROOT, POLICY)
    assert result.blocked_code == E_TRAVERSAL
    assert result.traversal_depth == expected_depth


def test_resolve_segments_root_boundary():
    stack, depth = resolve_segments(
        ["srv", "audit-root", "..", "etc"], root_depth=2
    )
    assert depth == 1
    assert stack == ["srv", "audit-root", "etc"]


def test_unicode_nfc_and_casefold_error_quantified():
    # U+0065 U+0301 在 NFC 下合并；大写 ASCII 折叠改变码点
    target = f"{ROOT}/Cafe\u0301/ABC.txt"
    strict = normalize_path(target, ROOT, POLICY)
    assert strict.encoding_error >= 1
    assert strict.casefold_error >= 1

    raw_policy = NormalizationPolicy(fold_case=False, unicode_nfc=False, name="raw")
    raw = normalize_path(target, ROOT, raw_policy)
    assert raw.encoding_error == 0
    assert raw.casefold_error == 0
    assert raw.canonical_key != strict.canonical_key


def test_symlink_escape_detected():
    links = (LinkDecl(at=f"{ROOT}/docs/outside", target="/etc/outside", link_type="symlink"),)
    result = normalize_path(f"{ROOT}/docs/outside/file.txt", ROOT, POLICY, links)
    assert result.blocked_code == E_LINK_ESCAPE
    assert result.link_steps == 1
    assert result.reachable_path == "/etc/outside/file.txt"


def test_hardlink_within_root_is_safe():
    links = (
        LinkDecl(at=f"{ROOT}/docs/hard", target=f"{ROOT}/var/data.bin", link_type="hardlink"),
    )
    result = normalize_path(f"{ROOT}/docs/hard", ROOT, POLICY, links)
    assert result.link_escape is False
    assert result.reachable_path == f"{ROOT}/var/data.bin"


def test_hardlink_escape_detected():
    links = (
        LinkDecl(at=f"{ROOT}/docs/escape", target="/var/outside.bin", link_type="hardlink"),
    )
    result = normalize_path(f"{ROOT}/docs/escape", ROOT, POLICY, links)
    assert result.blocked_code == E_LINK_ESCAPE
    assert result.reachable_path == "/var/outside.bin"


def test_disabled_link_resolution_keeps_virtual_key():
    links = (LinkDecl(at=f"{ROOT}/docs/outside", target="/etc/outside", link_type="symlink"),)
    policy = NormalizationPolicy(resolve_links=False, name="no_link")
    result = normalize_path(f"{ROOT}/docs/outside/file.txt", ROOT, policy, links)
    assert result.link_steps == 0
    assert result.link_escape is False
    assert result.canonical_key == f"{ROOT}/docs/outside/file.txt"


def test_link_loop_terminates():
    links = (
        LinkDecl(at=f"{ROOT}/loop", target=f"{ROOT}/loop2", link_type="symlink"),
        LinkDecl(at=f"{ROOT}/loop2", target=f"{ROOT}/loop", link_type="symlink"),
    )
    result = normalize_path(f"{ROOT}/loop/x", ROOT, POLICY, links)
    assert result.link_steps <= 64  # 有界终止，不无限跳转
