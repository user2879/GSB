from __future__ import annotations

import dataclasses
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from glyphforge.application import GlyphApplication
from glyphforge.diagnostics import DiagnosticService, DiagnosticViewState
from glyphforge.errors import ErrorCode, GlyphForgeError
from glyphforge.samples import sample_glyphs


def _service() -> DiagnosticService:
    glyphs = sample_glyphs()
    app = GlyphApplication(
        {key: glyphs[key] for key in ("A", "O")},
        active_sizes={"A": (12, 16), "O": (12,)},
        worker_budget=2,
        memory_budget_bytes=4_000_000,
    )
    app.render_all()
    return DiagnosticService(app)


def test_snapshot_is_versioned_and_immutable():
    service = _service()
    snapshot = service.snapshot()
    assert snapshot.version == service.current_version
    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot.version = 999  # type: ignore[misc]
    assert isinstance(snapshot.pixels, bytes) if hasattr(snapshot, "pixels") else True
    for page in snapshot.atlas_pages:
        assert isinstance(page.pixels, bytes)
        assert all(isinstance(region.content_hash, str) for region in page.regions)
    bitmap = next(iter(snapshot.bitmap_map.values())).bitmap
    with pytest.raises(ValueError):
        bitmap[0, 0] = 1


def test_view_state_replaces_whole_snapshot_and_rejects_stale_versions():
    service = _service()
    old = service.snapshot()
    state = DiagnosticViewState()
    assert state.apply(old) is True
    assert state.apply(old) is False

    service.application.edit_point(
        "A",
        contour_index=0,
        segment_index=1,
        role="control1",
        point=__import__("glyphforge").Point(530, 950),
    )
    new = service.snapshot()
    assert new.version > old.version
    assert state.apply(new) is True
    with pytest.raises(GlyphForgeError) as caught:
        state.apply(old)
    assert caught.value.code is ErrorCode.SNAPSHOT_STALE

    with pytest.raises(GlyphForgeError) as caught:
        service.require_current(old)
    assert caught.value.code is ErrorCode.SNAPSHOT_STALE


def test_diagnostic_view_has_no_application_or_recompute_reference():
    from PyQt5.QtWidgets import QApplication

    from glyphforge.gui import DiagnosticView

    qt_app = QApplication.instance() or QApplication([])
    view = DiagnosticView()
    assert not hasattr(view, "application")
    assert not hasattr(view, "command")
    assert qt_app is not None
