from __future__ import annotations

from pathlib import Path

from snapdiag.store import Session
from snapdiag.view import build_diagnostic_view, render_diagnostic_text


def test_view_has_stable_section_order(tmp_path: Path) -> None:
    session = Session.initialize(tmp_path / "view", block_size=512, block_count=8, threshold=0.5)
    session.create_snapshot("s0")
    session.write_block(1, b"x" * 512)
    session.audit()
    view = build_diagnostic_view(session, target="s0")
    text = render_diagnostic_text(view)
    headings = ["SNAPSHOTS", "BITMAP OVERVIEW", "ROLLBACK PATH", "TEMP AREA", "AUDIT", "MANIFEST", "RECENT EVENTS"]
    positions = [text.index(f"\n{heading}\n") for heading in headings]
    assert positions == sorted(positions)
    assert view["snapshots"][0]["snapshot_id"] == "s0"
    assert view["rollback_path"]["ok"]
    assert view["audit"]["available"]
    assert view["manifest"]["consistent"]

