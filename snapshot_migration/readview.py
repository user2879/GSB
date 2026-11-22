"""Two-dimensional cursor scans over a single, pinned generation."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .errors import CursorError
from .fragments import Fragment


@dataclass
class Cursor:
    range_id: str
    key: str
    fragment_id: str
    fragment_version: int

    def to_dict(self):
        return {"range_id": self.range_id, "key": self.key,
                "fragment_id": self.fragment_id,
                "fragment_version": self.fragment_version}


class ReadView:
    """A stable view bound to one generation.

    The scan plan (ordered range/key cells) is materialized when the view opens,
    so migration activity on later generations cannot change the result stream.
    Diagnostic repacking of the pinned generation is still detectable through
    fragment version anchors, which drive either safe continuation or E_CURSOR.
    """

    def __init__(self, engine, view_id: str, generation_id: str):
        self.engine = engine
        self.view_id = view_id
        self.generation_id = generation_id
        self.repo = engine.repo
        self.plan = self._materialize_plan()
        self.closed = False

    def _materialize_plan(self):
        plan = []
        for frag in self.repo.active_fragments(self.generation_id):
            for key in frag.keys:
                plan.append((frag.range_id, key, frag.fragment_id, frag.version))
        plan.sort(key=lambda cell: (cell[0], cell[1]))
        return plan

    # ------------------------------------------------------------ scanning
    def scan(self, *, limit: int, cursor: Optional[Cursor] = None,
             range_id: Optional[str] = None,
             key_low: Optional[str] = None,
             key_high: Optional[str] = None) -> dict:
        if self.closed:
            raise CursorError("read view is closed",
                              generation_id=self.generation_id)
        if not isinstance(limit, int) or limit <= 0:
            raise CursorError("limit must be a positive integer",
                              generation_id=self.generation_id)
        start = 0
        if cursor is not None:
            start = self._anchor(cursor) + 1

        rows: List[dict] = []
        next_cursor = None
        for cell in self.plan[start:]:
            cell_range, cell_key, frag_id, version = cell
            if range_id is not None and cell_range != range_id:
                continue
            if key_low is not None and cell_key < key_low:
                continue
            if key_high is not None and cell_key > key_high:
                continue
            if len(rows) >= limit:
                next_cursor = Cursor(cell_range, cell_key, frag_id, version)
                break
            rows.append({"range_id": cell_range, "key": cell_key,
                         "fragment_id": frag_id, "fragment_version": version})
        return {"generation_id": self.generation_id, "rows": rows,
                "next_cursor": next_cursor.to_dict() if next_cursor else None}

    def _anchor(self, cursor: Cursor) -> int:
        index = None
        for position, cell in enumerate(self.plan):
            if cell[0] == cursor.range_id and cell[1] == cursor.key:
                index = position
                break
        if index is None:
            raise CursorError(
                "cursor anchor key no longer exists in the pinned generation",
                generation_id=self.generation_id,
                fragment_id=cursor.fragment_id, field="key",
                range_id=cursor.range_id, **{"key": cursor.key})

        current = self._live_fragment(cursor.fragment_id)
        if current is None:
            raise CursorError(
                "cursor fragment disappeared and cannot be re-anchored",
                generation_id=self.generation_id,
                fragment_id=cursor.fragment_id,
                range_id=cursor.range_id, **{"key": cursor.key})
        if current.version != cursor.fragment_version:
            if self._key_still_ordered(current, cursor.key):
                # Safe continuation: re-anchor by (range, key); no row skipped or
                # re-read because the plan cell identity is the key itself.
                return index
            raise CursorError(
                "fragment was split/merged and the anchor key cannot be located "
                "in deterministic order",
                generation_id=self.generation_id,
                fragment_id=cursor.fragment_id, range_id=cursor.range_id,
                field="fragment_version", **{"key": cursor.key})
        return index

    def _live_fragment(self, fragment_id) -> Optional[Fragment]:
        versions = self.repo.fragment_versions(fragment_id)
        active = [item for item in versions if item.status == "active"]
        return active[-1] if active else None

    def _key_still_ordered(self, fragment: Fragment, key: str) -> bool:
        return key in fragment.keys

    def iterate_all(self, page_size: int) -> List[dict]:
        out: List[dict] = []
        cursor = None
        while True:
            page = self.scan(limit=page_size, cursor=cursor)
            out.extend(page["rows"])
            cursor = Cursor(**page["next_cursor"]) if page["next_cursor"] else None
            if cursor is None:
                return out

    def close(self):
        if not self.closed:
            self.engine.close_read_view(self.view_id, self.generation_id)
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
