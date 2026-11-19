from __future__ import annotations

import pytest

from npr_editor.errors import ERROR_CODES, NprError
from npr_editor.render import verify_render_hashes


REQUIRED_CODES = {
    "E_SCHEMA_VERSION",
    "E_FORMAT_ENDIAN",
    "E_FORMAT_INDEX",
    "E_STROKE_SAMPLE",
    "E_UNDO_EMPTY",
    "E_SNAPSHOT_STALE",
    "E_RENDER_MISMATCH",
}


def test_required_business_codes_are_registered():
    assert REQUIRED_CODES.issubset(ERROR_CODES)


def test_render_hash_mismatch_is_injectable():
    with pytest.raises(NprError) as caught:
        verify_render_hashes({"tree_crown": "expected"}, {"tree_crown": "actual"})
    assert caught.value.code == "E_RENDER_MISMATCH"
    assert caught.value.details["changed_shapes"] == ["tree_crown"]
    assert caught.value.to_dict()["code"] == "E_RENDER_MISMATCH"
