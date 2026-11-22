"""Test helpers shared by the suite."""

from __future__ import annotations

import json
import os
import tempfile

from workbench.loader import load_declaration


def spec_from_payload(payload: dict):
    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                         encoding="utf-8")
    json.dump(payload, handle)
    handle.close()
    try:
        return load_declaration(handle.name)
    finally:
        os.unlink(handle.name)


def tmp_declaration(payload: dict) -> str:
    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                         encoding="utf-8")
    json.dump(payload, handle)
    handle.close()
    return handle.name
