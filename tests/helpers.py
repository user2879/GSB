from __future__ import annotations

from typing import Any

import yaml

from story_workbench.samples import (
    boundary_spec,
    injection_spec,
    merge_spec,
    normal_spec,
)
from story_workbench.spec import parse_spec


def parse_document(document: dict[str, Any], name: str = "sample.yaml"):
    text = yaml.safe_dump(document, allow_unicode=True, sort_keys=False)
    return parse_spec(text, source_name=name)


def normal():
    return parse_document(normal_spec(), "normal.yaml")


def merge():
    return parse_document(merge_spec(), "merge.yaml")


def boundary():
    return parse_document(boundary_spec(), "boundary.yaml")


def injection():
    return parse_document(injection_spec(), "injection.yaml")
