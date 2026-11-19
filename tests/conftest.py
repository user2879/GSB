from __future__ import annotations

import copy

import pytest

from npr_editor.samples import build_sample_project, legacy_v1_document, legacy_v2_document


@pytest.fixture
def sample_project():
    return build_sample_project()


@pytest.fixture
def current_document(sample_project):
    return sample_project.to_dict()


@pytest.fixture
def v1_document(sample_project):
    return copy.deepcopy(legacy_v1_document(sample_project))


@pytest.fixture
def v2_document(sample_project):
    return copy.deepcopy(legacy_v2_document(sample_project))
