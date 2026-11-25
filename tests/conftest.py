import pytest

from obs_pipeline import gen
from obs_pipeline.spec import validate_spec


@pytest.fixture()
def sample_spec_dict():
    return gen.build_sample_spec(seed=42)


@pytest.fixture()
def sample_spec(sample_spec_dict):
    return validate_spec(sample_spec_dict)
