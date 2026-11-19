import pytest

from terrain_workbench.heightfield import generate_heightfield
from terrain_workbench.pyramid import build_pyramid
from terrain_workbench.quadtree import Quadtree


@pytest.fixture(scope="session")
def heights257():
    return generate_heightfield(257, 12345)


@pytest.fixture(scope="session")
def pyramid257(heights257):
    return build_pyramid(heights257)


@pytest.fixture(scope="session")
def quadtree257(heights257):
    return Quadtree(heights257)
