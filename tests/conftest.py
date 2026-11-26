import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from decchain.config import DecodeConfig
from decchain.models import CandidateSequence, Request, Step
from decchain.samples_gen import build_sample_requests


@pytest.fixture
def config():
    return DecodeConfig(width=4, tier="ROUND_HALF_UP", top_k=5,
                        top_p_mille=900, tolerance_units=2)


@pytest.fixture
def simple_request():
    steps = [
        Step(index=1, logprobs={100: 0.0, 101: -0.4, 102: -1.1,
                                103: -2.0, 104: -3.1}),
        Step(index=2, logprobs={100: -0.2, 101: -0.5, 102: -0.9,
                                103: -1.7, 104: -2.6}),
        Step(index=3, logprobs={100: -1.0, 101: -0.1, 102: -0.6,
                                103: -1.4, 104: -2.2}),
    ]
    seq = CandidateSequence(seq_id=1, seed_token=7, channel="SAMPLE",
                            temperature=1.0, steps=steps)
    return Request(request_id="R001", sequences=[seq])


@pytest.fixture
def sample_requests():
    return build_sample_requests()
