from receipt_ledger.deterministic import (
    build_parameters, forward, output_digest, quantize, request_destination,
)
import numpy as np


def test_parameters_bit_identical_across_calls(v1_spec):
    worker = v1_spec.workers[0]
    w1, b1, s1 = build_parameters(worker, 1)
    w2, b2, s2 = build_parameters(worker, 1)
    assert s1 == s2
    np.testing.assert_array_equal(w1, w2)
    np.testing.assert_array_equal(b1, b2)


def test_parameter_version_changes_summary(v1_spec):
    worker = v1_spec.workers[0]
    _, _, s1 = build_parameters(worker, 1)
    _, _, s2 = build_parameters(worker, 2)
    assert s1 != s2


def test_quantization_rules():
    arr = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    assert quantize(arr, "fp32").dtype == np.float64
    q = quantize(arr, "int8")
    levels = np.rint(q / (np.max(np.abs(arr)) / 127.0))
    assert np.all(np.abs(levels) <= 127)
    assert np.max(np.abs(q - arr)) <= 2.0 / 127.0


def test_forward_and_digest_stable(v1_spec):
    worker = v1_spec.workers[0]
    weight, bias, _ = build_parameters(worker, 1)
    req = v1_spec.requests[0]
    h1 = output_digest(forward(weight, bias, req.vector))
    h2 = output_digest(forward(weight, bias, req.vector))
    assert h1 == h2 and len(h1) == 64


def test_destination_independent_of_set_order():
    d1 = request_destination("client-aa", ("a", "b", "c"))
    d2 = request_destination("client-aa", ("c", "b", "a"))
    assert d1 == d2
