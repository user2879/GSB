from __future__ import annotations

import threading

from candidate_diagnostics.concurrency import analyze_many_concurrent
from candidate_diagnostics.config import Config, deep_merge
from candidate_diagnostics.pipeline import analyze_many


def test_concurrent_results_equal_serial(config, requests):
    serial = analyze_many(requests, config)
    concurrent, report = analyze_many_concurrent(requests, config)
    assert {rid: result.to_dict() for rid, result in concurrent.items()} == {
        rid: result.to_dict() for rid, result in serial.items()
    }
    assert report["business_fields_match_serial"] if "business_fields_match_serial" in report else True
    assert report["observed_max_concurrency"] <= config.max_workers
    assert report["never_exceeds_config"] is True
    assert list(report["stable_sort_keys"]) == sorted(requests)


def test_worker_count_comes_from_config_and_is_capped(requests):
    one_worker = Config(deep_merge({}, {"concurrency": {"max_workers": 1}}))
    _, report = analyze_many_concurrent(requests, one_worker)
    assert report["actual_max_workers_argument"] == 1
    assert report["observed_max_concurrency"] <= 1

    three_worker = Config(deep_merge({}, {"concurrency": {"max_workers": 3}}))
    _, report3 = analyze_many_concurrent(requests, three_worker)
    assert report3["actual_max_workers_argument"] == 3
    assert report3["observed_max_concurrency"] <= 3


def test_barrier_proves_parallel_execution_and_cap(config, requests, monkeypatch):
    import candidate_diagnostics.concurrency as concurrency_module

    original = concurrency_module.analyze_request
    barrier = threading.Barrier(2)

    def synchronized_analyze(request, current_config):
        barrier.wait(timeout=2.0)
        return original(request, current_config)

    monkeypatch.setattr(concurrency_module, "analyze_request", synchronized_analyze)
    two_worker = Config(deep_merge(config.raw, {"concurrency": {"max_workers": 2}}))
    results, report = analyze_many_concurrent(requests, two_worker)
    assert len(results) == len(requests)
    assert report["observed_max_concurrency"] >= 2
    assert report["observed_max_concurrency"] <= 2


def test_different_concurrency_degrees_produce_same_fields(config, requests):
    serial = analyze_many(requests, config)
    serial_dict = {rid: result.to_dict() for rid, result in serial.items()}
    for workers in (1, 2, 3):
        current = Config(deep_merge(config.raw, {"concurrency": {"max_workers": workers}}))
        concurrent, report = analyze_many_concurrent(requests, current)
        assert {rid: result.to_dict() for rid, result in concurrent.items()} == serial_dict
        assert report["observed_max_concurrency"] <= workers
