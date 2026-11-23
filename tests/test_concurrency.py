from fablock.pipeline import Pipeline


def test_concurrent_serial_equivalence(cfg, dataset):
    serial = Pipeline(cfg.with_overrides(thread_pool_workers=1), dataset)
    parallel = Pipeline(cfg.with_overrides(thread_pool_workers=4), dataset)
    r_serial = serial.run(2)
    r_parallel = parallel.run(2)
    for rs, rp in zip(r_serial, r_parallel):
        assert rs["outputs_fp"] == rp["outputs_fp"]
        assert rs["compare"] == rp["compare"]
        assert rs["overflows"] == rp["overflows"]
    assert r_parallel[-1]["peak_concurrency"] <= 4


def test_repeated_runs_identical(cfg, dataset):
    r1 = Pipeline(cfg, dataset).run(2)
    r2 = Pipeline(cfg, dataset).run(2)
    for a, b in zip(r1, r2):
        assert a["outputs_fp"] == b["outputs_fp"]
        assert a["compare"] == b["compare"]
