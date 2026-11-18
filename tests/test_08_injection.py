from lora_bench.injection import run_injection_suite


def test_all_injection_scenarios_hit_expected_codes(tmp_path):
    report = run_injection_suite(str(tmp_path / "report"))
    assert report["total"] == 6
    assert report["hit_count"] == 6
    for entry in report["entries"]:
        assert entry["hit"] is True
        assert entry["observed_code"] == entry["expected_code"]


def test_report_is_written_atomically(tmp_path):
    report_root = str(tmp_path / "report")
    run_injection_suite(report_root)
    import os

    report_path = os.path.join(report_root, "injection_report.json")
    assert os.path.exists(report_path)
    leftovers = [
        name
        for name in os.listdir(report_root)
        if name.endswith(".tmp") or name.startswith(".tmp-")
    ]
    assert leftovers == []


def test_task_fault_does_not_stop_other_tasks(tmp_path):
    report = run_injection_suite(str(tmp_path / "report2"))
    victim = next(
        entry for entry in report["entries"] if entry["injection_id"] == "INJ-06"
    )
    assert victim["hit"] is True
    assert victim["isolated"] is True
    assert victim["stage"] == "train"
