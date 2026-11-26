from arg_review.cli import main


def test_cli_generate_verify_and_diagnose(tmp_path, capsys):
    sample_root = tmp_path / "samples"
    state_root = tmp_path / "artifacts"
    report = tmp_path / "equiv.json"
    assert main(["generate-samples", "--root", str(sample_root)]) == 0
    assert main([
        "verify-edits",
        "--samples", str(sample_root),
        "--state-dir", str(tmp_path / "verify"),
        "--report", str(report),
    ]) == 0
    assert report.exists()
    assert main([
        "diagnose",
        "--samples", str(sample_root),
        "--state-dir", str(state_root),
    ]) == 0
    assert (state_root / "diagnostic_report.json").exists()


def test_numpy_is_available_as_declared_environment_dependency():
    import numpy as np

    assert np.__version__
