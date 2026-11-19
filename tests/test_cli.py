"""命令行端到端冒烟测试，保证各子命令返回码契约稳定。"""

from __future__ import annotations

from photometric_altopt.cli import main


def test_generate_run_gradcheck(tmp_path, small_config, monkeypatch):
    ws = str(tmp_path / "ws")
    assert main(["generate", "--workspace", ws, "--size", "24"]) == 0
    assert main(["run", "--workspace", ws, "--run-name", "base", "--max-iterations", "2"]) == 0
    assert main(["gradcheck", "--workspace", ws, "--samples", "4"]) == 0


def test_cli_business_error_exit_code(tmp_path, small_config):
    ws = str(tmp_path / "ws")
    main(["generate", "--workspace", ws, "--size", "24"])
    rc = main(["run", "--workspace", ws, "--run-name", "z", "--inject-zero-light"])
    assert rc == 1


def test_crash_resume_cli(tmp_path, small_config):
    ws = str(tmp_path / "ws")
    main(["generate", "--workspace", ws, "--size", "24"])
    main(["run", "--workspace", ws, "--run-name", "c", "--max-iterations", "4", "--crash-after", "1"])
    assert main(["resume", "--workspace", ws, "--run-name", "c", "--max-iterations", "4"]) == 0


def test_ill_conditioned_injection_cli(tmp_path, small_config):
    import json
    from pathlib import Path

    ws = str(tmp_path / "ws")
    main(["generate", "--workspace", ws, "--size", "24"])
    rc = main(
        ["run", "--workspace", ws, "--run-name", "ill", "--inject-ill", "--max-iterations", "1"]
    )
    assert rc == 0  # 状态型标记，不中断管线
    record = json.loads(
        (Path(ws) / "trajectories" / "ill" / "trajectories" / "round_000.json").read_text()
    )
    assert record["ill_conditioned_pixels"] == 24 * 24
