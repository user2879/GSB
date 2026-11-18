"""演示 SIGTERM 优雅退出与重启恢复，并比对业务结果一致性。"""

import json
import os
import signal
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(root):
    work_sig = os.path.join(root, "work_signal")
    cmd = [sys.executable, "-m", "xmodal.cli", "run",
           "--input", os.path.join(root, "input"),
           "--workdir", work_sig,
           "--config", os.path.join(REPO, "config/alignment.json"),
           "--spec", os.path.join(REPO, "config/format_spec.json"),
           "--injections", os.path.join(REPO,
                                        "config/injections_signal.json")]
    proc = subprocess.Popen(cmd, cwd=REPO)
    time.sleep(0.6)
    proc.send_signal(signal.SIGTERM)    # 可处理退出信号
    proc.wait(timeout=10)
    print("SIGTERM 退出码:", proc.returncode)
    assert proc.returncode == 0

    resume_cmd = [sys.executable, "-m", "xmodal.cli", "run",
                  "--input", os.path.join(root, "input"),
                  "--workdir", work_sig,
                  "--config", os.path.join(REPO, "config/alignment.json"),
                  "--spec", os.path.join(REPO, "config/format_spec.json")]
    subprocess.run(resume_cmd, cwd=REPO, check=True)
    resumed = json.load(open(os.path.join(work_sig, "report.json")))
    full = json.load(open(os.path.join(root, "work_v2/report.json")))
    for key in ("alignment", "retrieval", "content_hash"):
        assert resumed[key] == full[key], key
    print("恢复后业务结果与全量运行一致: content_hash =",
          resumed["content_hash"])


if __name__ == "__main__":
    main(sys.argv[1])
