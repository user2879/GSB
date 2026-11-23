"""任务执行前自检：生成确定性样本并验证临时目录读写。"""

import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from timbre_pipeline.synthesize import generate_sample_workspace


def main():
    with tempfile.TemporaryDirectory(prefix="timbre_selfcheck_") as tmp:
        probe = os.path.join(tmp, "probe.txt")
        with open(probe, "w", encoding="utf-8") as handle:
            handle.write("ok")
        with open(probe, "r", encoding="utf-8") as handle:
            assert handle.read() == "ok"
        layout = generate_sample_workspace(os.path.join(tmp, "sample_work"))
        wav_count = len([name for name in os.listdir(layout["wav_dir"]) if name.endswith(".wav")])
        with open(layout["config_path"], "r", encoding="utf-8") as handle:
            config = json.load(handle)
        with open(layout["injection_path"], "r", encoding="utf-8") as handle:
            injections = json.load(handle)
        assert wav_count == 6, wav_count
        assert config["sample_rate"] == 16000
        assert len(injections) == 4
        print("SELFCHECK_OK tmp={} wav={} dims={} injections={}".format(
            tmp, wav_count, len(config["dimensions"]), len(injections)))
        print("CONFIG", layout["config_path"])
        print("INJECTIONS", layout["injection_path"])


if __name__ == "__main__":
    main()
