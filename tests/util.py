import json
import os
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_config(tmp, **overrides):
    cfg = {
        "sample_seed": 20261126,
        "request_count": 8,
        "input_dim": 4,
        "hidden_dim": 6,
        "output_dim": 4,
        "batch_size": 2,
        "fixed_point": {"frac_bits": 8, "word_bits": 32,
                        "rounding": "floor", "saturation": "saturate"},
        "workers": [{"worker_id": "w0", "seed": 101},
                    {"worker_id": "w1", "seed": 202}],
        "scheduler": {"global_limit": 2, "per_worker_limit": 1, "timeout_ms": 3000},
    }
    for key, value in overrides.items():
        cfg[key] = value
    path = os.path.join(tmp, "pipeline.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh)
    return path


def make_ports(tmp, base=24401):
    ports = {"host": "127.0.0.1", "port_range": [1024, 65535],
             "workers": {"w0": base, "w1": base + 1}}
    path = os.path.join(tmp, "ports.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(ports, fh)
    return path


class TmpDir:
    def __enter__(self):
        self.dir = tempfile.mkdtemp(prefix="t9-")
        return self.dir

    def __exit__(self, *exc):
        import shutil
        shutil.rmtree(self.dir, ignore_errors=True)
