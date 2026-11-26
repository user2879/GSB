"""确定性样本生成：请求清单、三格式历史回执与异常样本。

全部内容按固定整数种子生成，运行前无需任何外部数据附件。
历史回执使用旧参数时代的工作进程种子（当前种子 + 9000）计算摘要，
模拟磁盘上并存的三版历史格式。
"""
import argparse
import json
import os

from .core import compute_batch, gen_requests, chunk_batches, load_config

OLD_SEED_OFFSET = 9000


def _old_cfg(cfg):
    old = dict(cfg)
    old["workers"] = [dict(w, seed=w["seed"] + OLD_SEED_OFFSET) for w in cfg["workers"]]
    old["workers_by_id"] = {w["worker_id"]: w for w in old["workers"]}
    return old


def generate(config_path, out_dir):
    cfg = load_config(config_path)
    os.makedirs(out_dir, exist_ok=True)
    requests = gen_requests(cfg["sample_seed"], cfg["request_count"])
    with open(os.path.join(out_dir, "requests.json"), "w", encoding="utf-8") as fh:
        json.dump({"sample_seed": cfg["sample_seed"],
                   "request_ids": [r.request_id for r in requests]}, fh,
                  ensure_ascii=False, indent=2)

    batches = chunk_batches(requests, cfg["batch_size"])
    old = _old_cfg(cfg)
    worker_ids = sorted(cfg["workers_by_id"])

    def old_digest(batch_idx, batch_id, rids, param_version):
        wid = worker_ids[batch_idx % len(worker_ids)]
        digest, _ = compute_batch(old, wid, param_version, batch_id, rids)
        return wid, digest

    r1_lines = []
    for batch_idx, (bid, rids) in enumerate(batches[:2]):
        _, digest = old_digest(batch_idx, bid, rids, 1)
        for rid in sorted(rids):
            r1_lines.append(f"r1|{bid}|{rid}|{digest}")
    _write(out_dir, "receipts_r1.txt", r1_lines)

    r2_lines = []
    for batch_idx, (bid, rids) in enumerate(batches[2:4], start=2):
        wid, digest = old_digest(batch_idx, bid, rids, 1)
        for i, rid in enumerate(sorted(rids)):
            r2_lines.append(f"r2|{bid}|{rid}|1|{17 + i}|{wid}|{digest}")
    _write(out_dir, "receipts_r2.txt", r2_lines)

    r3_lines = []
    if len(batches) >= 5:
        batch_idx = 4
        bid, rids = batches[batch_idx]
        wid, digest = old_digest(batch_idx, bid, rids, 1)
        ordered = sorted(rids)
        for rank, rid in enumerate(ordered):
            r3_lines.append(f"r3|{bid}|{rid}|1|23|{wid}|{rank}||{digest}")
        dup_rid = ordered[0]
        r3_lines.append(f"r3|{bid}|{dup_rid}|1|24|{wid}|{len(ordered)}|{dup_rid}|{digest}")
    _write(out_dir, "receipts_r3.txt", r3_lines)

    bid, rids = batches[0]
    rid0, rid1 = sorted(rids)[0], sorted(rids)[1]
    digest0 = "a" * 64
    _write(out_dir, "bad_field.txt", [
        f"r2|{bid}|{rid0}|1|12|w0|{digest0}",
        f"r2|{bid}|{rid1}|1|-3|w0|{digest0}",
    ])
    _write(out_dir, "bad_ref.txt", [
        f"r3|{bid}|{rid0}|1|12|w0|0||{digest0}",
        f"r3|{bid}|{rid1}|1|12|w0|0||{digest0}",
    ])
    _write(out_dir, "bad_ref_dangling.txt", [
        f"r3|{bid}|{rid0}|1|12|w0|0|req-9999|{digest0}",
        f"r3|{bid}|{rid1}|1|12|w0|1||{digest0}",
    ])
    return sorted(os.listdir(out_dir))


def _write(out_dir, name, lines):
    with open(os.path.join(out_dir, name), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="samplegen")
    parser.add_argument("--config", default="config/pipeline.json")
    parser.add_argument("--out", default="samples")
    args = parser.parse_args(argv)
    files = generate(args.config, args.out)
    print(json.dumps({"generated": files}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
