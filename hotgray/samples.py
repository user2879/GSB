"""Deterministic generation of gateway declaration sample files."""

import argparse
import json
import os


def build_sample_declaration(base_port=18000):
    backends = [
        {"backend_id": "stable-a", "model_version": "mlp-stable-1",
         "host": "127.0.0.1", "port": base_port, "model_dim": 8,
         "hidden_dim": 6, "seed": 101, "quant_version": 1},
        {"backend_id": "gray-a", "model_version": "mlp-gray-1",
         "host": "127.0.0.1", "port": base_port + 1, "model_dim": 8,
         "hidden_dim": 6, "seed": 202, "quant_version": 1},
        {"backend_id": "stable-b", "model_version": "mlp-stable-b-1",
         "host": "127.0.0.1", "port": base_port + 2, "model_dim": 6,
         "hidden_dim": 4, "seed": 303, "quant_version": 1},
        {"backend_id": "gray-b", "model_version": "mlp-gray-b-1",
         "host": "127.0.0.1", "port": base_port + 3, "model_dim": 6,
         "hidden_dim": 4, "seed": 404, "quant_version": 1},
    ]
    rules = [
        {"rule_id": "rule-chat", "prefix": "chat/",
         "stable_backend": "stable-a", "gray_backend": "gray-a",
         "gray_weight_bp": 4000, "batch_size": 4},
        {"rule_id": "rule-vision", "prefix": "vision/",
         "stable_backend": "stable-b", "gray_backend": "gray-b",
         "gray_weight_bp": 2500, "batch_size": 3},
    ]
    clients = [f"client-{i:02d}" for i in range(10)]
    stream = []
    for idx in range(24):
        rule = rules[idx % len(rules)]
        stream.append({
            "sample_id": f"sample-{idx:03d}",
            "rule_id": rule["rule_id"],
            "client_key": clients[idx % len(clients)],
        })
    return {"backends": backends, "rules": rules, "request_stream": stream}


def build_invalid_declaration():
    data = build_sample_declaration()
    data["backends"][1]["port"] = 80
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description="generate deterministic samples")
    parser.add_argument("--out-dir", default="samples")
    parser.add_argument("--base-port", type=int, default=18000)
    args = parser.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    valid_path = os.path.join(args.out_dir, "gateway_declaration.json")
    invalid_path = os.path.join(args.out_dir, "invalid_declaration.json")
    with open(valid_path, "w", encoding="utf-8") as handle:
        json.dump(build_sample_declaration(args.base_port), handle,
                  ensure_ascii=False, indent=2)
    with open(invalid_path, "w", encoding="utf-8") as handle:
        json.dump(build_invalid_declaration(), handle,
                  ensure_ascii=False, indent=2)
    print(valid_path)
    print(invalid_path)


if __name__ == "__main__":
    main()
