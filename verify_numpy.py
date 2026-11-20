"""使用 NumPy 对三种编码做批量编解码校验（确定性样本，无网络）。"""
from __future__ import annotations

import numpy as np

from colbench import codecs


def main() -> int:
    rng = np.random.default_rng(20261121)
    int_values = rng.integers(-1000, 1000, size=500, dtype=np.int64).tolist()
    str_values = [f"tag-{int(v) % 7}" for v in
                  rng.integers(0, 100, size=300)]
    rle_values = [5] * 50 + [8] * 30 + [5] * 20

    cases = {"numpy_int": int_values, "numpy_string": str_values,
             "rle_repeat": rle_values}
    for name, values in cases.items():
        for encoding in codecs.ENCODINGS:
            block_a = codecs.encode_block(values, encoding, compressed=True)
            block_b = codecs.encode_block(list(values), encoding,
                                          compressed=True)
            assert block_a.payload == block_b.payload, \
                f"{name}/{encoding} 非确定性"
            decoded = codecs.decode_block(
                block_a.payload, expect_encoding=encoding,
                expect_digest=block_a.digest)
            assert decoded == values, f"{name}/{encoding} 往返不一致"
            assert codecs.block_summary(decoded) == \
                codecs.block_summary(values)
            print(f"OK {name:<13} {encoding:<4} rows={len(values)} "
                  f"pre={block_a.pre_bytes} post={block_a.post_bytes}")
    print("NumPy 编解码校验全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
