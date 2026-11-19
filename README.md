# fwaudit — 固件镜像分区映射审计引擎

离线、确定性、纯标准库（运行期）的固件镜像与差分补丁审计工具，覆盖严格二进制
解析、分区映射重建、越界/跨槽位/写链检测、基镜像换代失效传播、资源预算、
原子检查点与 JSON/映射图报告。详细格式口径与验证记录见 `docs/VERIFICATION.md`。

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt

python -m fwaudit generate --samples-dir samples
python -m fwaudit audit \
  --images samples/base_v1.fwim \
  --patches samples/good.fwdf samples/bounds.fwdf samples/stale.fwdf \
  --reports-dir reports --workdir .fwaudit_state
python -m pytest tests/ -q
python scripts/acceptance_demo.py
```

## 模块结构

- `fwaudit/image.py` — 镜像头/分区表严格解析、重叠与自引用检测
- `fwaudit/patch.py` — 补丁解析、边界/槽位/写链检测、预算与降级
- `fwaudit/engine.py` — 代次水位、失效传播、并发与批隔离
- `fwaudit/storage.py` — 原子写、跨平台文件锁、检查点演进
- `fwaudit/generator.py` — 确定性合法/畸形样本与真值同步标记
- `fwaudit/verify.py` — 仅依据镜像内真值标记的独立验证端
- `fwaudit/report.py` — JSON 报告与分区映射图（JSON + 文本）
- `fwaudit/cli.py` — 命令行入口，退出码 0/1/2
- `tests/` — 43 个 pytest 用例，断言业务码、出错偏移、覆盖率与退出码

## 退出码

- `0` 全部成功
- `1` 部分失败（单补丁失败不影响同批其他补丁）
- `2` 参数错误

## 关键预算参数（CLI 可配）

`--image-byte-threshold`、`--op-count-threshold`、`--chunk-size`、
`--max-ops`、`--timeout`、`--concurrency`。
