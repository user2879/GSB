# fxrender 交付说明

面向“定点整数域多模态条件渲染 + 精确增量复用”的可集成 Python 包。原始需求见
`README.md`，设计与需求逐条映射见 `docs/TECHNICAL.md`。

## 目录结构

- `src/fxrender/`：包源码（校验、定点内核、v1/v2 格式、缓存账本、调度、快照、CLI）。
- `tests/`：48 个 unittest/pytest 用例，含信号终止子进程测试。
- `scripts/run_demo.py`：端到端确定性演示。
- `samples/bundle/`：固定种子生成的词表、提示、PNG、运动表、配置、编辑流。
- `vendor/tk/`：无 apt 权限下内置的 Tcl/Tk 运行时。
- `requirements.txt`、`pyproject.toml`：依赖与可编辑安装。

## 快速开始

```bash
python3 -m venv --without-pip .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install -e .
source .venv/bin/tk_env.sh        # 需要 tkinter 时
python scripts/run_demo.py
pytest -q
fxrender samples --out /tmp/fxs --frames 6
fxrender render --bundle /tmp/fxs --cache /tmp/fxc --export /tmp/fxo
```

## 验收对应

- 双版本往返：`tests/test_formats.py`
- 增量与全量逐帧一致、并发无关：`tests/test_pipeline_equivalence.py`
- 预算与待办语义：`tests/test_scheduler.py`、`tests/test_tier_and_crc.py`
- 快照恢复与强制终止：`tests/test_snapshot_interactive.py`、
  `tests/test_termination_recovery.py`
- 校验业务码与分块越界：`tests/test_loader_validation.py`、
  `tests/test_encoding_kernel.py`

实际验证结果记录于 `docs/VALIDATION.md`。
