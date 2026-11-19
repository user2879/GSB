# 消息中间件实验工作台（本地确定性）

对本地 NDJSON 确定性事件流做分区聚合、整数毫秒水位定稿、定点计算与交互延迟观测。
不实现消息投递语义、不依赖外部服务、不使用容器。技术细节见
[`TECH_DESIGN.md`](TECH_DESIGN.md)，Windows 运行步骤见
[`docs-windows.md`](docs-windows.md)。

## 环境

- Python 3.11+；`pip install numpy`（见 `requirements.txt`）。
- tkinter、sqlite3 随标准 Python 提供；无 tkinter 环境自动降级为纯无头模式。

## 快速开始

```bash
python -m unittest discover -s tests -v
python -m bench.acceptance --workdir run/acceptance \
  --json-out artifacts/acceptance_summary.json
```

## 主要 CLI

```bash
python -m bench.mw.cli --workdir run init     --spec samples/basic/spec.json --events samples/basic/events.ndjson
python -m bench.mw.cli --workdir run ingest   --spec samples/basic/spec.json --events samples/basic/events.ndjson --drain
python -m bench.mw.cli --workdir run query    --spec samples/basic/spec.json --inject-delay-ms 80
python -m bench.mw.cli --workdir run recompute --spec samples/basic/spec.json   # 缓存 vs 权威重放逐字节比对
python -m bench.mw.cli --workdir run latency  --spec samples/basic/spec.json
python -m bench.mw.cli --workdir run snapshot --spec samples/basic/spec.json
python -m bench.mw.cli --workdir run inject   --spec samples/resume/spec.json --events samples/resume/events_part2.ndjson --kind regression --partition p0 --seq 2 --ts-ms 10
python -m bench.mw.cli --workdir run gui
```

## 目录

- `bench/mw/`：核心实现（校验、引擎、账本、快照、恢复、注入、延迟、GUI、CLI）。
- `samples/`：确定性样本与预期聚合（basic、resume、invalid）。
- `tests/`：31 个 unittest 用例，覆盖 8 项业务要求。
- `artifacts/`：验收结构化总结。
- `run/`：运行时账本、快照、结构化日志（可由 `--workdir`/独立参数指定）。

## 错误码

`E_SPEC`(2) `E_WATER`(3) `E_VIEW`(4) `E_ORDER`(5) `E_SNAPSHOT`(6) `E_REPLAY`(7)；
`E_LATE`、`E_LAT` 为账本/日志中的观测记录而非抛出异常。
