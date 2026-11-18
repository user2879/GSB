# 流式窗口聚合中间件（SWA）

面向 Windows 本地环境的确定性流式窗口聚合中间件。仅依赖 Python 3.11+ 标准库
（`sqlite3`、`json`、`threading`、`concurrent.futures`、`hashlib`、`argparse` 等），
无第三方包、无容器、无网络访问。

## 1. 目录结构

```
src/swa/
  errors.py       业务码与结构化异常（E_SPEC/E_EVENT/E_LATE/E_LEDGER/E_BUDGET/E_REPLAY）
  spec.py         声明加载与强校验（含字段路径、重复 JSON 键、别名歧义）
  generator.py    种子驱动的确定性事件生成、入口校验、轮询投递计划
  scheduler.py    三重预算并行分片调度（短任务、非阻塞协调者）
  engine.py       事件总线、水位推进、窗口对齐、乱序 E_LATE、确定性结果
  ledger.py       SQLite 账本、规则版本谱系、幂等提交、陈旧标记
  replay.py       按账本规则版本重放并逐窗口比对
  runner.py       端到端编排、异常注入、强制终止、整体替换
  logging_setup.py 结构化 JSONL 日志
  cli.py          命令行入口（run / replay / view）
samples/          本地确定性声明、样本覆盖与异常注入参数
tests/            32 个单元与端到端测试
scripts/acceptance.py  一键验收驱动
run-acceptance.bat     Windows 验收入口
```

## 2. 数据模型与确定性

- 事件：`source_id / source_seq / timestamp_ms(整数毫秒) / group_key / value(整数)`。
- 每个事件源用独立 `random.Random("swa:v1:<source>:<seed>")`，按固定顺序消耗
  随机数（时间间隔→分组键→值），源内时间戳严格单调递增。
- 相同 `(seed, event_count)` 多次生成得到逐字节一致的事件流，事件摘要为
  规范化 JSON 的 SHA-256（见 `generator.event_stream_digest`）。
- 聚合仅含 `count / sum / max`，均与到达顺序无关；窗口结果按
  `(window_start_ms, rule_id)` 与分组键稳定排序后用规范化 JSON 序列化，
  因此任意调度交错、任意分组到达顺序都产出相同的窗口摘要与结果摘要。

## 3. 三重预算下的就绪顺序与窗口关闭

调度器（`scheduler.py`）采用“协调者非阻塞 + 工作线程短任务”模型，从结构上避免死锁：

- 分片 = `(规则, 窗口, 分组键)` 的聚合状态，常驻协调端字典。
- 全局并发上限与每规则配额是有界信号量，仅在短任务执行期间持有，处理完立即释放；
  阻塞任务排队等待而不占用线程，不会饿死可运行任务。
- 每窗口内存项预算是协调端准入状态：新分组在预算满时进入“停车”（back-pressure），
  记录单调时钟等待起止；绝不通过淘汰已有分组放行。持续超过阈值抛 `E_BUDGET`。
- 仍有停车事件的窗口不允许关闭；EOF 是最终标点，水位推进到极大哨兵值，
  先强制放行停车事件（不丢分组），再统一关闭所有窗口。

窗口关闭只由事件时间戳水位决定，与注入顺序无关：全局水位取各源水位最小值，
当 `window_end_ms <= 全局水位` 时窗口到期；早于已关闭窗口的事件登记为 `E_LATE`，
不改动任何已关闭结果。

## 4. 账本、恢复与规则演进

- 账本携带整数 `format_version`（当前 1），不支持的版本在打开前即以 `E_LEDGER` 拒绝。
- 每个窗口在单条 SQLite 事务内提交；`UNIQUE(rule_id, rule_version, window_start_ms)`
  与部分唯一索引 `uq_window_current` 保证同一窗口不会出现两条当前结果，重复提交幂等。
- 强制终止（`os._exit(137)`）只保留已完整提交的窗口；未完成 run 永久标记 `running`，
  未提交窗口视为未知并要求重跑，重启后复用已提交记录、补齐缺失窗口，不产生重复。
- 规则参数变更后版本号单调递增；旧版本窗口标记 `stale` 且不进入当前视图，
  当前视图整体替换并稳定列出受影响规则与窗口标识。
- 同版本号但签名（窗口秒数/聚合函数/预算等）不一致时以 `E_REPLAY` 报明规则与依据；
  `replay` 子命令按账本记录的规则版本重算并逐窗口比对内容摘要。

## 5. 命令行用法

```bash
# 干净运行（全局分片上限 3）
python -m swa run --spec samples/spec.json \
  --ledger out/run.sqlite --result out/run.json --log out/run.log \
  --global-limit 3 --report out/report.json

# 异常注入（仅改输入文件或启动参数，不改源码）
#   E_SPEC   声明文件损坏
#   E_LEDGER 账本格式版本不受支持
#   E_BUDGET 内存项预算持续耗尽
python -m swa run --spec samples/spec-corrupt.json --inject corrupt-spec ...
python -m swa run --spec samples/spec.json --inject unsupported-ledger-version ...
python -m swa run --spec samples/spec-tight-budget.json --inject budget-exhausted ...

# 强制终止后恢复 / 重放校验 / 当前视图
python -m swa run --spec samples/spec.json --crash-after-commits 3 ...
python -m swa replay --spec samples/spec.json --ledger out/run.sqlite --log out/r.log
python -m swa view --ledger out/run.sqlite
```

业务码退出码：`E_SPEC=2 E_EVENT=3 E_LATE=4 E_LEDGER=5 E_BUDGET=6 E_REPLAY=7`。
日志为 JSONL，每条含 UTC 时间、单调时钟毫秒、级别、事件、业务码、影响规则与窗口边界。

## 6. 实际验证结果（本机 Python 3.11.0rc1，SQLite 3.37.2）

- `python -m unittest discover -s tests`：32/32 通过（spec 10、generator 3、ledger 7、engine 6、cli 6）。
- `python scripts/acceptance.py`：全部 PASS，覆盖
  - 全局上限 1/2/3/5 下结果逐字节一致（确定性）；
  - 全局/每规则/内存项三项峰值均不超预算；
  - `os._exit(137)` 硬终止后恢复，窗口无重复，结果字节与干净运行一致；
  - replay 逐窗口校验通过；规则 v1→v2 整体替换并标记陈旧；同版本漂移报 E_REPLAY；
  - E_SPEC / E_LEDGER / E_BUDGET 三类注入路径与结构化日志可区分归属。
- Windows 上运行 `run-acceptance.bat` 即可复现；所有路径均使用 `os.path`，无 POSIX 专有依赖。
