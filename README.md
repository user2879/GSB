# 本地模型服务弹性伸缩决策组件

一个完全离线、确定性的推理服务集群弹性伸缩决策组件。所有结论都由纯整数算法和
SQLite 决策账本推导，交互快速路径与权威重算分离，并支持声明版本整体替换、
强制终止恢复、一致性重放和三类故障注入。无需 Docker、网络或外部账号。

## 1. 环境准备（Windows）

1. 安装 Python 3.11+（官方安装包），安装时勾选 **tcl/tk and IDLE**（提供 Tkinter）。
2. 确认环境：
   ```bat
   python --version
   python -m pip --version
   python -c "import tkinter; print(tkinter.TkVersion)"
   ```
3. 安装依赖（仅 NumPy；sqlite3、tkinter 为标准库/安装包组件）：
   ```bat
   python -m pip install -r requirements.txt
   ```
   无显示环境或缺 Tkinter 时，工作台自动进入无头（headless）模式，核心与测试不受影响。

## 2. 目录结构

```
elastic_scaler/
  errors.py          # E_SPEC/E_BUDGET/E_LOAD/E_VERSION/E_DIVERGE/E_REPLAY
  common.py          # 规范化 JSON、显式向上取整 ceil_div、定点格式
  spec.py            # JSON 声明加载与强校验（整体拒绝 + 字段路径）
  estimator.py       # 纯整数需求估算 + 统一预算裁剪提交
  ledger.py          # SQLite 幂等账本、进度游标、快照、只读版本视图
  engine.py          # 快速缓存/权威重算/版本替换/恢复/重放
  fault_injection.py # 负载缺口、非法预算、快照版本冲突注入
  stats.py           # 单调时钟延迟采样与整数定点均值/P95/最大值
  gui.py             # Tkinter 工作台（无 Tkinter 时无头降级）
  cli.py             # 命令行入口
  demo.py            # 三次确定性演示
samples/cluster_v1.json, cluster_v2.json
tests/               # 25 个 unittest 用例
run_demo.bat / run_tests.bat / run_gui.bat
```

## 3. 整数决策规则（无浮点参与）

对服务 `s`、分钟 `t`：

- 每副本分钟承载量（向下取整）：
  `capacity = floor(concurrency_limit * 60000 / request_ms)`
- 阈值后的有效承载量：
  `effective = max(1, floor(capacity * scale_threshold_pct / 100))`
- 未夹取需求（**显式整数向上取整**，`ceil_div(n,d)=(n+d-1)//d`）：
  `raw_demand = ceil_div(requests_t, effective)`
- 最小副本夹取 → 并发配额夹取（`quota // concurrency_limit`）→ 最大副本夹取。
- 统一裁剪阶段一次性提交：先按服务名字典序保留全部 `min_replicas`，
  剩余预算再按同一稳定顺序逐个满足额外需求；裁剪原因使用稳定代码
  `CLAMP_MIN_REPLICAS / CLAMP_MAX_REPLICAS / TRIM_CONCURRENCY_QUOTA /
  TRIM_GLOBAL_BUDGET / SATURATED`。
- 拒绝量：`rejected = max(0, requests - allocated * capacity)`，恒非负。

每次估算都在 `rules` 中显式记录所用整数除法与向上舍入规则；相同声明与分钟
重复估算结论逐字节一致（规范化 JSON 使用排序键、无空白分隔符）。

## 4. 快速路径与权威重算

- `query`：一次内存读取返回增量缓存结论，决策读取不触碰 SQLite。
- `authoritative` / `verify`：按账本中记录的声明版本与负载逐分钟重放
  `estimate + trim`，比较分配副本数、拒绝量、裁剪原因和估算中间项。
- 任一不一致以 `E_DIVERGE` 报告首个差异的分钟、服务、字段，并在
  `cache_basis_spec_version` 中说明缓存依据的账本版本。
- `tests/test_ledger_engine.py` 中的缓存破坏钩子演示了该错误的触发与定位。

## 5. 版本演进与整体替换

- 声明内容经规范化 SHA-256 摘要识别；修改后版本号单调递增，旧版本置
  `STALE`→`READ_ONLY`，旧视图保留只读，不参与当前汇总。
- 新版本推进时**丢弃整个缓存并整体替换当前视图**，不做逐分钟增量合并；
  替换报告稳定列出受影响服务、分钟区间和参数差异（见 `VersionReplacement`）。

## 6. 幂等账本与强制终止恢复

- 每个分钟在**单个 SQLite 事务**内写入分钟决策、服务占用、拒绝量、声明版本和
  进度游标；重复写同一分钟走 upsert，不产生重复行。
- 进程在提交前被杀死时，该分钟无任何部分记录；重启 `recover()` 只信任
  `COMMITTED` 分钟，下一未完成分钟标记为 `UNKNOWN` 并要求重跑，明确不生成
  强制终止时刻的退出报告。

## 7. 重放校验与故障注入

- `verify` 恢复后按记录版本重算全部已提交分钟，差异以 `E_REPLAY` 定位首个
  分钟/服务/字段并给出依据（`spec_version=...`）。
- CLI 注入（均写结构化 JSONL 日志，退出码 2）：
  ```bat
  python -m elastic_scaler.cli inject load-gap --spec samples\cluster_v1.json --service llm --minute 4
  python -m elastic_scaler.cli inject bad-budget --spec samples\cluster_v1.json --bad-budget 0
  python -m elastic_scaler.cli inject version-conflict --spec samples\cluster_v1.json
  ```
  分别报 `E_LOAD`、`E_BUDGET`、`E_VERSION`。

## 8. 工作台与延迟度量

- GUI 仅读引擎，刷新不改账本；展示每分钟每服务需求/分配、全局预算占用曲线、
  拒绝量与裁剪原因、当前版本与快照列表，渲染按服务名字典序稳定排列。
- 延迟使用单调时钟 `time.perf_counter_ns`，原始样本按微秒向下取整，
  **单样本量化误差 < 1 微秒**；均值/P95/最大值以毫秒三位定点字符串输出，
  P95 采用确定性最近秩法。

## 9. 一键验证

```bat
run_tests.bat
run_demo.bat
```

演示顺序固定为：冷启动推进 → 声明变更整体替换后推进 → 强制终止、恢复、重放。
产物：`data/demo/demo-summary.json` 与 `data/demo/events.jsonl`。

### 常用 CLI

```bat
python -m elastic_scaler.cli --db data\s.sqlite3 --log data\e.jsonl init --spec samples\cluster_v1.json
python -m elastic_scaler.cli --db data\s.sqlite3 --log data\e.jsonl advance
python -m elastic_scaler.cli --db data\s.sqlite3 --log data\e.jsonl query --minute 4
python -m elastic_scaler.cli --db data\s.sqlite3 --log data\e.jsonl verify
python -m elastic_scaler.cli --db data\s.sqlite3 --log data\e.jsonl replace-spec --spec samples\cluster_v2.json
python -m elastic_scaler.cli --db data\s.sqlite3 --log data\e.jsonl gui
```

## 10. 已记录的验证结果（当前环境）

- Python 3.11.0rc1 / NumPy 1.26.4 / sqlite3 3.37.2；本 Linux 沙箱无 Tkinter，
  工作台以无头模式验证，Windows 官方安装包勾选 tcl/tk 即可启动图形界面。
- `python -m unittest discover -s tests`：25 个用例全部通过。
- `python -m elastic_scaler.demo`：8 分钟冷启动、版本 2 整体替换、
  崩溃恢复后 8 分钟重放全部 `ok=true`。
- 样本高负载分钟同时出现配额裁剪、全局预算裁剪与非零拒绝量，且每分钟
  分配总和不超过全局预算、每服务不超过配额与最大副本数。
