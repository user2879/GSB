# cfgpipe — 跨组件配置分发增量回执与派生失效收敛验证管线

面向 Windows 本地环境（同时兼容 Linux/macOS）的纯 Python 实现：平台管理若干
本地模拟组件节点（独立子进程，127.0.0.1 回环 TCP + JSON 行协议），对声明的
配置项执行定点量化编译、变更集驱动的增量下发、回执收敛判定、SQLite 账本
持久化与崩溃恢复重放校验。

## 环境准备

- Python 3.11+（Windows 官方安装包需勾选 tcl/tk 以使用 Tkinter 界面）
- `python -m pip install numpy`（或 `python -m pip install -r requirements.txt`）
- 空工作目录（声明、账本、快照、节点日志均在运行目录下生成）
- 127.0.0.1 上有足够空闲高位端口（演示脚本自动探测空闲端口）

## 快速复现

```bat
python scripts\run_demo.py --workdir demo_workspace
python scripts\verify_full_rebuild.py --workdir demo_workspace
python -m unittest discover -s tests
python -m cfgpipe.gui --workdir demo_workspace
```

顺序执行前两条即可复现全部结论；演示脚本任一检查失败即以非零码退出。

## 目录结构

```
cfgpipe/            领域组件包（可集成）
  spec.py           声明/变更集加载与强校验（E_SPEC）、世界状态
  quantize.py       定点量化口径、截断/饱和规则、误差界三段拆分
  graph.py          派生依赖图、稳定拓扑序、失效传播
  compiler.py       规范化产物编译（版本 = 内容 SHA-256）
  sample.py         确定性样本生成与整数定点分组聚合（平台/节点共用）
  node.py           节点子进程（JSON 行协议服务器）
  scheduler.py      全局+每节点双重配额的并行调度、真实超时与退避重试
  converge.py       增量回执收敛判定（E_CONVERGE）
  ledger.py         SQLite 账本：批次/回执/失效轨迹/误差界/游标/inflight
  replay.py         恢复后一致性重放校验（E_REPLAY）
  pipeline.py       编排器：批次执行、快照、恢复续跑、全量重建比对
  crash_batch.py    强制终止模拟（inflight 后 os._exit，不写退出报告）
  gui.py            Tkinter 诊断界面（只读）
scripts/
  run_demo.py             端到端演示（含崩溃恢复），失败即非零退出
  verify_full_rebuild.py  独立校验：增量结果 vs 从零全量重建逐项比对
examples/           声明与变更集样例（端口为占位，演示脚本自动分配空闲端口）
tests/              48 个单元/集成测试
```

## 声明与变更集格式

见 `examples/declaration.json`：会话参数（`scale_exp` 0..6、分组数、全局/每节点
并发、超时与退避序列）、配置项（键、值、版本、过期毫秒、派生引用与系数）、
节点注册表（标识、`127.0.0.1` 字面量、端口、订阅、样本种子与规模）。
变更集见 `examples/changesets/`：单调递增版本号 + 逐项变更（取值、过期毫秒、
派生系数、`subscribers_add`/`subscribers_remove`）。

校验规则（任一违反即以 `E_SPEC` 整体拒绝，并稳定报告首个违规键与字段）：
键名合法唯一；取值/系数可量化为定点整数；声明内版本号按声明顺序严格递增，
变更集版本号彼此严格递增且不小于声明最大版本；过期毫秒为正整数；派生目标
存在且依赖图无环（报告环上字典序最小键）；节点标识唯一、host 必须为回环
字面量 `127.0.0.1`、端口 1024..65535；订阅引用必须存在。

## 定点量化口径（显式规则）

- 缩放因子 `scale = 10 ** scale_exp`，`scale_exp ∈ [0, 6]`。
- 量化 `q(x) = trunc_toward_zero(x * scale)`，**截断方向恒为向零**。
- **溢出饱和规则**：所有定点运算饱和到 int64（`sat(v)`），加法用 `sat_add`。
- 派生值 `sat(trunc(coeff_q * parent_q / scale))`；分组聚合、阈值比较全部整数。
- 耗时为单调时钟毫秒整数（`time.monotonic_ns() // 1_000_000`）。
- 逐配置项误差界（单位 ulp = 1/scale），累计误差界三段拆分：
  - 量化截断 `trunc = 1`（本项自身量化）；
  - 派生放大 `derived_i = trunc(|coeff_q_i| * chain_total_parent / scale)`；
  - 聚合合成 `agg = max(ceil(sample_size / groups))`（取所有订阅节点最大值，
    每个加项在整除处至多再引入 1 ulp）。
  - `total = trunc + derived + agg`，逐批次写入账本 `error_bounds` 表。

## 失效传播与增量下发

- 内容变化（取值/过期/派生系数）使该项失效并沿派生边传播到全部下游派生项；
  订阅集合变化只影响新增/移除的订阅节点，**不改变项内容与版本号**，既有
  订阅者的产物逐字节不变。
- 失效集合按稳定拓扑序（Kahn + 堆）输出，每步记录传播依据；全部排序基于
  键名字典序，不依赖 dict/set 遍历顺序——同一变更集在任意声明条目顺序下
  输出逐字节一致（`tests/test_invalidation.py` 覆盖）。
- 产物为规范化 JSON（键排序、无空白），产物版本 = 内容 SHA-256，因此未受
  影响项的产物与版本天然逐字节不变；增量批次只对受影响项重新编译并只向
  受影响节点下发（`upsert` 仅含受影响键，退订键走 `remove`）。

## 调度、超时与重试

- 双重配额：信号量限制全局在途数与单节点在途数；配额耗尽时任务排队，
  记录等待起止的单调时钟毫秒（`wait_start_ms`/`wait_end_ms`）。
- 真实套接字超时（`dispatch_timeout_ms`）；超时/连接失败按声明退避序列
  重试，重试发送的线路字节逐字节不变（版本与切片内容不变，
  `tests/test_scheduler.py::test_retry_keeps_payload_bytes` 验证）。

## 回执收敛（E_CONVERGE）

仅对存在回执变化的节点逐字节比对：回执摘要、逐键聚合摘要、口径版本
（`scale_exp`）与配置版本；未受影响节点标记 `carried` 并保留依据版本，
同时用当前世界状态重算其期望摘要验证未漂移。缺失回执、口径版本不同、
摘要不一致均以 `E_CONVERGE` 报告，首个偏差稳定定位到节点/配置键/字段。

## 账本、恢复与重放

- 每批次在**同一 SQLite 事务**写入：批次记录、回执、失效轨迹、误差界、
  进度游标，并清除 inflight 标记；批次开始前先在自动提交事务写 inflight。
- 账本 `meta` 携带整数格式版本（当前 1）与内容摘要（声明文件 SHA-256）；
  重新打开时任一不一致即**整体替换**账本文件，不做增量合并。
- 强制终止后重启：仅依据已完整提交的记录恢复；残留 inflight 批次中的节点
  标记为 `unknown` 并要求重跑；被终止的进程不生成、也不被声称能生成退出
  报告（`cfgpipe/crash_batch.py` 以 `os._exit` 模拟）。
- 恢复后 `replay.py` 依据声明与变更集重新推导每个已提交批次的失效集合、
  误差界与逐节点期望回执，同账本逐项比较；首个不一致以 `E_REPLAY` 定位到
  节点、配置键与字段。

## 错误码

| 代码 | 含义 |
| --- | --- |
| `E_SPEC` | 声明/变更集校验失败（稳定列出首个违规键与字段） |
| `E_CONVERGE` | 回执收敛失败（缺失回执/口径版本不同/摘要不一致/沿用漂移） |
| `E_REPLAY` | 恢复后重放校验与账本不一致 |
| `E_DISPATCH` | 重试耗尽后下发失败 |

## 实际验证结果（本机，Python 3.11，numpy 2.2.6）

`python scripts/run_demo.py --workdir demo_workspace` 全部通过：

- 变更集 v10（alpha 取值）：失效 `{alpha,beta,delta}` + 节点 `{n1,n2}`；
  仅下发 `n1:[alpha,beta,delta]`、`n2:[beta,delta]`；未受影响项
  `gamma/epsilon` 产物逐字节不变；`n3` 标记 `carried`。
- 变更集 v11（epsilon 取值 + gamma 仅订阅变化）：失效项仅 `epsilon`；
  `gamma` 产物逐字节不变（订阅变化不提升版本号），仅向新订阅者 `n1`
  补发；`n2` 沿用上一轮回执（`carried`，依据版本保留）。
- 强制终止模拟：子进程非零退出且无退出报告；恢复时 `n1,n3` 标记
  `unknown`，清除 inflight 后续跑并提交 `batch_0002`。
- 重放校验通过；`verify_full_rebuild.py` 确认 3 个已提交批次的全部快照
  字节与账本摘要同从零全量重建逐项一致。
- 账本误差界示例（batch_0001）：`alpha (1,0,50,51)`、
  `beta (1,2,50,53)`、`delta (1,1,50,52)`（trunc/derived/agg/total，ulp）。
- `python -m unittest discover -s tests`：48 个测试全部通过。

注：Tkinter 界面需本机装有 tcl/tk（Windows 官方安装包默认附带）；界面为
只读视图，刷新仅重读文件，不写账本。
