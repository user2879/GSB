# 技术说明：多集群配置修订分发诊断台

## 1. 总体架构

```
samples/                本地确定性声明（生成脚本产出，无网络依赖）
  clusters.json         集群标识 / 127.0.0.1 端口 / 节点 / 整数样本种子
  tasks.json            任务标识 / 配置键集合 / 目标集群 / 聚合口径
  revisions.json        单调递增、显式父引用、可带派生引用的修订链
  revisions_amended.json  用于演示失效传播的修正案
  seed_params.json      固定种子与 NumPy 生成器参数
diagconsole/
  model.py              强校验与领域模型（E_SPEC，含字段定位）
  aggregate.py          NumPy 固定种子样本 + 整数定点分组聚合
  service.py            单集群 JSON 行模拟服务（子进程）
  orchestrator.py       Windows 子进程编排、握手、回执校验
  ledger.py             SQLite 结果节点账本（事务、幂等、崩溃日志）
  engine.py             失效传播 / 增量 / 权威重建 / 回滚 / 恢复 / 重放
  snapshot.py           会话快照、链版本指纹与内容摘要
  inject.py             三类异常注入
  ui.py / ui_main.py    Tkinter 增量界面（无头环境保护）
  cli.py / demo.py      命令行入口与可重复演示
data/                   运行期账本、快照、链状态（统一数据目录）
```

平台与模拟服务复用同一个 `aggregate.py`，因此“服务侧聚合”和“平台侧重算”
在相同输入下必然得到字节一致的聚合摘要。

## 2. 确定性聚合契约

- 样本行：`numpy.random.Generator(numpy.random.PCG64(seed))` 生成
  `sample_rows x 2` 的 `int64` 矩阵，取值域 `[0,1000)`。
- 分组：第一列模 4 落到 4 个固定分组。
- 口径：
  - 口径 1 `group_sum`：分组求和 × 策略因子 + 派生项；
  - 口径 2 `group_floor_mean`：整数地板均值 × 策略因子 + 派生项；
  - 口径 3 `group_weighted_sum`：分组求和 × 策略因子 × (1+配置标量) + 派生项。
- 策略因子：`canary=1, rolling=2, full=3`，全程整数运算，无浮点。
- 派生项：修订 `derives` 中上游版本号之和（稳定排序后求和）。
  派生引用因此进入确定性契约：派生变化会改变内容摘要并沿 DAG 传播失效。
- 内容摘要：对规范化 JSON（键排序、紧凑分隔、UTF-8）做 SHA-256。

固定 numpy==1.26.4 保证 PCG64 流在 Windows / Linux 上一致。

## 3. 失效传播模型

修订链是线性父链（parent 边），`derives` 额外构成只指向更早修订的 DAG 边。
两类边都规范化为“早版本 -> 晚版本”的前向边：

- parent 边：失效只在同一 `(任务, 集群)` 泳道内向下传播；
- derives 边：上游结论失效会传播到派生它的修订下全部目标节点（跨泳道）。

种子失效来源三类：配置取值变化、策略变化、派生引用变化。
BFS 按版本升序、任务稳定序、集群声明序输出失效集合与每步 `basis` 依据。
增量应用只对失效集合调用模拟服务重算；未失效节点仅在本地复算并与账本逐字节
比对（聚合 JSON 与内容摘要），任何非预期变化抛 `E_PREV`。

## 4. 账本事务与强制终止

- 节点主键 `(task_id, cluster_id, version)`，状态含 `committed/unknown`。
- 批次表记录 `staging/committed`；节点写入、父引用校验、根指针更新在同一
  SQLite 事务内提交，WAL 下提交前对其他读取者不可见。
- 重复写入：内容一致返回 `identical`；无替换授权的内容冲突抛 `E_LEDGER`；
  失效重算/回滚/恢复重跑显式带 `allow_replace` 才允许 `replaced`。
- 强制终止：在途节点以 `unknown` 崩溃日志行持久化，staging 批次保留。
  重启 `recover` 只依据已提交节点与根指针重建视图，unknown 节点要求重跑，
  且明确 `exit_report_available=false`（不伪造退出报告）。

## 5. 回滚与快照

- 回滚到历史修订 r：根指针、界面视图、统计口径整体切换到 r 的祖先链；
  链外节点保留为只读 isolated，不参与展示与统计，并稳定列出其摘要。
- 回滚返回隔离清单、重算集合，以及每个节点回滚前后内容摘要的逐字比较。
- 快照包含链指纹（全部修订规范化 JSON 的 SHA-256）、根指针、节点摘要、
  进度游标与整体快照摘要；重开时链指纹或快照摘要不符即 `E_SNAPSHOT`，
  执行整体替换而非增量合并。

## 6. 错误码映射

| 代码     | 触发场景 |
|----------|----------|
| E_SPEC   | 声明强校验失败（唯一性/回环字面量/单调版本/父链/派生/配置键覆盖），定位首个违规修订与字段 |
| E_PORT   | 回环端口占用或服务绑定失败，定位集群标识 |
| E_HANDSHAKE | ready 握手超时、服务中断、无回执，定位集群标识 |
| E_REPLY  | 回执字段缺失或身份/摘要不符，定位集群标识 |
| E_PREV   | 权威重建或增量比对的首个不一致节点（集群/任务/修订/字段） |
| E_REF    | 平台内派生/父引用悬空（加载后损坏），定位修订与引用 |
| E_SNAPSHOT | 快照摘要或链版本指纹不符 |
| E_REPLAY | 重放首个不可复现节点与字段 |
| E_LEDGER | 未授权的非幂等内容改写 |

## 7. 进程与端口

- 每个集群由 `python -m diagconsole.service` 独立子进程承载，Windows 下使用
  `CREATE_NEW_PROCESS_GROUP`；启动前做端口探测，等待 stdout 的 `ready` 行。
- JSON 行协议：每行一个请求/回执，连接随请求关闭；测试与演示退出时终止全部
  子进程并关闭句柄，端口通过声明值（样本）或系统临时空闲端口（单测）获得。
