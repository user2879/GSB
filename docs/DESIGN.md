# 跨组件配置分发实验台 — 技术说明

## 1. 架构与进程模型

- 平台进程 `configbench.platform.Platform` 编排若干节点子进程。
- 每个节点由 `python -m configbench.node_main <spec.json>` 启动，绑定注册表
  声明的唯一 `127.0.0.1` 高位端口，平台与节点之间使用回环 TCP 上的
  JSON 行（每行一个 JSON、`\n` 分隔）协议。
- 平台不内嵌任何业务结果真值：事实来源是 SQLite 账本 `configbench/ledger.py`，
  界面缓存可随时由账本重建。

模块职责：

| 模块 | 职责 |
| --- | --- |
| `errors.py` | E_MANIFEST/E_PORT/E_HANDSHAKE/E_FIELD/E_PREVIEW/E_LATENCY/E_REPLAY/E_LEDGER 等稳定错误码 |
| `fixedpoint.py` | int64 饱和、向零截断、定点均值/最近秩 P95/最大值、规范化 JSON 字节 |
| `manifest.py` | 注册表/版本清单/样本声明加载与固定顺序强校验 |
| `samples.py` | 固定种子 PCG64 确定性样本生成与定点重缩放 |
| `aggregate.py` | 整数定点分组聚合（节点与权威重算共用同一纯函数） |
| `compiler.py` | 内容寻址增量编译、派生失效传播、节点切片（含派生依赖闭包） |
| `scheduler.py` | 全局/每节点并发配额、排队时钟、尝试序号与结局分类 |
| `transport.py` | 子进程编排、真实 socket 超时、握手/回执字段校验 |
| `node_main.py` | 节点子进程协议实现与 port/handshake/field/flaky 故障注入 |
| `ledger.py` | 事件溯源、格式版本、链式内容摘要、损坏整体替换与只读备份 |
| `platform.py` | 发布、双路径一致性、延迟预算、强制终止恢复与重放 |
| `ui.py` | Tkinter 视图；无 tkinter/无显示时降级为终端视图 |
| `cli.py` | validate/run/query/recompute/latency/recover 命令 |

## 2. 定点口径（显式声明，摘要内回传）

- 表示：`v = raw / 10**scale`，`raw` 为有符号 64 位整数。
- 截断方向：所有除法/重缩放一律**向零截断**（`trunc_div`），区别于 Python
  `//` 的向负无穷。
- 溢出：先用 Python 任意精度整数计算，再 `sat_int64` 饱和到
  `[-2^63, 2^63-1]`，杜绝 NumPy 静默回绕。
- 摘要逐字节一致：`fixedpoint.canonical_bytes` 使用排序键、固定分隔符、
  无空白的 UTF-8 JSON；同切片+同种子任意次运行得到相同 SHA256。
- 延迟统计缩放因子 `LATENCY_SCALE=1000`（raw = 毫秒整数 × 1000），向零
  截断，单次量化误差上界 `1/1000` 毫秒；P95 用最近秩
  `ceil(0.95n)`，无插值、确定性。

## 3. 强校验（E_MANIFEST，整体拒绝）

按固定顺序检查并在**首个违规**处抛出，错误体带 `field` 与 `basis`：
版本号唯一且严格单调递增；键名正则；数值类型仅 `int|fixed`；目标节点必须
注册且端口全局不重复（仅允许 127.0.0.1）；派生引用必须存在且无环（稳定
Kahn 拓扑，候选集始终排序，遍历不依赖字典/哈希顺序）。

## 4. 增量编译与失效传播

- 编译产物为内容寻址对象，`artifact_id = sha256(canonical(content))`；
  未变化项跨版本的内容与 id 逐字节不变。
- 直接变化口径：键值、目标节点集合、派生系数（见 `_diff_items`）。
- 传播：沿派生依赖边求下游闭包，并汇总受影响节点；输出稳定拓扑序的
  `invalidated` 与每步 `{trigger_source, affected_keys, affected_nodes}`。
- 节点切片包含直接目标项及其派生依赖闭包（聚合派生项需要父项）。
- 发布时仅对受影响节点（首版本为全部节点）生成下发任务。

## 5. 调度、超时与重试

- 信号量 + 条件变量保证：在运行任务 ≤ `max_global`，单节点在运行
  ≤ `per_node_quota`；配额耗尽即排队，记录 `wait_started_ms/wait_ended_ms/
  wait_queued_ms`（单调时钟）。
- 单次下发设置真实 connect/handshake/ack 超时；结局分类
  `SUCCESS/TIMEOUT/CONN_REFUSED/HANDSHAKE/FIELD/EXHAUSTED/UNKNOWN`。
- 仅 `TIMEOUT/CONN_REFUSED` 按策略重试；每次尝试带 1 基序号，重试使用同一
  `task_id` 与同一切片，节点聚合幂等，故不改变单次下发语义。
- 端口占用 E_PORT、握手超时 E_HANDSHAKE、回执缺字段 E_FIELD 分别上报，
  失败节点不会出现在成功回执统计中。

## 6. 双路径一致性

- 快速路径：`query_preview` 只读内存增量缓存（版本、摘要、进度）。
- 权威路径：`authoritative_recompute` 从账本折叠状态并调用同一
  `aggregate_slice` 纯函数重算。
- `compare_paths` 逐节点比较版本、规范化摘要（逐字节）、耗时（误差界
  `timing_tolerance_ms`，默认 0），首个不一致即以 E_PREVIEW 定位
  节点、版本与字段。

## 7. 延迟预算

时钟基准 `time.monotonic()` 整数毫秒；真值来源是账本 `latency` 事件。
每次查询/重算都落账，超预算额外写 `latency_violation`（E_LATENCY，含请求
标识、实测值、预算），请求绝不静默剔除；`latency_report` 给出整体与按节点
的均值/P95/最大值定点表示、缩放因子、截断方向与量化误差上界。

## 8. 账本、强制终止与恢复

- SQLite 事件溯源，事件含 `seq/ts_ms/kind/payload/digest/prev_digest`，
  链式 SHA256；整数 `LEDGER_FORMAT_VERSION` 存于 meta。
- 每条事件独立事务完整提交。强制 kill 时未提交事务随进程消失：恢复时
  游标声明但无 `task_outcome` 的任务标记为 UNKNOWN 并要求重跑
  （`rerun_unknown_tasks`），恢复流程不生成任何退出报告。
- 打开账本识别格式版本并校验摘要链：不一致则把旧账本复制为
  `*.bak-时间戳` 只读备份后整体替换，绝不增量合并。
- 恢复后对已完成任务逐条重放比对，首个不一致以 E_REPLAY 定位任务与字段。

## 9. Windows 适配

- 子进程带 `CREATE_NO_WINDOW` 标志，不弹控制台窗。
- 仅使用标准库 + NumPy；端口绑定、`subprocess`、`sqlite3`、`tkinter` 均为
  Windows 10/11 + Python 3.11 原生能力。只读备份在 NTFS 上通过去掉写位实现。
- 当前验证环境为无头 Linux（无 tkinter），UI 模块在该环境导入不失败，
  图形窗口不可用时以 `E_UI_UNAVAILABLE` 提示并可用终端视图替代。
