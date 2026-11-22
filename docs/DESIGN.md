# 配置分发结果验证与差异回归分析平台 — 技术说明

## 1. 总体架构

平台仅依赖 Python 3.10+ 标准库与 NumPy，全部通信走 `127.0.0.1` 高位
端口的 TCP JSON 行协议，持久化只用标准库 `sqlite3`，无容器、无外部服务。

| 模块 | 职责 | 对应需求 |
| --- | --- | --- |
| `configreg/manifest.py` | 清单/注册表数据模型与强校验、无环检查、键集合差异 | 需求 1 |
| `configreg/canon.py` | 规范化字节序列化、SHA-256 摘要、稳定拓扑/BFS | 需求 2、7 |
| `configreg/fixedpoint.py` | 定点量化、整数乘加、截断/饱和规则、误差界 | 需求 3 |
| `configreg/diff.py` | 差异分类、派生链影响范围、不可区分判定 | 需求 2、3 |
| `configreg/node.py` | 节点子进程、JSON 行服务、NumPy 确定性分组聚合、故障注入 | 需求 4、6 |
| `configreg/nodepool.py` | 子进程生命周期、单次采集客户端与真实套接字超时 | 需求 4、6 |
| `configreg/scheduler.py` | 全局/每节点并发配额、排队计时、重试隔离 | 需求 6 |
| `configreg/collect.py` | 任务计划、按配置项原子分批、观测配对 | 需求 4、6、8 |
| `configreg/regression.py` | 预算判定、E_REGRESSION、链级归因与三类成因 | 需求 4、5 |
| `configreg/ledger.py` | SQLite 账本、批次事务、快照与整体替换 | 需求 7、8 |
| `configreg/session.py` | 会话编排、崩溃恢复、E_REPLAY 重放校验 | 需求 7、8 |

## 2. 关键设计决策

### 2.1 确定性

* 节点样本种子 = `SHA256(会话种子, 节点ID, 配置项键)` 前 8 字节，两版清单
  在同一节点/配置项上拿到逐元素相同样本，差异只来自下发的配置值。
* 所有跨运行产物（影响链、账本摘要、快照、归因排序）经
  `canonical_bytes` 以字典序键、无多余空白序列化，BFS/拓扑同层按键排序，
  因此与 dict 遍历顺序、线程调度顺序、节点启动顺序无关，逐字节稳定。

### 2.2 定点口径（显式声明）

* `scale = 10**precision`；叶子值 `q = trunc_toward_zero(x * scale)`。
* 定点乘：`(q_a * q_b) // scale`（向零截断）；定点加：饱和加。
* 饱和上下限 `[-2^63, 2^63-1]`，越界钳制；饱和后误差界退化为 `inf`。
* 单值量化误差 ≤ 1 ulp；派生误差按系数绝对值合成
  `E(y) = ulp(1+Σ|a_i|) + Σ|a_i|E(x_i)`；两版差异误差界为两侧界之和并
  额外覆盖系数变更贡献。`|Δ| < 界` 判为不可区分差异并记录依据。

### 2.3 批次原子性与恢复

一个配置项的基线与候选观测必须落在同一批次
（`split_item_atomic_batches`），否则先提交的批次会把“对侧观测尚未采集”
误判成缺失。每批在单个 SQLite 事务内写差异切片、判定切片、观测、调度
记录与进度游标。强制终止时只有完整批次可见；未提交配置项在重启时进入
未知列表并整体重跑，平台不会生成终止时刻的退出报告（CLI 返回
`E_TERMINATED`，码 3）。

### 2.4 并发与重试隔离

* 全局 `BoundedSemaphore(global_concurrency)` + 每节点
  `BoundedSemaphore(per_node_concurrency)` 双重约束。
* 排队等待的单调时钟毫秒（`time.monotonic_ns()`）分别记录全局与节点配额。
* 每次尝试使用独立 `request_id`，超时/回执缺失按退避（含抖动）重试；
  只有最终成功尝试进入观测与差异判定，杜绝双计入。
* 节点 `fail_rate` 注入真实丢包（不回包），由真实 `socket.settimeout`
  触发超时，而非模拟异常。

### 2.5 三类归因的显式阈值

1. 仅 `elapsed_ms` 超预算、摘要逐字段相等 → `node_execution_dispersion`。
2. 触发项自身派生系数/常量变更，或主导层级 ≥1 且路径最大系数绝对值
   ≥ 1.5 → `derived_coefficient_amplification`。
3. 其余（触发层主导、直接取值差异）→ `direct_value_diff`
   （直接差异判定阈值 0.5）。

### 2.6 快照整体替换

会话身份包含两版版本号、两版清单摘要、精度、精度参数版本、样本种子、
账本格式版本。重开时任一字段不一致即在单事务内清空全部业务表，记录
`replacement_events` 并稳定列出受影响配置项与节点，不做增量合并。

### 2.7 重放校验

按账本身份重放全部采集，重新评估后与已提交的状态、归因、摘要逐字段
比较；非严格计时模式下耗时只比较“是否超预算/结论”，规避真实调度抖动，
`--strict-timing` 则连毫秒值都要求一致。首个不一致以
`E_REPLAY` 定位到 `item`、`node`、`field`。

## 3. 数据布局

```
data/                  主场景清单与注册表（脚本生成）
data/dispersion/       节点执行离散场景
artifacts/             账本、快照、节点日志、JSON 结论与演示产物
configreg/             平台源码
tests/                 unittest 测试（21 个）
scripts/               输入生成与一键验收演示
docs/                  技术说明与运行手册
```
