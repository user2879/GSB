# 租约化批调度子系统 —— 本地技术验证说明

## 1. 目标与边界

在不使用容器、不访问外部服务的前提下，于 WSL Ubuntu 本地验证以下链路可被
**唯一复核**：

多工作进程（独立子进程，127.0.0.1 回环 TCP，JSON 行协议）按批前向，
按租约占用并发槽位，r1/r2/r3 三种历史回执格式归一，全部结论共用同一套
整数定点水位口径，并通过 SQLite 账本支持强制终止后的恢复重放。

## 2. 模块地图

| 模块 | 职责 | 对应业务要求 |
| --- | --- | --- |
| `leasebatch/spec.py` | JSON/NDJSON 加载、来源行号追踪与 E_SPEC 强校验 | 1 |
| `leasebatch/receipt.py` | r1/r2/r3 编解码、E_FIELD/E_RANK/E_REF 校验、归一与补齐依据 | 2 |
| `leasebatch/plan.py` | 确定性切批、稳定序列化的批标识与内容摘要 | 3 |
| `leasebatch/fixedpoint.py` | NumPy 整数定点前向、确定性输入/参数、输出摘要 | 4 |
| `leasebatch/worker.py` | 真实回环 TCP 工作进程子进程与客户端 | 4 |
| `leasebatch/lease.py` | 单调时钟租约、乐观续约、到期/取消唯一归属裁决 | 5 |
| `leasebatch/scheduler.py` | 全局+每工作进程双上限、线程池、排队单调计时 | 6 |
| `leasebatch/invalidation.py` | 参数版本→批→请求依赖链失效传播与重算子集 | 7 |
| `leasebatch/ledger.py` | SQLite 原子提交、快照整体替换、恢复与 E_REPLAY | 8 |
| `leasebatch/engine.py` | 端到端编排与单事务落盘 | 4/6/8 |

## 3. 关键设计决策

### 3.1 唯一复核的基础：规范化整数摘要

`digest.canonical_json` 使用键排序、无空白、UTF-8、禁用 NaN 的 JSON 编码，
所有进入摘要的值均为整数/字符串/布尔/None 或其有序集合。批标识、批内容
摘要、输出摘要、参数摘要、采纳结论、账本内容摘要全部基于该函数，因此不依赖
字典遍历顺序与派发顺序。

### 3.2 整数定点水位

`fixedpoint.SCALE=1000` 为统一十进制定点缩放；矩阵乘后以整除 `// SCALE`
完成定点收缩，ReLU 与偏置均在 `int64` 域运算。输入与参数由
`(种子, 标识, 参数版本)` 经 SHA-256 派生的 NumPy 生成器确定性产生，
重算逐字节一致。

### 3.3 三格式归一与补齐依据

- r1：`batch_id, request_ids, output_digest`
- r2：r1 + `param_version, elapsed_ms, worker_id`
- r3：r2 + `ranks, superseded_refs, param_digest`

归一结果携带 `source_format`、`filled_fields` 与每个字段的 `fill_basis`
（例如 `merge_rules.default_param_version`、
`fixedpoint.parameter_digest(...)`、稳定名次规则等），使补齐过程可追溯。

### 3.4 租约与槽位唯一归属

租约以真实 `time.monotonic_ns()`（毫秒）计时；续约采用版本号乐观校验，
版本不符抛 `E_LEASE` 且不延长到期。到期回收与批次取消回收共用
`LeaseManager._reclaim`，通过 `released_by` 保证每次释放仅归属一个原因，
二次回收被拒绝（幂等），槽位只回收一次并可被重新签发。

### 3.5 双上限并行调度

在途批次数等于活跃租约数；调度循环在 `global_max_active` 与工作进程
`slot_cap` 同时满足时才经线程池派发，不存在绕过租约的派发路径。
排队等待以同一单调时钟记录 `wait_start_ms/wait_end_ms/wait_ms`。

### 3.6 失效传播

沿 `工作进程参数版本 → 批 → 请求归属` 链：参数版本升级只失效该工作进程的
批；种子变更失效全部批；批大小/成员变化按结构差异失效。输出按
`(worker_id, batch_index)` 与请求排序稳定排列，每步带传播依据，未失效批的
内容/输出摘要逐字节保持。

### 3.7 SQLite 原子性与重放

每张快照、每批四要素（回执、采纳结论、租约状态、进度游标）均在单事务内
写入并携带整数 `format_version=1` 与内容摘要。快照替换在一个事务里清空旧
快照并写入新快照。恢复仅信任 `committed=1` 的行，未提交行标记为
`unknown` 并先回收其配额；重放逐字段比对，首个不一致以 `E_REPLAY`
定位到批标识与字段。

## 4. 运行方式

```bash
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests -v
python3 run_demo.py
```

## 5. 验证结论

见仓库根目录 `docs/VALIDATION.md` 中记录的实际运行结果。
