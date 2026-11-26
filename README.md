# quantledger

本地低精度量化推理前向链路、回环 TCP 工作子进程、带代次租约和 q1/q2/q3 历史回执归一账本的完整实现。

## 1. 交付内容

- `quantledger/`：可安装 Python 包，包含配置校验、量化算子、工作端协议、租约状态机、回执转换/裁决、原子账本、命令回放、诊断与 CLI。
- `configs/`：三工作进程、三层算子链、确定性批样本、9 条变更命令和 4 项故障注入配置。
- `configs/receipts/`：跨工作进程并存的 q1/q2/q3 历史回执样本。
- `scripts/generate_samples.py`：无外部网络依赖的确定性样本生成器。
- `tests/`：配置、量化、缓存键、回执往返、租约、协议、增量/重建、强杀恢复测试。
- `artifacts/ledger.json`：本次验收实际生成的归一账本。
- `artifacts/receipts/`：本次验收落盘的 q1、q2、q3 与 q3→q1 旁路。
- `artifacts/verification-report.json`：可复核的验收结果。

## 2. Linux/WSL 环境

要求 Python 3.10+、NumPy 1.26.4，仅绑定 `127.0.0.1`。

```bash
cd /home/zsl/feedback/uidfc5909d3-default-361

# 若系统自带 venv 可用：
python3 -m venv .venv
# 本机 ensurepip 缺失时可直接使用：
python3 -m virtualenv --clear -p /usr/bin/python3.11 .venv

. .venv/bin/activate
python -m pip install -e . 'pytest==8.2.0'
python -c "import numpy; print(numpy.__version__)"   # 1.26.4
```

默认端口为 `19101`、`19102`、`19103`。执行前应确认端口空闲；启动器会预检并返回 `E_PORT_IN_USE`。

## 3. 快速运行

```bash
# 重新生成配置、命令、历史回执（幂等）
python scripts/generate_samples.py

# 配置/命令校验；坏条目输出业务码并跳过
python -m quantledger validate \
  --config configs/base.json \
  --commands configs/commands.json

# 增量运行或续跑（账本已存在时必须扩展原命令前缀）
python -m quantledger run \
  --config configs/base.json \
  --commands configs/commands.json \
  --runtime runtime/demo

# 从空白全量重建
python -m quantledger rebuild \
  --config configs/base.json \
  --commands configs/commands.json \
  --runtime runtime/rebuild

# 一键完整验收（分阶段增量、空白重建、三回执、租约、故障、强杀恢复、复用）
python -m quantledger verify \
  --config configs/base.json \
  --commands configs/commands.json \
  --injections configs/injections.json \
  --runtime runtime/verify \
  --output artifacts/verification-report.json \
  --ledger-output artifacts/ledger.json \
  --receipts-output artifacts/receipts
```

`verify` 的退出码为 `0` 表示全绿。验证完成后可直接检查：

```bash
python -m quantledger recover --ledger artifacts/ledger.json
python -m quantledger receipts \
  artifacts/receipts/q1.json \
  artifacts/receipts/q2.json \
  artifacts/receipts/q3.json \
  --sidecar artifacts/receipts/q3_to_q1_sidecar.json
```

## 4. 配置与命令

`configs/base.json` 的主要结构：

```json
{
  "workers": [{
    "worker_id": "w1",
    "host": "127.0.0.1",
    "port": 19101,
    "concurrency_quota": 2,
    "lease_limit": 2,
    "frac_bits": 8,
    "quant_level": 8
  }],
  "operator": {
    "revision": 10,
    "frac_bits": 8,
    "quant_level": 8,
    "layers": [
      {"name": "l1", "rows": 3, "cols": 4, "group_size": 2, "weight_seed": 1101}
    ]
  },
  "batches": [
    {"batch_id": "b1", "sample_seed": 2101, "batch_size": 3}
  ]
}
```

约束包括：进程标识/端口唯一；并发配额、租约上限、维度、分组粒度均为正整数；分组粒度整除层元素数；相邻层输入输出宽度一致；权重与样本有限；小数位宽只允许 `4/6/8/10/12`；量化档位只允许 `4/8/16`；工作端只允许 `127.0.0.1`。

命令支持：

- `append_batch`：追加确定性批样本。
- `replace_weight` / `set_weight`：替换单层权重（种子或显式矩阵）。
- `set_quant_level`、`set_frac_bits`：切换量化档位与定点小数位宽。
- `set_revision`、`rollback_revision`：修订版本前移或回滚。
- `set_lease_limit`：运行中调整指定工作端租约上限。
- `remove_batch`、`noop`。
- 会改变算子内容的命令可附带 `revision`，让修订号与该次权重/量化变更原子生效。

典型业务码：`E_DUP_WORKER_ID`、`E_DUP_PORT`、`E_BAD_QUOTA`、`E_BAD_LEASE_LIMIT`、`E_UNKNOWN_LAYER`、`E_BAD_DIMENSION`、`E_GROUP_NOT_DIVISIBLE`、`E_NONFINITE_WEIGHT`、`E_BAD_QUANT_LEVEL`、`E_BAD_FRAC_BITS`。校验层对坏条目返回码并跳过，合法条目继续执行。

## 5. 定点量化与算子语义

每层执行：

1. 输入、权重分别做对称定点量化：`scale` 是 `2^-frac_bits` 的整数倍，量化范围为 `[-2^(level-1), 2^(level-1)-1]`。
2. 使用 `int64` 完成矩阵乘累加：`ACC = Qx @ Qw.T`。
3. 反量化：`Y = ACC * scale_x * scale_w + bias`。
4. 结果按层继续前向；全部中间整数、量化 scale、反量化输出、摘要和缓存键写入账本。

`configs/` 样本使用固定 PCG64 种子生成权重和批样本，不使用 GPU、不访问外部网络。

## 6. 真实 JSON 行子进程协议

每个 worker 是独立 Python 子进程，在 `127.0.0.1:<port>` 上使用一行一个 JSON 对象：

- `ping`：就绪与状态探测。
- `submit`：提交批、样本、层权重、修订号、量化参数和 TTL；立即返回租约。
- `fetch`：获取运行状态或完成结果。
- `renew`：携带 `lease_id`、当前 `generation`、TTL，可同时校验修订号。
- `release`：携带当前公认代次释放槽位。
- `events`：拉取 `E_LEASE_EXPIRED`、`E_RECONCILE_PENDING` 等回收事件。
- `status`：活动租约、运行中任务、配额、上限和当前代次。
- `set_lease_limit`：在线调整租约上限。
- `shutdown`：有序停止。

协议边界上的错误返回业务码而不是断链。实测坏请求后同一 worker 仍可继续处理有效请求。

### 租约状态机

- 每次签发递增全局 `generation`，并写入真实单调时钟到期毫秒。
- `renew` / `release` 携带错误代次时返回 `E_STALE_GENERATION`，当前代次不变。
- 过期扫描做真实状态迁移：移出活动租约集合、释放租约槽位、设置取消事件、记录 `E_LEASE_EXPIRED`，并输出 `reclaimed_units=1` 与回收后计数。
- 过期后到达的结果不能发布，产生 `E_RECONCILE_PENDING`。
- 计算并发受 `concurrency_quota` 约束，租约总数受 `lease_limit` 约束。

## 7. q1/q2/q3 回执与裁决

- q1：`batch_id`、`layer_name`、`output_summary`。
- q2：在 q1 上增加 `revision`、整数 `duration_ms`、`worker_id`。
- q3：再增加 `stable_rank`、`superseded_refs`、`quant_params_summary`。

统一内部表示以显式字符串 `__MISSING__` 补齐缺失字段，并生成 `completeness.label` 与 `missing_fields`。同一批/层多候选按：

1. 修订版本降序；
2. 整数耗时降序；
3. 工作进程标识升序；
4. 上述三项完全相同时，较高回执格式优先，最后用内容引用做确定性兜底。

裁决的候选键、缺失字段策略、胜者引用与裁决摘要都写入账本 `authority` / `arbitration_decisions`。

q3→q1 会把丢失字段写入 `q3_to_q1_sidecar.json`；q1 记录携带内部 `sidecar_key` 指针。q1→q3 时从旁路恢复，因此 q3→q1→q3 后输出、名次、覆盖引用与量化摘要结论不变。

## 8. 内容指纹、缓存与失效

每个层结果包含：

- 层内容指纹（权重与 bias）；
- 每个分组的权重内容指纹；
- 层配置指纹（形状、分组粒度、bias）；
- 量化档位、小数位宽、修订号与激活值指纹；
- 量化、整数累加、反量化三个缓存键；
- 引用批样本集合。

账本条目缓存键还包含整条链的逐层缓存键和最终输出摘要。替换单层权重时，只复用其上游前缀，失效该层及下游；独立复用实测结果为 3 个上游复用条目、仅 `l2/l3` 失效。全量量化参数或修订号变更会使受影响键全部失效。

重启后先验证账本 schema、基础配置哈希和命令前缀；匹配则续跑，基础哈希变化或命令前缀不兼容时要求全量重建。

## 9. 原子账本与恢复

账本是唯一恢复依据。每次命令执行分两次持久化：

1. 写入命令状态、失效/复用事件、批次 `pending` 和待办队列；
2. 全部结果合并后写入 `complete`、缓存、回执和诊断。

每次保存均写临时文件、`fsync`、`os.replace`，不会出现半份 JSON。强杀后 `recover` 只读该账本，报告：

- `reuse_entries`；
- `pending_queue`；
- `E_LEASE_EXPIRED`；
- `E_RECONCILE_PENDING`；
- `incomplete_batches`；
- 缓存层/账本条目数。

## 10. 本次实际验证结果

环境：Python 3.11.0rc1，NumPy 1.26.4，虚拟环境 `.venv`。

```text
pytest: 12 passed
增量语义快照 == 空白重建: true
快照摘要: 8e59c30113c2911a8cfd4fc7570b94b0b7b88c5408672564e145446d1eb8e005
分阶段增量: 6 条前缀命令 + 3 条新增命令，共 9 条
q1/q2/q3 归一记录数: 29 / 29 / 29
裁决批次数: 5；三版本重新裁决与账本一致: true
往返结论不变: true
租约过期回收: E_LEASE_EXPIRED=1，活动租约归零，后续请求成功
陈旧代次: E_STALE_GENERATION，代次不变
故障注入: 4/4 命中（INJ-001..INJ-004），隔离后有效请求成功
强杀恢复: 原子账本读取，reuse_entries=3，pending_queue=["b1"]，E_RECONCILE_PENDING=1
单层替换复用检查: 3 个上游复用条目，仅 l2/l3 与账本条目失效
```

完整机器可读证据在 `artifacts/verification-report.json`；账本与三版本回执在 `artifacts/ledger.json`、`artifacts/receipts/`。

## 11. 可复核性说明

协议真实测量的单调耗时保存在 q2/q3 回执和账本 `duration_ms` 中。验收比较使用“确定性语义快照”：逐字段比较命令、状态、逐层量化整数、累加值、反量化输出、缓存键、事件顺序、回执内容、裁决结果和诊断计数；仅排除机器相关的真实耗时数值、Linux PID 和提交代次这类运行时遥测。这样既保留真实计时，又能对任意重建/重放路径做稳定的业务结论比较。
