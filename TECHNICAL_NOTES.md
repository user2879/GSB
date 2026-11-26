# 训练检查点诊断 — 技术说明与验证记录

## 1. 环境与运行

- 平台：WSL Ubuntu（无 Docker / 容器），Python 3.11，仅依赖 `numpy` 与标准库
  （`sqlite3`、`socketserver`、`concurrent.futures`、`threading`）。
- 当前镜像缺少 `python3.11-venv` 且无 sudo，因此用已安装的 `virtualenv` 建环境：

```bash
python3 -m virtualenv .venv
.venv/bin/pip install -r requirements.txt
```

- Tkinter 未参与实现：全部交互通过回环 TCP JSONL、SQLite 与文件完成，无需 GUI，
  因此 `python3-tk` 缺失不影响任何功能。
- 端口由 `ckpt_diag.protocol.pick_free_port()` 绑定 `127.0.0.1:0` 自动选取，
  声明生成时写入 JSON，重复端口会在加载期以 `E_SPEC` 拒绝。

端到端演示：

```bash
.venv/bin/python demo/run_demo.py
```

单元 / 集成测试：

```bash
.venv/bin/python -m pytest tests/ -q
```

## 2. 包结构

```
src/ckpt_diag/
  errors.py        # E_SPEC/E_FORMAT/E_FIELD/E_LEASE/E_DIVERGE/E_REPLAY
  clock.py         # 单调毫秒时钟（time.monotonic_ns）+ 可注入 VirtualClock
  spec.py          # R1 声明加载与强校验
  digest.py        # canonical JSON + SHA-256，保证逐位一致
  compute.py       # R3 确定性参数、整数定点 Q14 前向与损失累加
  receipts.py      # R2 c1/c2/c3 编解码、补齐归一、混合仲裁
  lease.py         # R4 租约签发/续约/释放/真实过期
  quota.py         # R5 纯确定性唯一归属裁决状态机
  scheduler.py     # R6 三重配额约束的真实线程池调度
  invalidation.py  # R7 依赖链失效传播与子集重算
  ledger.py        # R8 SQLite 原子提交、快照替换、崩溃重放
  protocol.py      # 回环 TCP JSONL 收发
  shard_process.py # 独立分片子进程服务器
  orchestrator.py  # 子进程编排、调度、裁决、账本串联
  samplegen.py     # 本地确定性声明与历史回执生成
demo/run_demo.py   # 覆盖 R1-R8 的端到端演示
tests/             # 53 个测试（含真实子进程集成测试）
```

## 3. 需求到实现的映射

### R1 强校验（`spec.py`）
分片/检查点标识唯一、host 必须为字面量 `127.0.0.1`、端口全局唯一、
租约上限与并发配额为非负整数；层 `parent` 必须存在且依赖图 DFS 染色检测无环；
检查点 `generation` 分片链内严格递增、回滚目标必须是链上检查点；事件 `seq`
严格递增。所有失败在第一个出错位置抛 `E_SPEC`（含 JSON 路径与字段名）并拒绝加载。

### R2 回执归一（`receipts.py`）
- 未知 `format_version` → `E_FORMAT`；必需字段缺失 → `E_FIELD`；
  `elapsed_ms < 0` → `E_FIELD`；定宽字段宽度不足 → `E_FIELD`。
- 显式补齐规则集中在 `BACKFILL_RULES`，归一结果携带 `source_format`、
  `backfilled_fields` 与 `backfill_basis`（补齐依据）。
- 同批次多格式按 `(param_version 降序, format_version 降序)` 稳定选择，
  被否决回执保留 `rejected_reason`。

### R3 确定性定点计算（`compute.py`）
- Q14 定点，全部 `int64` 运算并带溢出守卫，无浮点参与前向/损失。
- 权重与样本由 `numpy.random.SeedSequence/Generator` + 显式整数种子派生；
  摘要前对层、样本排序并使用 canonical JSON + SHA-256，因此与下发顺序、
  字典遍历顺序、并发度无关（测试中 16 线程重算逐位一致）。
- 损失摘要绑定分片、批次、参与样本集合、参数版本、参数摘要。

### R4 租约（`lease.py` + `shard_process.py`）
租约含 `lease_id/shard_id/granted_ms/expires_ms/renew_count/lease_version`。
续约用 `lease_version` 做乐观校验，版本不符抛 `E_LEASE` 且不延长到期；
同一批次不能重复持租；过期判定基于真实单调时钟。续约逻辑在分片子进程内执行。

### R5 唯一归属裁决（`quota.py`）
全局资源单位池的每个单位在任意时刻只有一个归属：
`active / expired_reclaim / void_reclaim / free`。到期回收与回滚作废回收互斥，
同一单位不能同时计入两类回收，也不能被两个分片持有。裁决器是纯事件状态机，
任意事件前缀重放得到相同归属序列与容量余量，每次变化记录依据、事件序号、
参数版本与租约版本。

### R6 三重约束调度（`scheduler.py`）
全局预算、分片租约上限、分片并发配额在准入时同时检查；不满足则在
`threading.Condition` 上真实阻塞（无人工 sleep），并以单调毫秒记录
`wait_start_ms/wait_end_ms`；工作真实投射到 `ThreadPoolExecutor`。

### R7 失效传播与子集重算（`invalidation.py`）
层 scale/维度/parent 变更 → 变更层 + 传递派生层闭包 → 命中 scope 的批次
→ 损失摘要 → 冻结对应参数版本的检查点，按分片序、批次序稳定输出每步依据。
仅失效子集重算，并与从空白全量重建逐字段比对（不一致 `E_DIVERGE`），
未失效批次 canonical 字节保持不变。

### R8 SQLite 账本（`ledger.py`）
每个批次在单事务内写批次归属、参数摘要、损失摘要、检查点状态、租约归属、
进度游标，并带整数格式版本与内容摘要。重开快照时版本或摘要不符则删除整库
整体替换（不增量合并）。强制终止后仅信任已提交记录，未提交批次进入
`pending` 标记为 unknown 并要求重跑；恢复后逐批重算，首个差异抛 `E_REPLAY`。

## 4. 验证结果

### 自动化测试

```
53 passed in ~3.2s
```

覆盖：R1 非法输入 12 例、R2 编解码/补齐/仲裁 9 例、R3 确定性 5 例、
R4 租约 5 例、R5 归属/重放 8 例、R6 真实线程配额阻塞 4 例、
R7 失效与逐字节稳定 3 例、R8 账本 5 例、真实分片子进程集成 2 例。

### 端到端演示（`artifacts/demo_run.log`）
- 三个分片子进程在回环端口接收 JSONL，完成 4 个批次定点计算并回传毫秒耗时。
- c1/c2/c3 混合批次稳定采纳 c3，c1/c2 带否决理由；4 类坏回执分别报
  `E_FORMAT/E_FIELD`。
- 归属序列展示 `free→active→expired_reclaim/void_reclaim→free` 的唯一归属，
  最终容量余量 3/6。
- L1 缩放变更仅失效 alpha 的两个批次与两个检查点，beta 批次字节不变，
  子集重算与全量重建一致。
- 账本 4 批次重放校验通过；版本不符触发整库替换；模拟崩溃的未提交批次
  进入 rerun 列表。
