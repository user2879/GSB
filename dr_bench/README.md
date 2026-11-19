# 容灾切换决策实验台（DR Decision Bench）

面向 Windows 本地环境的**纯离线**容灾切换决策实验台。实验台不执行健康探测、
不启动任何被探测实例进程，只消费本地 JSON 场景中的观测样本表，按观测轮次推进
实例健康状态机、判定切换条件、选举切换目标，并通过三条独立计算路径产出
**逐字节一致**的结论，同时提供 SQLite 决策账本、快照恢复与强制终止重放。

## 1. 环境准备（Windows）

1. 安装 Python 3.11 或更高版本（python.org 官方安装包，勾选 *Add python.exe to PATH*）。
2. 打开 `cmd` 或 PowerShell：
   ```bat
   python --version
   python -m pip install -r requirements.txt
   python -m dr_bench check-env
   ```
   - `numpy` 来自 pip；`sqlite3` 是 Python 标准库，无需单独安装。
   - `tkinter` 随官方 Windows 安装包提供；`check-env` 会报告 `tkinter_importable`。
     实验台核心不依赖 GUI，Tkinter 不可用时仅查看器 `dr_bench/viewer.py` 不可用。
3. 不使用 Docker；所有文件都在项目目录内读写。

## 2. 目录结构

```
dr_bench/
├── dr_bench/            # 库与 CLI 源码
│   ├── errors.py        # E_SCENE/E_INJECT/E_SNAPSHOT/E_DIVERGE/E_REPLAY 等
│   ├── canonical.py     # 规范化 JSON 与首个差异定位
│   ├── models.py        # 服务/实例/观测/状态/运行时角色模型
│   ├── loader.py        # 场景强校验（全有或全无，带字段路径）
│   ├── statemachine.py  # unknown/healthy/unhealthy 状态机
│   ├── switchover.py    # 触发判定、稳定目标选举、事务仲裁
│   ├── engine.py        # 全量/增量/差分合成三条路径
│   ├── planner.py       # 规模自适应路径选择与稳定决胜
│   ├── ledger.py        # SQLite 账本、原子提交、快照版本识别
│   ├── recovery.py      # 恢复重放、跨路径校验、崩溃注入
│   ├── injection.py     # 三类异常注入与结构化日志
│   ├── session.py       # 会话编排
│   ├── samplegen.py     # 确定性样本生成
│   ├── demo.py          # 三阶段演示
│   ├── viewer.py        # 可选 Tkinter 查看器（优雅降级）
│   └── cli.py           # 命令行入口
├── samples/             # 确定性样本场景
├── scripts/             # 跨平台 Python 脚本与 Windows .bat 包装
├── tests/               # unittest 测试（27 个）
├── snapshots/           # 快照输出目录
└── data/                # SQLite 账本与 JSONL 日志输出目录
```

## 3. 场景文件格式

```json
{
  "scene_id": "dr-demo-scene",
  "scene_version": 1,
  "services": [
    {
      "service_id": "order-service",
      "strategy": "auto",
      "cooldown_ms": 1000,
      "min_healthy_replicas": 1,
      "success_threshold": 2,
      "failure_threshold": 2
    }
  ],
  "instances": [
    {"service_id": "order-service", "instance_id": "ord-1", "role": "primary", "version": 1}
  ],
  "observations": [
    {
      "service_id": "order-service",
      "instance_id": "ord-1",
      "round_index": 0,
      "observed_at_ms": 100,
      "result": "success"
    }
  ]
}
```

- 观测结论仅支持 `success`、`connection_refused`、`timeout`、`error_return`。
- 角色仅支持 `primary`、`replica`；每个服务恰好一个 `primary`。
- 策略仅支持 `auto`、`manual`；阈值与冷却均为非负整数。

## 4. 核心语义

### 4.1 强校验（E_SCENE，全有或全无）

- 服务/实例标识唯一；服务内角色合法且至多一个主实例。
- 观测轮次为非负整数且实例内不重复；观测时刻为整数毫秒且实例内单调不减。
- 观测结论属于四类受支持值；阈值/冷却为非负整数（布尔值被显式拒绝）。
- 任一错误都整体拒绝加载，错误列表给出 `path`（如 `observations[3].result`）。

### 4.2 健康状态机

- 状态：`unknown -> healthy/unhealthy`，连续成功达到 `success_threshold` 转 healthy，
  连续失败达到 `failure_threshold` 转 unhealthy；阈值为 0 时按 1 次观测处理。
- 每次迁移记录 `from_state/to_state/round_index/observed_at_ms/reason`。
- 某轮缺观测的实例保持上一状态，记录 `no_observation=true`；unknown 永不进入
  可用实例集合，不参与切换目标选举。

### 4.3 切换判定与目标选举

- 触发条件：主实例 unhealthy，或健康副本数低于 `min_healthy_replicas`。
- `manual` 策略拒绝自动切换并记 `E_MANUAL`；冷却窗口未结束记 `E_COOLDOWN`
  （冷却只比较观测时刻差与 `cooldown_ms`，不使用墙钟）。
- 无健康候选记 `E_NO_TARGET`；成功切换记 `SWITCHED`。
- 目标排序（显式稳定规则，不依赖字典/哈希遍历）：
  1. 健康度降序（healthy 优先，再按连续成功次数；unhealthy 按连续失败次数）；
  2. 实例版本号降序；
  3. 实例标识升序。

### 4.4 三条计算路径

| 路径 | 名称 | 行为 |
| --- | --- | --- |
| 全量重判 | `full_rebuild` | 每轮扫描全部实例与服务，重新推导完整状态集合 |
| 增量迁移 | `incremental_migration` | 只推进当轮有观测的实例，只重判受影响服务 |
| 轮次差分合成 | `round_diff_synthesis` | 只产出变化的实例/服务，再由上轮全量结论合成 |

三路径输出相同的状态集合、可用实例集合、切换目标与业务码序列；
`consistency` 命令对规范化 JSON 做**逐字节**比较，不一致以 `E_DIVERGE`
报告首个不一致实例与字段路径。

### 4.5 自适应路径选择

`planner` 依据服务数、实例数、观测行数、轮次跨度与上轮变更比例，按确定性
成本模型估算三条路径工作量；取最小值，若候选与最小值之差在 `+5%` 阈值带内，
按固定优先级 `full_rebuild -> incremental_migration -> round_diff_synthesis`
稳定决胜，并在输出中标注 `threshold_influenced=true`。

### 4.6 事务唯一性与原子提交

- 同一服务同一轮只允许一个切换事务；竞争请求按请求标识升序仲裁，
  落败者返回 `E_BUSY` 并记录原因。
- 主备角色互换与实例版本号提升（`max(version)+1`）在同一次提交中整体更新。
- 事务与状态迁移写入同一条 SQLite 事务（单连接 `BEGIN/COMMIT`），可整体回滚。

### 4.7 账本、快照与强制终止恢复

- 每轮结论（含三路径摘要、事务、业务码、进度游标）写入 SQLite，携带
  `LEDGER_FORMAT_VERSION` 整数格式版本与 SHA-256 内容摘要。
- 重开会话时比较快照场景版本与内容摘要；不一致则**整体替换**角色、策略与状态，
  不做增量合并，并写 `E_SNAPSHOT` 结构化告警。
- 强制终止时只恢复 `committed` 轮次；`pending` 轮次标记为未知并删除重跑，
  实验台不生成所谓的"终止时退出报告"。
- 恢复后按账本重放剩余轮次并逐项比对，无法复现以 `E_REPLAY` 报告依据。

## 5. 命令行用法

```bat
python -m dr_bench check-env
python -m dr_bench gen-samples --output-dir samples
python -m dr_bench validate samples\demo_scene.json
python -m dr_bench consistency samples\demo_scene.json
python -m dr_bench plan samples\demo_scene.json
python -m dr_bench run samples\demo_scene.json --db data\session.sqlite3 --snapshot-dir snapshots
python -m dr_bench replay samples\demo_scene.json --db data\session.sqlite3
python -m dr_bench crash-demo samples\demo_scene.json --db data\crash.sqlite3 ^
  --snapshot-dir snapshots --log data\events.log.jsonl --crash-round 3
python -m dr_bench inject --kind corrupt_scene --scene samples\demo_scene.json --log data\events.log.jsonl
python -m dr_bench inject --kind bad_inject_ratio --ratio 1.5 --log data\events.log.jsonl
python -m dr_bench inject --kind missing_snapshot_version --scene samples\demo_scene.json ^
  --db data\snap.sqlite3 --snapshot-dir snapshots --log data\events.log.jsonl
```

业务错误以退出码 `2` 返回，并打印结构化 `code/path/detail`。

## 6. 一键演示（三场景顺序执行）

```bat
scripts\run_demo.bat
```

或跨平台直接运行：

```bat
python scripts\run_demo.py workspaces\demo-run
```

演示顺序：
1. **冷启动全量判定**：全量路径建账，并做三路径逐字节一致性校验。
2. **故障注入后增量推进**：依次注入场景损坏（E_SCENE）、比例越界（E_INJECT）、
   快照版本缺失（E_SNAPSHOT），再以增量路径提交。
3. **强制终止恢复**：在第 3 轮提交前模拟 kill，重启后只依据已提交轮次恢复、
   重放未提交轮次并校验摘要。

输出 `data/demo_summary.json`（规范化、稳定键序）与
`data/structured.log.jsonl`（每行一条结构化日志）。

## 7. 测试

```bat
python -m unittest discover -s tests -v
```

覆盖：强校验各失败路径、状态机迁移与无观测标注、冷却抑制、人工策略、
事务仲裁 E_BUSY、三路径逐字节一致、规划器确定性、账本持久化、
强制终止 pending 重跑、快照整体替换、三类注入与 E_DIVERGE/E_REPLAY 定位。

## 8. 实际验证结果（记录）

- Python 3.11，sqlite 3.37，numpy 2.4（缺失 numpy 时自动回退纯 Python 估算）。
- `python -m unittest discover -s tests`：27/27 通过。
- 演示场景（3 服务、8 实例、51 条观测、8 轮）三路径逐字节一致；
  冷却场景验证 `OK -> SWITCHED -> E_COOLDOWN -> SWITCHED`；
  崩溃在第 3 轮注入后，恢复验证轮次 `[0,1,2]`、重跑轮次 `[3,4,5,6,7]`。
