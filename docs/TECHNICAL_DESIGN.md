# 多轮对话状态管理管线 — 技术说明

本地、确定性、无外部服务依赖的对话状态工程。所有指纹均基于
`sort_keys` 的规范 JSON 与 SHA-256，运行结果与字典顺序、平台哈希种子
无关；同一最终输入下，增量重放与从空白全量重建逐字段一致。

## 1. 模块结构

| 模块 | 职责 |
| --- | --- |
| `dialog_state_pipeline/loader.py` | 逐行 JSONL 解析；追加/替换/删除物化；稠密 1-based 轮序校验 |
| `dialog_state_pipeline/config.py` | FSM YAML 与管线 JSON 配置加载、校验与指纹 |
| `dialog_state_pipeline/slots.py` | 槽位类型解析、范围/枚举校验、四种确定性合并策略 |
| `dialog_state_pipeline/state_machine.py` | 转移规则索引、合法转移与拒绝记录 |
| `dialog_state_pipeline/engine.py` | 单会话推进、哈希链缓存键、轮次检查点、异常隔离 |
| `dialog_state_pipeline/scheduler.py` | 线程池并发、全局预算与单会话配额、可审计账本、待办续跑 |
| `dialog_state_pipeline/snapshot.py` | 临时目录写入 + 符号链接整体替换、清单版本/哈希判定 |
| `dialog_state_pipeline/injection.py` | 独立故障注入配置与四类真实输入故障 |
| `dialog_state_pipeline/report.py` | 诊断报告、确定性失败排序、NumPy 统计、比较视图 |
| `dialog_state_pipeline/pipeline.py` | 加载/推进/结算三阶段编排 |
| `dialog_state_pipeline/sample_data.py` | 固定种子 (`SEED=20261122`) 样本生成 |

## 2. 关键设计

### 2.1 事件物化与轮序

事件行支持 `op ∈ {append, replace, delete}`：

- `append` 的 `turn` 必须等于下一稠密位置，跳号/回退即 `E_STREAM_INVALID`；
- `replace` 原地替换同位置载荷；
- `delete` 删除后对后续轮次重新稠密编号，保留载荷。

校验失败只进入失败清单并丢弃问题行，不中断其它事件。

### 2.2 槽位合并（定点整数）

置信度为 `[0, 10000]` 的整数（可由配置调整）。加权累积维护两个整数
累加器：`weighted_sum = Σ round(value·10^d)·confidence`、
`weight_total = Σ confidence`，有效取值仅在读取时做一次半偶舍入除法，
运算顺序严格等于事件顺序。四种策略：

- `overwrite` 覆盖；`larger` 仅接受更大值并记录跳过；
- `weighted` 定点加权；`reject` 首写生效、后续拒绝并留痕。

每条合并历史都记录策略、动作、来源轮次与批次。非法取值抛
`E_SLOT_INVALID`，且一整轮的操作先全部解析再落库，坏操作不改变任何槽位。

### 2.3 缓存键与精确污染判定

每轮缓存键为哈希链：

```
key(t) = H( key(t-1) | content_fp(t) | fsm_fp | config_fp )
```

其中 `content_fp` 覆盖话语、动作、槽位指令、批次与来源位置。编辑某轮只
改变该轮及其下游键；最长匹配前缀之前的检查点直接复用，未受影响会话不
重新推进。FSM 或配置指纹变化会使所有键失效，从而安全触发全量重放。

### 2.4 并发预算调度

- 会话按 ID 字典序排列，待办会话优先；每轮最多给每个未完成会话分配
  1 个状态条目，单轮总分配永不超过全局剩余预算；
- 工作线程只接收固定分配，预算账本由编排线程单写，结果按会话顺序合并
  （而非完成顺序），因此账本序列可复现、可审计；
- 超单会话配额 → `E_QUOTA_EXCEEDED` 终止该会话；全局预算耗尽 →
  `E_BUDGET_PENDING` 保留待办；续跑优先处理待办。
- `budget_refill_on_resume=True` 时把配置预算视为新预算周期，累计消耗
  与无空洞账本（连续 `seq`）仍持久化在快照中。

### 2.5 阶段化原子快照

`load` / `advance` / `settle` 每阶段写入 `versions/vN/` 临时目录，再用
`os.symlink` + 单次替换切换 `current`，不存在逐文件覆盖。清单包含：
格式版本、全局版本、配置哈希、FSM 哈希、已完成批次数、已消耗预算、
待办会话、扣减账本、各产物版本与检查点哈希。

启动时按格式版本、配置哈希、FSM 哈希判定：不一致输出
`E_SNAPSHOT_STALE`（报告顶层 `snapshot_event`）并清空重放；一致则从
已完成批次之后继续并优先重放待办。

### 2.6 故障注入与隔离

注入文件独立于管线配置，四项注入与业务码映射：

| injection_id | 故障类型 | 命中业务码 |
| --- | --- | --- |
| `INJ-BAD-SLOT-01` | 非法槽位取值 | `E_SLOT_INVALID` |
| `INJ-TURN-JUMP-01` | 轮序号跳变 | `E_STREAM_INVALID` |
| `INJ-UNKNOWN-TX-01` | 未知转移目标 | `E_TRANSITION_REJECTED` |
| `INJ-QUOTA-01` | 配额越界覆盖 | `E_QUOTA_EXCEEDED` |

单条事件异常在引擎内被捕获并记录会话 ID、轮次、阶段、业务码后继续；
报告的 `injections.hits` 以注入编号回指命中证据。

## 3. 失败清单排序

报告层按业务码固定优先级排序（`report.FAILURE_ORDER`），再按会话、
轮次、阶段、消息稳定排序，保证增量与全量报告的失败清单顺序逐字段相同。

## 4. 运行方式

```bash
python -m pip install numpy pytest
python -m dialog_state_pipeline.cli gen-samples --out samples
python run_demo.py                       # 三大验收场景
python -m pytest -q                      # 32 个测试
```

CLI 常用参数：`--reset`（忽略旧快照）、`--refill-budget`（续跑预算续期）、
`--stop-after-stage {load,advance,settle}`（模拟中断）、
`--injection`（指定注入文件）。

## 5. 实际验证结果

见 `docs/VERIFICATION.md`（由 `run_demo.py` 与 pytest 实测生成/核对）。

