# 跨任务迁移学习组件（本地确定性小样本）

面向本地确定性小样本的跨任务迁移学习组件。核心解决：共享编码器基座被逐层
局部替换后，各任务适配器应当**复用、部分重算还是全量重训**的贡献追踪与
失效边界判定问题，并保证任意替换批次序列下**增量运行与从零全量重放逐字段
一致**。

纯 Python 3.10+ / NumPy 1.24+，无需 GPU 与外部数据集，不使用容器。

## 目录结构

```
tlc/                    组件包
  loaders.py            输入加载与校验层（业务码）
  encoder.py            共享编码器前向 + 逐层贡献追踪
  adapter.py            线性探针/低秩适配器训练与评估
  invalidation.py       基于贡献偏移的失效边界判定
  scheduler.py          局部重训预算、待办保留、确定性并发
  state.py              阶段化快照、整体替换提交、恢复判定
  pipeline.py           单批运行编排（阶段 A~E）与全量重放
  datagen.py            固定种子合成数据生成
  cli.py                命令行入口
scripts/
  generate_data.py      生成合成样本
  run_demo.sh           端到端演示（增量 vs 全量重建比对）
  kill_resume_test.sh   真实 kill -9 强制终止与恢复比对
tests/                  pytest 测试（21 项）
data/                   生成的样例输入（tasks/features/encoder/contributions/plan/config + 批次覆盖）
```

## 输入格式（data/）

- `tasks.json`：任务名、类别集合、训练/验证样本 ID、目标层白名单、
  适配器（类型/秩/缩放）、每任务局部重训预算。
- `features.jsonl`：每行 `{"id", "features": [...], "labels": {task: label}}`。
- `encoder.json`：输入维度、归一化统计量、逐层 `name/version/weight/bias/checksum`。
- `contributions.json`：逐层逐任务敏感度系数与冻结标记（基座贡献清单）。
- `replacement_plan.json`：按批的基座层替换（层名 + 新版本号）。
- `config.json`：种子、学习率、轮数、批量、最大并发数、贡献偏移阈值、闭合容差。
- `batches/batch_<k>/`：可选批次覆盖（tasks.json / features.jsonl / config.json），
  用于演示任务增删、类别集合修改、样本特征替换、秩与缩放变更、阈值调整。

## 运行

```bash
python3 -m tlc gen-data --out data                 # 生成固定种子样本
python3 -m tlc run --data data --batch 0 --state state --report reports/run0.json
python3 -m tlc run --data data --batch 1 --state state --report reports/run1.json
python3 -m tlc rebuild --data data --batches 3 --state state_full --report reports/rebuild3.json
python3 -m pytest tests/ -q
bash scripts/run_demo.sh
bash scripts/kill_resume_test.sh
```

## 核心设计

### 贡献追踪（encoder.py）
特征按编码器归一化统计量标准化后逐层 `h = h @ W.T + b`。任务适配器损失对
各层输出的链式梯度（Taylor 一阶项 `mean(<g_i, h_i>)`）与贡献清单敏感度系数
按固定层序合成，并归一化到总损失：每层每任务仅保留一个份额记录，不持久化
中间激活；份额之和与总损失的相对闭合误差写入报告 `closure.rel_err`
（容差 `closure_rel_tol`）。全部运算 float64、固定顺序，多次运行逐元素一致。

### 失效边界判定（invalidation.py）
基座替换后，对存量任务在当前基座下重算贡献份额，与上一批份额（均归一化）
逐层比较取最大偏移，与 `contribution_shift_threshold` 比较：超阈值判失效
（重训），未超阈值判复用。**禁止"引用即失效"**：目标层未被替换但上游层被
替换时同样走偏移判定；判定结果给出全部参与比较的层与偏移值
（`invalidation.compared`）。任务侧内容（类别/秩/缩放/特征/训练超参）变化
单独以内容键判定，与基座偏移判定互不替代。

### 预算与待办（scheduler.py）
失效任务的成本 = 训练轮数 × 训练样本数；超过该任务 `retrain_budget` 即按
稳定排序键（任务名 + 目标层名）保留待办，命中 `E_RETRAIN_BUDGET_PENDING`，
下一批优先处理。待办任务不写入验证汇总（`metrics: null`），其贡献偏移仍
写入诊断（`invalidation`）。训练用线程池并发（不超过配置上限），每任务
随机源由任务名与目标层名派生，结果按稳定键归并，与单线程字典序执行逐位一致。

### 快照与恢复（state.py）
阶段 A_forward → B_decide → C_train → D_shares → E_eval 的产物先写入
`<state>.staging/` 临时目录，提交前比对清单中的基座层版本指纹与配置哈希，
然后 rename 整体替换（不逐文件覆盖）。强制终止后：暂存区 `run_key`（含配置
哈希、基座指纹、任务/特征指纹、前序 run_key）与当前输入一致则断点续跑，
不一致输出 `E_SNAPSHOT_STALE`；恢复结果与未中断运行逐字段一致。

### 增量 vs 全量一致性语义
"从零全量运行"定义为：**从空白状态按同一批次序列重放，每批全量计算所有
阶段**（`tlc rebuild`）。状态是已应用批次序列的纯函数，因此增量链末端与
全量重放末端逐字段一致（报告、快照、清单均含）。复用判定保留的适配器参数
在两条路径中来自同一批次、同一基座，故逐位相同。

## 业务码

| 业务码 | 含义 |
| --- | --- |
| `E_TASK_INVALID` | 任务名重复/类别空或重复/样本交集/适配器配置非法/标签越界/目标层不存在 |
| `E_FEATURE_MISSING` | 样本 ID 缺失或重复、特征向量缺失或含非有限值、缺少任务标签 |
| `E_ENCODER_SHAPE` | 层名重复、权重/偏置维度不一致、归一化统计量非法、校验摘要不匹配 |
| `E_CONTRIB_INVALID` | 贡献清单缺层×任务条目或敏感度非有限值 |
| `E_PLAN_INVALID` | 替换计划引用未知层、版本号回退（须单调不减）、替换冻结层 |
| `E_RETRAIN_BUDGET_PENDING` | 局部重训预算不足，任务保留待办 |
| `E_SNAPSHOT_STALE` | 暂存区基座指纹/配置哈希与当前输入不一致 |
| `E_CLOSURE_VIOLATION` | 贡献份额闭合误差超容差（自检，正常不触发） |

## 测试钩子

- `TLC_CRASH_AFTER_STAGE=<阶段名>`：在指定阶段提交后模拟强制终止
  （`TLC_CRASH_MODE=exit` 为 `os._exit`，否则抛出 `CrashError`）。
- `TLC_STAGE_DELAY_MS=<毫秒>`：每阶段前延迟，便于真实 kill 测试。

## 实际验证结果（本仓库，Python 3.11 + NumPy 1.26）

- `pytest tests/ -q`：21 项全部通过。
- 批次 1（layer2 替换）：task_alpha 偏移 0.0065 ≤ 0.02 判复用（上游层被替换
  但不重训，证明非"引用即失效"）；task_beta 0.4375、task_gamma 0.1981 超阈值
  判重训。
- 批次 2（layer1 替换 + 任务变更）：task_gamma 需重训但预算 50 < 144，
  命中 `E_RETRAIN_BUDGET_PENDING`，不写入验证汇总、偏移诊断保留；
  task_delta 新增训练；task_beta 秩变更按内容键重训。
- 批次 3（layer3 替换 + 特征替换 + 阈值调整）：待办任务优先处理并完成训练。
- 增量链（batch 0→3）与 `rebuild` 全量重放的最终报告/快照逐字段一致。
- 并发度 3 与并发度 1 的全部批次报告逐字段一致（除配置自身字段）。
- `kill -9` 中断后续跑，与未中断运行逐字段一致；中断后修改配置再续跑，
  正确输出 `E_SNAPSHOT_STALE`。
