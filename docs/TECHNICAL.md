# 定点整数域 PEFT 低秩适配组件 — 技术说明

## 总览

`peft_fp` 是一个面向参数高效微调（PEFT）场景的可集成低秩适配（LoRA）组件。
适配器的训练、合并、导出与跨版本数值诊断全程在 **int64 定点整数域** 完成，
所有样本与配置均为本地确定性合成数据，不依赖任何真实模型权重或网络资源。

```
samples/                     本地样本目录（scripts/gen_samples.py 确定性生成）
  base_pack/layers.json      逐层形状描述（层名、shape、dtype、二进制文件名）
  base_pack/*.bin            逐层权重/偏置二进制（float64 小端）
  tasks.json                 任务清单（任务名、目标层、适配秩、缩放系数）
  samples.jsonl              指令样本（ID、键、目标层、受控词表 token 序列）
  peft_config.yaml           PEFT 配置（白名单、lr、轮数、位宽/档位、容差、并发、预算、快照间隔）
  upgrade_plan.json          基座升级计划（按步替换指定层，种子确定性再生）
  teacher.json               合成教师低秩增量（用于构造样本目标）
faults/                      故障注入样例（scripts/gen_faults.py 生成）
peft_fp/                     组件包
  fixedpoint.py              定点算术核心 + 残差账本
  basepack.py                基座包加载校验 + 版本指纹 + 单层原子替换
  config.py                  PEFT 配置加载校验
  tasks.py                   任务/样本加载校验 + token->向量确定性映射
  train.py                   定点确定性训练 + float64 参考路径
  merge.py                   定点合并 / 可逆拆分 / 合并预算调度器
  diagnose.py                逐级误差传播追踪与翻转归因
  store.py                   阶段化快照（整体替换）+ 版本清单
  pipeline.py                主编排：校验->训练->合并->诊断->导出
scripts/                     样本生成、故障注入、端到端演示
tests/                       pytest 测试（26 项）
```

## 关键设计

### 1. 定点算术与残差账本（fixedpoint.py）
- 所有值以 int64 存储，小数位宽 `frac_bits` 由量化档位决定（枚举
  `{0:8, 1:12, 2:16, 3:20}`，诊断可在最高档之上再扩 8 位）。
- 舍入统一为 **half-away-from-zero**，通过“绝对值加半再算术移位”实现，
  负数路径与正数完全对称，保证跨运行逐元素一致。
- 每次丢位操作（量化、乘、矩阵乘、除、移位、饱和）都向 `Ledger` 追加一条
  `{op, count, round_up, round_down, exact, max_abs_residual_quanta}` 记录，
  随适配器一并导出，构成残差账本。
- 溢出策略：原始乘积先以任意精度（Python int）计算再裁剪到 ±2^62，
  存储值再经 `saturate` 钳制到存储带宽（2^40）。恶意输入（如巨大缩放系数）
  只会触发可检测的饱和，不会静默回绕。

### 2. 确定性训练（train.py）
- 初始化由 `sha256(task|layer|rank)` 派生的本地 RNG 完成，不触碰全局随机状态。
- 批量顺序为 `sample_id` 字典序，学习率/缩放系数量化为定点常数。
- 前向（Wx、Ax、BAx、缩放）、损失（平方和）、梯度（2/out 缩放、外积）、
  参数更新全部以定点整数执行；每轮记录平均损失（定点整数 + 浮点视图）。
- `train_adapter_float64` 提供逐步对应的 float64 参考路径，作为诊断真值。

### 3. 基座指纹与兼容判定（basepack.py / pipeline.py）
- 层指纹 = sha256(算法版本 | 层名 | 形状 | 权重字节 | 偏置字节)；
  基座指纹 = sha256(算法版本 | 全部层指纹按层名序连接)。
- 适配器产物同时携带训练时的基座指纹与目标层指纹；适配器键
  `key = sha256(task|layer|rank|alpha|层指纹|训练配置哈希)`。
- 加载时按 **目标层指纹** 判定兼容：基座单层替换只改变该层指纹，因此仅使
  引用该层的适配器失效（记 `E_BASE_VERSION_MISMATCH` 并重训），其余适配器
  直接复用已导出产物，不重训。全局基座指纹仍随产物记录用于续跑判定与审计。

### 4. 合并与可逆拆分（merge.py）
- 每个适配器增量在 `frac_bits + 4` 保护位精度下计算 `B@A*scaling`，
  一次性舍入到当前档位（舍入计入账本），再按 **任务名字典序** 精确累加。
- 由于累加为整数精确加法，合并结果对顺序不敏感（定点数的本来性质）；
  顺序仍固定以保证中间值记录可复现。唯一的顺序敏感路径是存储带宽饱和。
- 可逆拆分仅从 **合并产物 + 各适配器导出内容** 重算增量并逆序减除；
  还原误差超过当前档位半步（0.5 quantum）即输出 `E_IRREVERSIBLE_MERGE`
  并给出涉及的适配器与层名（饱和场景触发，见 faults/huge_alpha）。

### 5. 合并预算调度（merge.py）
- 单任务配额 = 预算 // 任务数；合并计算量估算 = 适配秩 × 目标层元素数。
- 单任务成本超配额 → 中止并记 `E_MERGE_BUDGET_EXHAUSTED`；
  全局预算耗尽 → 余下任务按任务名序保留待办队列并记 `E_MERGE_PENDING`。
  扣减序列与待办顺序完全可复现（测试双重运行比对）。

### 6. 逐级误差传播追踪（diagnose.py）
- 对同一任务分别以配置档位与高一档位完整重算 **适配器训练 → 合并权重 →
  目标层输出** 三个环节，并以 float64 参考路径为真值输出逐环节
  max/mean 绝对差；首个超过 `atol + rtol·|ref|` 的环节标注为
  `first_exceeding_stage`。
- 当某样本在两档位下的最终判定不一致（翻转）时，归因优先级：
  基座指纹变化（`base_version_change`）→ 合并顺序探针（同层全部适配器
  正/反序合并结果不同，`merge_order`）→ 量化截断（`quant_truncation`）。
  归因结论写入报告 `diagnostics[].flips`。

### 7. 增量与全量等价（pipeline.py）
- 训练结果只取决于 (任务名, 目标层, 秩, 缩放系数, 层指纹, 训练配置哈希)，
  因此“按需重训 + 键命中复用”的增量运行与空白全量重建在适配器权重、
  合并权重、损失轨迹、误差归因、失败清单上逐字段一致；
  复用/重训清单（disposition）反映存储历史，其 **顺序** 均为任务名字典序。
  测试 `test_incremental_edits_match_full_rebuild` 施加“删任务 + 改秩/缩放 +
  单层替换 + 档位切换”序列后验证该等价性。

### 8. 并发、快照与报告（pipeline.py / store.py）
- 训练用 `ThreadPoolExecutor`，worker 数 = min(配置 max_workers, 待训任务数)，
  只并行纯函数式单任务训练；结果按 (任务名, 目标层) 稳定键重排，
  与单线程串行执行逐字段一致（测试覆盖 max_workers=1 vs 4）。
- 每个阶段（validate/train/merge/export）先写入 `stages/<stage>.tmp-<pid>`，
  再通过目录 rename 整体替换提交；清单记录全局版本号、配置哈希、
  基座指纹与各阶段产物版本；启动时依据版本号 + 基座指纹 + 配置哈希判定续跑。
- 每轮运行输出 JSON 报告：逐任务损失轨迹、合并预算消耗、逐级误差归因、
  失败清单；`snapshot_interval` 控制整店快照（`snapshots/v<版本>`）。

## 错误码

| 代码 | 含义 |
|---|---|
| `E_BASE_INVALID` | 基座包非法（层名重复、形状不符、非有限权重等） |
| `E_TASK_INVALID` | 任务/样本非法（目标层不在白名单、秩/缩放非正、token 越界等） |
| `E_QUANT_INVALID` | 配置非法（位宽/档位越枚举、容差非正有限数等） |
| `E_BASE_VERSION_MISMATCH` | 适配器目标层指纹与当前基座不符，拒绝加载并重训 |
| `E_IRREVERSIBLE_MERGE` | 合并不可精确逆拆（还原误差 > 0.5 quantum） |
| `E_MERGE_BUDGET_EXHAUSTED` | 单任务合并成本超配额，任务中止 |
| `E_MERGE_PENDING` | 全局预算耗尽，任务保留待办队列 |

## 验证结果（本机，Python 3.11 / numpy 1.26 / PyYAML 6.0）

- `python3 -m pytest tests/ -q`：**26 passed**
  - 定点舍入/账本/矩阵乘确定性；七类故障注入分别命中预期错误码；
  - 重复运行逐元素一致；并发与串行一致；预算耗尽/待办可复现；
  - 单层升级只失效引用层适配器；增量编辑序列 ≡ 全量重建；
  - 合并可逆与饱和不可逆；紧容差下翻转归因 `quant_truncation`。
- `python3 scripts/run_demo.py`：初始运行 4 任务全部收敛（如 task_delta
  损失 0.14624 → 0.00244）；重跑全部 `reused` 且输出一致；升级 step 0 后
  仅 task_alpha/task_beta 重训；紧容差运行产生 16 个翻转样本，归因均为
  `quant_truncation`；增量 ≡ 全量。报告见 `demo_out/report.json`。

## 复现

```bash
pip install numpy pyyaml pytest
python3 scripts/gen_samples.py samples
python3 scripts/gen_faults.py samples faults
python3 -m pytest tests/ -q
python3 scripts/run_demo.py
python3 -m peft_fp run --samples samples --store store_demo --report report.json
python3 -m peft_fp upgrade --samples samples --step 0
```
