# LoRA 增量训练与诊断工作台 — 技术说明

## 1. 目标与范围

工作台在**完全离线、纯 CPU、无真实模型权重**的条件下，解决共享冻结基座下
多任务 LoRA 适配器在数据与配置演进时的**精确失效判定**问题：

- 目标层白名单或基座版本变化属**基座级依赖**，波及全部任务；
- 单任务样本、秩、缩放系数变化属**任务级依赖**，仅失效该任务；
- 未失效任务必须直接复用产物、不得重新训练；
- 任意编辑序列后的增量结果必须与从空白工作区的全量重建逐字段一致。

技术栈仅为 Python 3.11+ 标准库与 NumPy，全部随机数来自固定种子的
`numpy.random.Generator`，不存在全局随机状态。

## 2. 目录结构

```
lora_bench/
  errors.py      稳定业务错误码（E_*）与失败记录结构
  utils.py       确定性哈希、RNG、规范化 JSON、原子写盘/目录替换
  generator.py   固定种子生成样本 JSONL、任务清单、两版冻结基座参数包
  loader.py      逐行样本校验、任务清单校验、基座二进制张量加载与指纹复核
  config.py      适配配置解析、秩/白名单校验、配置哈希（剔除调度类字段）
  forward.py     冻结基座前向表示与按样本内容寻址缓存
  lora.py        旁路 LoRA、解析梯度小批量训练、逐轮损失/指标/指纹、验证
  deps.py        双层依赖身份、缓存键、纯函数失效判定与稳定排序键
  snapshot.py    暂存目录、原子整体替换、冻结版本、内容寻址任务缓存
  pipeline.py    前向/训练/验证三阶段编排、线程池、故障隔离、间隔提交
  rebuild.py     从空白工作区全量重建与逐字段对账
  injection.py   六类异常注入、隔离执行、原子诊断报告
  cli.py         generate / run / rebuild / inject 命令行
configs/         秩、目标层、并发、提交间隔等配置变体
tests/           29 个 pytest 用例，覆盖 8 条业务规则
scripts/run_demo.py  端到端验收演示
```

## 3. 确定性设计

- **样本**：按任务独立 RNG（种子 = 主种子 + 任务序号 × 97）生成长度 4–8 的
  token 序列，标签由受控词表上的确定性函数（取模、奇偶、归一化求和、正弦）得到。
- **基座**：词表嵌入 + 4 个 `8×8` 全连接层，权重/偏置为固定种子正态采样的
  float32 原始二进制；`shapes.json` 记录形状、dtype、激活函数与整体内容指纹。
- **前向**：层序号升序，激活固定 `tanh`，输入为 token 嵌入均值（与序列长度无关），
  缓存每层 input/pre/output 与 final；基座权重只读，无反向更新。
- **LoRA**：每个目标层维护 `B(h×r) @ A(r×h)`，缩放 `alpha/r`；初始化种子由
  `("lora", 任务名, 层号)` 派生，目标读出向量种子由任务名派生；梯度为解析形式，
  批量按样本 ID 升序固定切片，学习率固定，逐轮记录损失与适配器指纹。
- **指纹**：数组指纹对 `形状:dtype:` 头 + 原始字节做 SHA-256；JSON 指纹使用
  排序键、紧凑分隔符，保证跨进程/跨平台一致。

## 4. 双层依赖与缓存键

每个任务构造一份身份（`deps.build_task_identity`），包含：

- 任务级：训练/验证样本内容指纹、秩 `rank`、缩放 `alpha`；
- 基座级：目标层白名单、基座指纹、以及影响数值结果的学习率/批量/轮数/指标规则。

缓存键为该身份的 SHA-256。失效判定 `deps.plan_invalidation` 是**纯函数**，
只比对新旧身份、不读取缓存目录，返回按稳定排序键排序的 `invalidated`、
`reused` 与 `removed_tasks`，原因按优先级排列：
`base_version > target_layers > task_samples > rank > alpha > 学习率/批量/轮数/指标`。

这保证增量运行与全量重建在同一条编辑基线上得到**完全相同的失效/复用清单**，
而两者的差异只在执行层（是否命中内容缓存、是否实际训练）。

## 5. 阶段化快照与原子提交

```
workspace/
  current/                 最新一次已提交运行（原子目录替换）
  versions/v000001/...     每次提交冻结的不可变副本
  staging/stage-xxxx/      进程内临时目录，提交后即消失
  cache/forward/           按 (基座指纹, 样本指纹) 寻址的前向表示
  cache/tasks/<key>.npz    按任务身份寻址的适配器数组
  cache/task_summaries/    对应的损失轨迹、指标、指纹摘要
```

- 前向、训练、验证三阶段均先写暂存目录，再以同卷 `os.replace` 整体发布，
  POSIX 与 Windows 均为原子操作；单文件写入走“临时文件 + `os.replace`”。
- 每完成 `commit_interval` 个任务发布一次中间快照（`final=false`），
  最后发布完整快照（`final=true`）；版本号取已冻结版本的最大号递增，
  配置变更后的新运行不会覆盖旧配置谱系的版本。
- 清单记录全局版本号、配置哈希、基座版本/指纹、已完成/失败任务、各阶段产物版本。
- 启动时以配置哈希 + 基座指纹对账：严格模式不一致直接 `E_CONFIG_MISMATCH`；
  非严格模式记录 `mismatch_fields` 并构建全新快照，绝不混用旧版本产物。
- 配置哈希只包含影响数值结果的字段；`concurrency` 与 `commit_interval`
  仅影响调度与快照节奏，被显式排除，因此改变并发度不会触发无意义失效。

## 6. 并发一致性

线程池最大并发取自配置，任务之间无共享可变状态（NumPy 矩阵运算释放 GIL）。
每个任务携带由 `任务名|layers=升序层号` 构成的稳定排序键；任务完成后统一按键排序，
再生成验证汇总、失败清单与快照内容，因此线程调度顺序与并发度无法影响输出顺序、
适配器参数表顺序、损失表顺序或复用清单顺序。

## 7. 异常注入与诊断

`injection.py` 在相互隔离的独立目录中执行 6 个注入项，报告以临时文件原子替换写盘：

| 注入 ID | 场景 | 业务码 |
| --- | --- | --- |
| INJ-01 | 样本缺必需字段 | `E_SAMPLE_INVALID` |
| INJ-02 | 基座权重含非有限数 | `E_BASE_FINITE_MISMATCH` |
| INJ-03 | 目标层超出基座层范围 | `E_TARGET_LAYER_UNKNOWN` |
| INJ-04 | 秩为零 | `E_RANK_INVALID` |
| INJ-05 | 验证样本为空 | `E_VALIDATION_EMPTY` |
| INJ-06 | 训练期任务级异常 | `E_TRAIN_TASK_FAULT`（其余任务继续） |

单任务异常记录任务名、阶段名、异常类型与消息后继续处理其余任务，
失败清单按 `(任务, 阶段, 错误码, 消息)` 稳定排序。

## 8. 本地运行（Windows / WSL 通用）

```powershell
py -3.11 -m pip install numpy pytest
py -3.11 -m lora_bench.cli generate data/input
py -3.11 -m lora_bench.cli run data/input workspace/ws --config configs/base_config.json
py -3.11 -m lora_bench.cli rebuild data/input workspace/ws workspace/rebuild --config configs/base_config.json
py -3.11 -m lora_bench.cli inject workspace/injection
py -3.11 -m pytest tests/ -q
py -3.11 scripts/run_demo.py
```

所有文件 I/O 仅使用标准库与同卷原子替换，路径拼接全部基于 `os.path`，
不依赖任何 POSIX 专有特性，可直接在 Windows 原生 Python 3.11 上运行。
