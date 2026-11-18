# texmat — 本地确定性纹理与材质生成组件

面向本地固定种子样本的程序化纹理渲染、四通道材质合成、定点量化与数值可信度诊断组件。
仅依赖本地 Python 3.10+、NumPy 与 Pillow，不访问网络，不使用容器。

## 1. 设计总览

### 1.1 定点数模型
- 所有进入管线后的值均为带符号整数定点数 `Q(F)`（隐式比例 `2**F`），浮点仅出现在
  参数量化点与坐标派生噪声处，且原始噪声立即被量化为整数。
- 层累加在更宽的工作段 `Q(16)`（`texmat/pipeline.py` 中 `WORK_BITS`）进行：
  - `quantize=true`：每层量化收窄到配置的 `Q(F)`，误差在该层即被提交，下一层在收窄后的场上累加；
  - `quantize=false`：宽段 `Q(16)` 原样向后传递，收窄只发生在通道合成阶段。
  因此关闭量化会真实改变中间定点段，而不只是改报告字段（有专门测试覆盖）。
- 舍入模式 `truncate / half_up / half_even` 全部由整数 `divmod` 实现，见 `texmat/fixedpoint.py`。

### 1.2 确定性噪声
- 噪声是像素坐标、层指纹与种子的纯函数，向量化 SplitMix64（`texmat/noise.py`），无任何 RNG 采样。
- 层叠加顺序固定为层序号升序，层内像素固定行优先（C 序）；同输入同档位多次运行逐像素一致。

### 1.3 四通道材质
- 每个材质把所选层映射为 `diffuse / roughness / metalness / normal` 四通道，
  通道值 = `base + scale * height + bias`，全部为定点整数运算。
- 混合模式 `weighted / additive / multiply` 均为定点实现；权重先归一化再量化。
- 等权材质的每个像素记录为 tie 像素；舍入恰好处于 `.5` 边界的像素记录为 rounding-boundary 像素。

### 1.4 误差逐级归因
- 同一输入分别在 `F` 与 `F+1` 两档位完整计算，比较四个环节（`texmat/diagnostics.py`）：
  `noise_overlay → quantize → material_blend → channel_compose`，覆盖全部像素，不抽样。
- 输出各环节逐像素差值、最大/平均绝对差，以及每个像素首次超过容差的环节。
- 当两档位在最终 uint8 判定区间（`bin_width` 定义在字节域）落在不同侧或舍入到不同字节时，
  记录为边界敏感像素，并按四个环节差值强度给出主导误差来源。

### 1.5 内容指纹与精确失效
- 节点分四类：`texture / layer / channel / final`，缓存键由上游内容指纹与稳定参数指纹派生
  （`texmat/cache.py`、`texmat/hashing.py`）。层记录引用的基础纹理，通道记录引用的层集合。
- 替换纹理、改某层频率/振幅、切混合模式或位宽时，仅受影响节点及其下游进入重算集合，
  未失效节点直接复用且不重复计算（落盘前还会复核载荷指纹）。

### 1.6 并发、预算与稳定顺序
- 层噪声与材质混合通过线程池并发，最大并发数取自配置（`texmat/scheduler.py`）。
- 调度器按 `(阶段, 序号, 稳定键)` 排序做预算准入，扣减序列、待算标记、失败清单与最终装配顺序
  与线程调度时机无关；超预算任务以 `E_POINT_BUDGET_PENDING` 保留而非丢弃。
- 单任务成本超过整批预算时进入保底专用批次，保证任何输入最终都能推进，不会死锁。

### 1.7 快照与恢复
- 所有产物先写临时文件再 `os.replace` 原子替换（`texmat/snapshot.py`）。
- 清单记录全局版本号、配置哈希、位宽档位、混合模式、已完成批次游标与各产物版本。
- 启动按配置哈希判定续跑：严格模式不一致报 `E_CONFIG_MISMATCH`；非严格模式记录失配并按失效重建。
- 收到可处理退出信号（或 `--stop-after-batches` 确定性钩子）时，完成当前批原子提交后以退出码 0 退出；
  强制终止后只依据清单中已完整替换的产物恢复，游离临时文件被忽略。

## 2. 目录结构

```
texmat/            组件源码（错误码、定点数、噪声、配置、加载、管线、诊断、缓存、调度、快照、引擎、CLI）
scripts/           样本生成与端到端验证驱动
tests/             49 个 unittest 用例（仅用标准库 + numpy + Pillow）
samples/           本地确定性样本（由脚本生成）
outputs/           快照与 JSON 报告输出目录
```

## 3. 使用方式

生成确定性样本：

```bash
python3 scripts/generate_samples.py --out samples/basic --size 32 --workers 4 --budget 100000
```

执行渲染（支持断点续跑）：

```bash
python3 -m texmat run \
  --sample-dir samples/basic \
  --snapshot-dir outputs/snap \
  --report outputs/report.json
```

常用开关：`--strict-resume`（配置哈希不一致即报错）、`--stop-after-batches N`
（确定性优雅退出）、`--stall-seconds N`（信号测试钩子）。

运行测试与端到端验证：

```bash
python3 -m unittest discover -s tests
python3 scripts/run_verification.py --workdir outputs/verification
```

## 4. 错误码

| 代码 | 含义 | 处置 |
| --- | --- | --- |
| `E_CONFIG_INVALID` | 配置/材质描述非法（档位、频率振幅、位宽、权重、通道完整性等） | 致命，写报告并以非零退出 |
| `E_TEXTURE_DECODE` | PNG 头/尺寸/模式/解码失败 | 进入失败清单，不中止其余样本 |
| `E_CONFIG_MISMATCH` | 恢复时清单配置哈希与当前输入不一致 | 严格模式报错，非严格模式重建 |
| `E_POINT_BUDGET_PENDING` | 任务超出单批点数预算被保留 | 标记未完成，下次运行续算 |

## 5. 实测结果

### 5.1 单元/集成测试
`python3 -m unittest discover -s tests`：**49 个用例全部通过**，覆盖定点舍入、噪声确定性、
配置校验、PNG 故障隔离、量化开关、混合模式、四环节误差归因、精确失效、增量与全量逐像素一致、
并发度 1/2/4/8 一致、预算恢复、信号优雅退出、强制终止恢复与 CLI 报告。

### 5.2 端到端验证
`python3 scripts/run_verification.py` 全部 PASS，包括：
- 增量编辑序列（纹理替换、层频率振幅调整、混合模式切换、位宽变更）的 59 个快照载荷
  与空白工作区一次性全量重建逐个数组相等；
- 并发度 1 与 8 逐像素一致；
- 紧预算（180 点）经 11 次恢复运行后完整收敛；
- 损坏 PNG 被隔离为 `E_TEXTURE_DECODE` 且其余样本继续；非法配置报 `E_CONFIG_INVALID`。

### 5.3 误差归因示例（F=8，参考 F=9，32×32 样本）

| 环节 | 最大绝对差(Q单位) | 平均绝对差 |
| --- | --- | --- |
| noise_overlay | 256 (Q16，即 1 个 Q8 单位) | 74.1250 |
| quantize | 1 | 0.2829 |
| material_blend | 2 | 2.0000 |
| channel_compose | 1 | 1.0000 |

收紧判定区间（`bin_width=2`）时可检出边界敏感像素，并标注主导误差环节（如 `noise_overlay`）
与首次越界环节；报告同时给出 tie 与舍入边界像素清单。

## 6. 一致性保证说明

增量运行与全量重建的“最终材质像素、逐层中间定点数组”逐字段相等由测试直接断言。
报告中的缓存命中/失效统计天然依赖运行起点（增量运行有命中、空白全量无命中），因此其确定性字段
（失效规划 `invalidation_plan`、重算集合、稳定键顺序）逐字段可比，而命中计数按设计反映运行历史；
这是缓存语义的预期行为，不影响任何像素与中间定点结果的可复现性。
