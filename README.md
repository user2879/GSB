# CondFusion 条件融合定点渲染诊断组件

一个完全本地、确定性的“文生视频 + 图生视频 + 运动条件”融合渲染诊断组件与完整示例。无需 Docker、GPU、外部网络服务或在线账号；NumPy 负责定长数组运算，PNG 用标准库 `zlib/struct` 确定性读写，快照使用 JSON + 原子符号链接切换。

原始任务书保留在 `docs/TASK_SPEC.md`，业务码见 `docs/BUSINESS_CODES.md`，依赖/Tk 说明见 `docs/DEPENDENCIES.md`。

## 快速开始

```bash
pip install -r requirements.txt
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

python3 -m condrender.cli generate-samples --root .
python3 -m condrender.cli validate --root .
python3 -m unittest discover -s tests -v
python3 -m condrender.cli verify --root . --artifacts-dir artifacts/verification
```

常用命令：

```bash
# 单次全量渲染：输出 full PNG、逐 tile PNG、报告和原子快照
python3 -m condrender.cli render --root . \
  --snapshot-dir artifacts/examples/snapshot \
  --output-dir artifacts/examples/render

# 10 条编辑命令的增量/全量一致性验收
python3 -m condrender.cli run-commands --root .

# Tk 诊断面板（系统 python3-tk 或 source scripts/tk_vendor_env.sh）
source scripts/tk_vendor_env.sh
python3 -m condrender.cli gui --root .
```

也可直接运行 `bash scripts/verify.sh` 或 `make verify`。

## 确定性样本

`python3 -m condrender.cli generate-samples` 会生成：

- `samples/controlled_vocab.yaml`：6 个实体 + 8 个属性，共 14 个受控词项。
- `samples/prompt_table.yaml`：4 帧提示词。
- `samples/images/condition_a.png`、`condition_b.png`、`condition_c.png`：固定 `8x8` RGB PNG，固定种子绘制纹理块、矩形、圆和三角图元。
- `samples/motion_vectors.yaml`：4 帧 × 2 运动块行 × 4 块列 × `dx/dy`。
- `samples/noise_table.json`：`noise-v1`，8 个候选步、每步 384 个定点噪声整数。
- `samples/render_config.yaml`：维度、种子、权重、量化档位、调度、预算和并发上限。
- `samples/command_stream.yaml`：覆盖提示词、图像替换/删除、运动、权重、步数、分块行、引导强度和量化档切换。
- `samples/injections.yaml`：4 个独立故障注入项。
- `samples/negative/`：词表、配置、维度、PNG、运动和噪声负例。

输出视频几何为 4 帧、`16x8` RGB；潜变量形状同样是 `8x16x3`，因此每个水平分块行只读取自己的连续潜变量切片。

## 管线

1. **校验加载**：`condrender/bundle.py` + `condrender/validation.py` 解析 YAML/JSON/PNG/命令流，收集而非吞掉所有 `E_xxx`。
2. **文本条件**：`condrender/conditions.py:encode_text` 按受控词表顺序计数，定点归一化为 14 维 BoW，并记录词项、类别、token 序号和字符位置。
3. **图像条件**：2×2 块分别计算 RGB 归一化均值和水平/垂直梯度，得到 20 维向量。
4. **运动条件**：校验 `[4,2,4,2]` 表并按行展开为 16 维；每个运动块行使用其中 8 维参与融合。
5. **定点融合**：融合向量为 `14+20+8=42` 维；严格按帧号、运动块行、维度升序计算，逐维记录权重、分子/分母、舍入方向和残差。
6. **潜变量初始化**：用融合摘要、种子、潜变量参数指纹、帧号和快照版本派生 64-bit PCG64 种子；只使用局部 `np.random.Generator`，不触碰全局随机状态。
7. **去噪步**：每步独立计算噪声转码残差、引导乘法残差/方向和 alpha 更新残差/方向；最后按步号升序累加。步产物不依赖上一步键，因此可在线程池中并发。
8. **解码/分块**：3×3 定点线性投影 → 增益/中心映射 → 先裁剪到 `[0,255Q]` → 确定性整数舍入 → `uint8` → 标准库编码 PNG。每个 tile 函数只接收该 tile 的潜变量切片。
9. **组合输出**：按行号升序拼回完整帧，同时写每块 PNG；调度顺序由 `tile_scheduling: zigzag` 决定，例如 4 行时为 `[0,3,1,2]`，线程池结果仍按规范顺序入账。

所有 Q 格式为 `Q=2^frac_bits`，允许 `Q8_FAST/Q12_BALANCED/Q16_DIAG`。切换量化档会改变缓存键、随机区间、舍入和最终像素；测试明确断言像素变化。

## 指纹、复用与精确失效

每个缓存键由以下内容再次 SHA-256 派生：

- 上游产物完整指纹/键；
- 步骤、帧、tile 边界等参数指纹；
- `quant_profile`、`frac_bits`、舍入模式；
- `snapshot_schema` 与整数 `SNAPSHOT_VERSION=1`。

账本条目包含 `HIT/RECOMPUTE`、阶段、帧、索引、键、成本点和命中依据。最终血缘记录每帧引用的全部步骤键、tile 键和合成键。

示例命令流在基线渲染后逐条编辑的实际计数：

| 命令 | 编辑 | HIT | RECOMPUTE | 说明 |
|---|---|---:|---:|---|
| C1 | 改帧 0 提示词 | 38 | 10 | 仅帧 0 的文本→融合→潜变量→步→tile 下游失效；帧 1–3 全复用 |
| C2 | 替换帧 2 图像 | 38 | 10 | 仅帧 2 下游失效 |
| C3 | 删除帧 3 图像 | 38 | 10 | 删除被建模为 `absent` 零向量指纹 |
| C4 | 改帧 1 运动 | 38 | 10 | 仅帧 1 下游失效 |
| C5 | 改全局权重 | 12 | 36 | 三模态编码命中，4 帧融合及下游重算 |
| C6 | 步数 3→5 | 32 | 24 | 旧步骤 0/1/2 命中，仅新步 3/4 与帧/tile 下游重算 |
| C7 | tile 2→4 | 44 | 20 | 条件、潜变量、步骤和帧潜变量全部命中，仅 tile/compose 重算 |
| C8 | 引导强度变化 | 20 | 44 | 所有步骤参数变化，编码/融合/初始潜变量命中 |
| C9 | Q12→Q16 | 0 | 64 | 量化档进入每个缓存键，全部重算 |
| C10 | 步数 5→4 | 36 | 24 | 保留步 0–3 命中，丢弃步 4，重算合成下游 |

测试还用固定种子生成 6 组随机命令序列，混合全部编辑类型和 1/2/4 并发度；增量结果与最终状态冷渲染在像素、完整残差数组、方向数组、tile 顺序和血缘键上逐字段相等。

## 并发

`max_workers: 4` 是硬上限。`RenderEngine(workers=99)` 会收缩到配置上限；测试用任务内计数器观察实际并发，并断言 `max_observed_concurrency <= worker_limit <= config.max_workers`。步骤和 tile 均通过 `ThreadPoolExecutor.map` 按规范顺序回收，因此物理线程时序不会改变账本/完成顺序。

## 快照与 SIGKILL

`SnapshotStore` 将 `snapshot.json` 写入 `snapshots/.tmp-*`，`flush + fsync` 后 rename 到 `snapshots/versions/v...`，再用临时符号链接 + `os.replace` 原子切换 `current`。清单包含：

- 整数快照版本和 schema；
- `global_version`、完整 `config_hash`、`state_fingerprint`；
- 每帧条件指纹；
- 完整内容缓存、账本、待办队列；
- 已完成帧、tile 游标；
- 已消耗重算点数和预算；
- 各阶段产物键/指纹/版本。

恢复决策包括 `resume_pending`、`incremental_config_changed`、`incremental_state_changed` 和 `full_rebuild_snapshot_version`。未提交的 `.tmp-*` 不被当作 current，可在下次启动时清理；不承诺补写一半进度。

真实验证中，子进程以 35ms/task 延迟运行，轮询到帧 0 已提交后发送 `SIGKILL`：

- 被杀进程返回码：`-9`；
- 终止前提交：`global_version=1`、`completed_frames=[0]`；
- 新进程恢复决策：`resume_pending`；
- 恢复时帧 0 的 12 个产物 HIT，其余 36 个产物 RECOMPUTE；
- 4 个完整 PNG 与同状态冷全量输出逐字节相等。

## 预览语义与故障注入

每条交互命令携带 `interaction_budget_ms`。预算内完成则返回 `fresh=true` 的最新摘要；预算为 0 或到期时返回上一已完成摘要，并带 `E_STALE_PREVIEW`、旧 `config_hash`、旧条件指纹、受影响帧区间、tile 完成情况、HIT/重算数和进度。

`samples/injections.yaml` 的四类注入均为真实输入破坏：

- `INJ-PNG-01`：把 PNG 截到一半，得到 `E_PNG_TRUNCATED`；
- `INJ-NAN-02`：在图像统计量中注入 NaN，有限性检查得到 `E_NONFINITE_CONDITION`；
- `INJ-MOT-03`：把运动块行从 2 改成 1，得到 `E_MOTION_BLOCK_MISMATCH`；
- `INJ-NOISE-04`：删除目标帧噪声表 `version`，得到 `E_NOISE_VERSION_MISSING`。

单注入测试断言目标帧、阶段、业务码和注入编号匹配，其余 3 帧完成；组合注入的失败码顺序为四个预期码。

## 主要目录

```text
condrender/
  errors.py          # E_xxx 与异常
  hashing.py         # canonical JSON / 数组指纹
  pngcodec.py        # 无 Pillow 的 PNG 编解码与截断检测
  fixedpoint.py      # Q 格式、确定性舍入、数组残差
  samples.py         # 固定种子样本生成
  validation.py      # 加载校验与失败清单
  bundle.py          # 样本包、状态、命令应用
  conditions.py      # 文本/图像/运动编码与融合
  engine.py          # 缓存、线程池、潜变量、去噪、解码、tile
  snapshot.py        # 原子快照与恢复决策
  session.py         # 交互会话、增量/全量字段比对
  acceptance.py      # 全部验收场景编排
  tk_app.py          # 可选 Tkinter 诊断面板
tests/test_pipeline.py
scripts/
samples/
docs/
artifacts/verification/
```

## 实际验证

最近一次完整结果由以下命令生成：

```bash
PYTHONPATH=. python3 -m condrender.cli verify \
  --root . --artifacts-dir artifacts/verification
```

机器可读结果：`artifacts/verification/verification_report.json`；Markdown 记录：`artifacts/verification/VERIFICATION_REPORT.md`。在最后一次完整运行中，以下六项均为 `true`：

- `command_stream_incremental_equals_full`
- `negative_validation_codes`
- `injection_isolation`
- `stale_preview`
- `sigkill_snapshot_recovery`
- `tile_cursor_sigkill_recovery`

完整 `unittest` 套件共 13 个测试，还包含随机命令序列、tile 边界、并发上限、普通快照恢复和半提交 tile 游标恢复。依赖安装与当前环境 Tk 库的实际状态见 `docs/DEPENDENCIES.md`。
