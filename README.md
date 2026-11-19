# 空间声学与声场重建诊断工作台

这是一个可离线复算的确定性诊断组件。它从本地合成的 48 kHz / 16 bit PCM / 4 通道 RIR 出发，完成球谐投影、三档规模自适应拟合、方向参考误差核验、跨档一致性判定、事务型快照和等距柱状投影诊断视图。

默认配置为 8 / 24 / 96 个源位置，对应 32 / 98 / 200 个球面重建采样方向。三档使用同一房间、听点、真值生成规则和误差口径。完整算法与口径见 `docs/ALGORITHM.md`，实际验收记录见 `docs/VALIDATION.md`。

## 快速开始

### Windows（一键）

```bat
run_demo.cmd
```

或 PowerShell：

```powershell
.\run_demo.ps1
```

脚本会创建 `.venv`、安装 `requirements.txt` 中固定的依赖，然后执行：

```bat
python -m spatial_diag run --workspace artifacts
```

### 已有 Python 环境

```bash
python -m pip install -r requirements.txt
python -m spatial_diag run --workspace artifacts
```

一条命令完成：三档 RIR 生成 → 球面事件投影 → 策略选择 → SH 拟合 → 方向误差核验 → 跨档一致性 → 原子快照 → 等距柱状投影 PNG。

## 输出

```text
artifacts/
  CURRENT.json                          # 最新已提交版本与内容哈希
  runs/<version>/snapshot.json          # 不可变诊断快照
  runs/<version>/samples/small/...      # 4 通道 WAV + 真值 manifest
  runs/<version>/samples/medium/...
  runs/<version>/samples/large/...
  runs/<version>/reports/direction-errors.jsonl
  runs/<version>/reports/step-hashes.json
  views/<version>/small.png             # 能量、参考差值、低置信标记、降级账本
  views/<version>/medium.png
  views/<version>/large.png
  latest-report.json
```

复算全部内容哈希并检测缺失/篡改：

```bash
python -m spatial_diag verify --workspace artifacts
```

单独重绘当前快照：

```bash
python -m spatial_diag render --workspace artifacts
```

PyQt5 查看器（只加载已校验的当前不可变快照）：

```bash
python -m spatial_diag qt --workspace artifacts
# 无显示器环境可导出窗口截图：
python -m spatial_diag qt --workspace artifacts --offscreen
```

## 关键设计

- **确定性生成**：固定种子、嵌套 R3 源目录、镜像源房间模型、固定噪声底、无时间戳 WAV/manifest/JSON。相同输入两次运行的样本、快照、报告和 PNG 逐字节一致。
- **真值来源**：生成阶段直接写入每条路径的方向、到达样本/时间和渲染后能量。参考场与误差核验只读 manifest，不从重建结果反推。
- **4 通道到球面事件**：RIR 为 `W/X/Y/Z` FOA 排列；用有功声强 `W·(X,Y,Z)` 估计到达方向，避免奇次反射的 180° 相位翻转。
- **球谐口径**：Fibonacci 等面积采样，权重 `4π/N`；实球谐为 ACN 序号的 orthonormal basis。
- **低置信**：每个方向输出有效样本数；低于阈值的方向显式标记，绝不静默进入等权拟合。
- **规模自适应**：在“全量最小二乘 / 聚类代表 / 粗到细”之间做确定性选择；记录候选、选择依据、误差上界、峰值内存、降级顺序和守恒检查。
- **逐方向验收**：报告 p50/p90/p95/max、超差比例、最差方向。跨档不一致时按 ACN 分量与方向角排序给出首个分歧方向及原因。
- **事务快照**：完整 staging → 原子 run 目录 → 最后原子发布 `CURRENT.json`。强制终止后只保留最近提交版本；未提交 staging、旧 run 和旧 view 作为孤儿回收。
- **内容寻址**：快照版本为 `sd1-<SHA-256 前 16 位>`，不依赖时钟。旧版本不会被静默复用；同版本只有内容哈希相同才允许复用。

## 默认策略结果

| 档位 | 源/球面点 | 策略 | 阶数 | 拟合行 | 峰值内存 / 预算 |
|---|---|---|---:|---:|---:|
| small | 8 / 32 | coarse_to_fine | 3 | 30 | 12,544 / 131,072 B |
| medium | 24 / 98 | full_least_squares | 6 | 63 | 77,224 / 131,072 B |
| large | 96 / 200 | clustered_representatives | 8 | 97 | 195,496 / 196,608 B |

默认配置下逐方向误差与跨档分歧的实测值见 `docs/VALIDATION.md`。跨档严格判定为不一致时，报告仍会给出首个分歧方向的坐标、主导 SH 分量和归类原因，不把总体均值当作一致性结论。

## 配置

默认模板：`spatial_diag/assets/default_config.yaml`。可复制修改后运行：

```bash
python -m spatial_diag run --workspace artifacts --config my_config.yaml
```

重要参数包括：

- `seed`、`room.wall_absorption`、`room.reflection_order`、`audio.noise_floor_dbfs`；
- 三档的 `source_positions`、`sphere_points`、`requested_order`、`direction_resolution_budget_deg`、`memory_budget_bytes`；
- `projection.min_effective_samples`、`oversampling_factor`、`ridge`；
- `validation.relative_tolerance`、`significant_relative_floor`、参考核宽；
- `consistency.relative_tolerance`。

## 测试

```bash
python -m pytest -q
```

当前结果：`15 passed`。测试覆盖 WAV 格式与真值、两次生成/运行的字节一致性、SH 正交归一、低置信排除、三档策略分化、预算变化局部影响、跨档首个分歧、快照恢复与孤儿回收、篡改检测、PNG 与 Qt 离屏视图。

## 目录

```text
spatial_diag/          核心包（生成、投影、SH、策略、核验、一致性、快照、渲染、CLI、Qt）
spatial_diag/assets/   默认 YAML 配置
tests/                 自动测试
docs/                  算法口径与验收记录
run_demo.cmd           Windows 一键演示
run_demo.ps1           PowerShell 一键演示
```
