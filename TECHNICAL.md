# 图文部件计数标注一致性核验与冲突仲裁工作台 — 技术说明

## 总览

工作台从本地确定性合成的图文样本出发，依次执行 **加载校验 → 图像检测聚合 → 统一核对账本 → 一致性判定与冲突仲裁** 四个阶段，输出可审计的 `report.json`。支持类别级增量重算、线程池并发（结果与串行一致）、阶段化快照恢复与故障注入隔离。全程离线，不依赖容器与网络。

## 目录结构

- `config.yaml` — 核验配置（枚举、容差、检测参数、裁决优先级、并发上限、全局版本号）
- `workbench/config.py` — 配置加载校验与三级指纹（整体 / 检测参数 / 仲裁参数）
- `workbench/loader.py` — 样本、描述、双标注清单加载校验层（业务码见下）
- `workbench/detect.py` — 纯函数式图像检测：颜色容差连通区域、面积/长宽比过滤、遮挡碎片合并
- `workbench/ledger.py` — 四来源（detection/description/label_a/label_b）统一账本聚合
- `workbench/arbitrate.py` — 一致/偏差/冲突判定与优先级仲裁、僵局标记
- `workbench/cache.py` — 内容指纹缓存键（检测级 + 类别级）
- `workbench/snapshot.py` — 阶段快照：临时文件 + 整体替换提交 + 版本/配置哈希校验
- `workbench/pipeline.py` — 阶段编排、线程池并发、稳定排序键、报告产出
- `workbench/edits.py` — 编辑命令序列执行器（图像替换/断言增删改/清单替换与单条修正/容差与优先级变更）
- `workbench/inject.py` — 注入配置执行与业务码映射
- `workbench/samplegen.py` + `scripts/gen_samples.py` — 固定种子确定性样本生成
- `tests/test_workbench.py` — 9 项验收测试

## 业务码映射

| 码 | 含义 | 触发 |
|---|---|---|
| `E_SAMPLE_INVALID` | 样本容器损坏 | PNG 头截断/解码失败/尺寸不符、描述 JSON 解析失败、count 非非负整数 |
| `E_LABEL_ENUM` | 枚举越界 | 类别或颜色不在配置枚举内 |
| `E_BOX_OUT_OF_RANGE` | 边界框越界 | 非整数、反向或超出图像范围 |
| `E_ORDER_GAP` | 出现次序跳号 | 清单内 order 非连续递增（首个跳号处截断并记录） |
| `E_ARBITRATION_DEADLOCK` | 仲裁僵局 | 两名标注计数相同且检测、描述、标注三者互不相同 |
| `E_SNAPSHOT_STALE` | 快照过期 | 启动时版本号/配置哈希/来源哈希不一致，或提交前清单被改写 |

失败清单按 `(sample_id, source, code)` 稳定排序；失败来源的数据被剔除，其余来源继续参与下游。

## 检测口径（可解释、可复核）

1. 每像素分配到距离最近且距离 ≤ `color_tolerance` 的枚举颜色（避免重复计数）。
2. 行主序 BFS 连通区域；按 `min_area` 与 `aspect_ratio_range` 过滤噪声。
3. 按填充率分类形状（square ≥ 0.9，circle ≥ 0.6，其余 triangle）。
4. 同色同形且边界框间隙 ≤ `merge_gap` 的碎片并查集合并，合并依据（碎片边界框、面积、颜色距离）逐条写入 `fragments`。
5. 实例按 `(y0, x0, y1, x1, color, category)` 排序并赋稳定 `instance_id`；产物写入图像内容指纹与检测参数指纹。检测阶段不读取描述与标注的任何字段。

## 账本与仲裁

- 账本条目 = `(sample, category, color, source)`，携带整数计数、来源指纹、合并明细；计数为零的条目保留并显式 `zero: true`。
- 判定：四来源计数极差 ≤ `count_tolerance` 记为 `consistent`（保留各来源偏差量）；否则进入冲突处理。
- 僵局优先于优先级：标注 A=B 且三者互不相同 → `deadlock` + `E_ARBITRATION_DEADLOCK`，不任意选取一方。
- 冲突按 `arbitration_priority`（detection / description / label_majority / label_a）取首个适用规则给出仲裁计数，被否决来源及其偏离量写入 `rejected`。

## 类别级增量失效

- 检测缓存键 = `H(样本, 图像内容指纹, 检测参数指纹)`。
- 类别缓存键 = `H(样本, 类别, 图像指纹, 该类描述断言指纹, 该类标注A指纹, 该类标注B指纹, 配置指纹)`。
- 单条描述修正只失效对应 `(样本, 类别)`；整份标注替换只失效条目实际变化的类别；图像替换失效该样本全部类别；容差/优先级变更失效全部类别条目但检测直接复用。未失效条目逐字节复用，复用/重算清单与全量运行逐项对齐（测试逐步断言）。

## 并发与快照

- 任务排序键：`detect:{sample_id}`、`ledger:{sample_id}:{category}`；线程池上限取自 `max_concurrency` 且不可超过；汇总后按排序键重排，`--verify-serial` 以单线程重算并逐字段比对，结论写入 `concurrency.serial_consistent`。
- 快照：四阶段产物先写临时文件再 `os.replace` 整体替换；清单记录全局版本号、配置哈希、来源清单哈希集合与已完成样本游标；提交前重新读取清单，版本变化即抛 `StaleCommit`（输出 `E_SNAPSHOT_STALE`），不做逐文件覆盖。

## 使用

```bash
pip install -r requirements.txt
python3 scripts/gen_samples.py                      # 生成确定性样本
python3 -m workbench.cli run --config config.yaml --data data --out out \
    --cache out/cache.json --snapshots out/snapshots --verify-serial
python3 -m workbench.cli apply-edits --config config_edited.yaml --data data_edited --edits data/edits.json
python3 -m workbench.cli run --config config_edited.yaml --data data_edited --out out2 \
    --cache out/cache.json --incremental
python3 -m workbench.cli inject-run --config config.yaml --data data \
    --injections data/injections.json --out out_inject
python3 -m pytest tests/ -q
```

## 实际验证结果（本机）

- `pytest`：9 项全部通过（全量账本/仲裁/僵局、加载校验四业务码、检测确定性、逐编辑增量等价、并发=串行、快照过期与恢复、提交时版本检查、注入映射、优先级变更）。
- 基线全量：8 样本、48 条账本、12 条仲裁、s04 僵局 `E_ARBITRATION_DEADLOCK`、s02 遮挡碎片合并（2 碎片→1 实例）、s05 零计数条目保留、`serial_consistent: true`。
- 编辑序列（8 步）后增量运行与空白全量运行逐字段一致；图像替换仅重算 s07 检测，容差变更时 7 个样本检测全部复用。
- 注入 4 项全部命中并映射：`INJ-001→E_SAMPLE_INVALID`、`INJ-002→E_LABEL_ENUM`、`INJ-003→E_ORDER_GAP`、`INJ-004→E_BOX_OUT_OF_RANGE`。
