# 多光谱瓦片分类与跨瓦片融合调度工作台 — 技术说明

## 组成

| 模块 | 职责 |
| --- | --- |
| `workbench/loader.py` | 瓦片清单与分层波段加载校验（E_TILE_SHAPE / E_ADJ_ASYMMETRIC / E_BAND_VERSION_UNKNOWN / E_BAND_MISSING / E_BAND_CORRUPT / E_ADJ_UNKNOWN_REF / E_TILE_DUPLICATE），失败瓦片进失败清单、不中止运行 |
| `workbench/classify.py` | 窗口化像素级分类：核心区分块互不重叠、上下文重叠仅参与特征；植被/水体指数、纹理梯度、高程坡度四特征 |
| `workbench/fixedpoint.py` | 定点整数打分：`scale = 2^frac_bits`，所有特征/得分按量化档位 `quantum` 吸附（round-half-away-from-zero），固定权重顺序累加，同分按类别编码升序 |
| `workbench/fusion.py` | 重叠带加权投票（fs-v1 线性距离 / fs-v2 平方距离 × 两侧定点类概率），平票按瓦片 (row,col) 字典序决断；最小连通域剔除与邻接多数回填（逐条记录依据）；全局栅格装配 |
| `workbench/fingerprints.py` | 瓦片缓存键 = SHA256(波段内容 ‖ 波段集合版本 ‖ 指数权重指纹 ‖ 定点参数)；融合带指纹 = SHA256(两侧缓存键 ‖ 类概率摘要 ‖ 重叠几何 ‖ 融合策略版本 ‖ 定点参数) |
| `workbench/scheduler.py` | 三重约束调度：计划阶段完全串行（预算扣减序列、中止/待办/保留清单与并发度无关），执行阶段线程池并行 |
| `workbench/snapshot.py` | 阶段化快照：产物先写临时文件再 `os.replace` 整体替换，manifest_state 最后提交 |
| `workbench/pipeline.py` | 编排：加载校验 → 逐瓦片分类 → 重叠带投票 → 后处理四阶段；增量失效判定；强制终止恢复 |
| `workbench/edits.py` / `workbench/inject.py` | 编辑命令流应用（改样本数据）/ 故障注入（仅作用于内存副本与字节流，不改被测源码与磁盘样本） |
| `workbench/samplegen.py` | 固定种子样本生成：世界真值图与瓦片行列解耦（固定 8×8 瓦片世界），任一瓦片可独立再生成 |

## 关键确定性规则

- 每个瓦片任务稳定排序键为 `(row, col, tile_id)`，融合带任务为邻接对 `(rowA, colA, rowB, colB)`；计划阶段的预算扣减、待办与保留顺序只依赖该键，线程池只改变完成顺序。
- 全局预算按排序键顺序扣减；瓦片像素数 > 单瓦片配额 → `E_TILE_BUDGET_EXHAUSTED`（中止）；> 全局剩余 → `E_GLOBAL_BUDGET_EXHAUSTED`（保留待办，下轮续跑）；融合带超出每批上限 → 按邻接对顺序保留并记 `E_MERGE_BATCH_PENDING`。
- 类概率为定点伪概率：`(score - min) * SCALE // (Σ + 1)`；融合投票、边界判定全程 int64。
- 边界敏感：重叠像素最高/次高票差 ≤ `quantum/2` 时输出双精度参考投票；判定不一致记 `E_BOUNDARY_SENSITIVITY` 并给出翻转对端。
- 水平与垂直缝合带交叉的角点像素：按融合对稳定顺序后者覆盖前者，两次投票明细均留痕。
- 快照 manifest 记录全局版本号、配置哈希、波段集合版本、融合策略版本、逐瓦片结果版本与逐融合带版本；版本号均为内容哈希前缀，与运行路径无关，保证增量与全量可逐字段比对。
- 强制终止：SIGTERM/SIGINT 或 `stop_after` 触发后，当前已完成瓦片/融合带产物落盘并提交 manifest 与待办队列，以退出码 75 退出，不生成报告；恢复仅依赖已持久化快照与待办队列。

## 一致性语义

- 增量==全量比较覆盖：全局类别栅格、逐瓦片类概率摘要、重叠像素投票明细、剔除回填清单、失效与复用账本（缓存键/版本）、待办与保留队列、失败清单顺序、边界事件、混淆统计。
- 账本的 `origin`（reused/recomputed）与预算消耗轨迹属于执行路径元数据，按定义不参与内容等价比较；同一配置与样本重复运行则全部字段（含轨迹）逐字段可复现（`test_report_reproducible`）。

## 实际验证结果（本机，Python 3.11 + numpy 1.26 + pillow 11）

- `pytest tests -q`：**26 passed**（加载校验 6、分类 4、融合 3、调度 4、快照恢复 2、增量一致性 3、诊断 4）。
- 主样本（3×3 瓦片，64px，重叠 8px）全量运行：0 失败，全局栅格 176×176，混淆准确率 29343/30976 ≈ 94.7%，边界敏感事件 8 起，碎小类别回填 627 条。
- 编辑序列（波段补丁 → 指数权重 → 融合策略 fs-v2 → 邻接修改 → 增瓦片 → 删瓦片）逐步增量运行后 vs 空白全量重建：`INCREMENTAL==FULL: True`。
- 三重预算：单瓦片配额 512（瓦片 1024 像素）→ 全部 `E_TILE_BUDGET_EXHAUSTED`；全局预算 2048 → 2 片分类 + 2 片 `E_GLOBAL_BUDGET_EXHAUSTED` 待办，下轮续跑后等于全量；`merge_batch_limit=1` → 4 个融合带分批 4 轮收敛，结果等于全量。
- 并发一致：线程池 1 vs 4 的预算扣减序列、待办/保留/中止清单、全部内容字段完全一致。
- 快照恢复：`--stop-after-tiles 3` 以退出码 75 终止且不产出报告，续跑结果与不间断全量一致（全局版本 `e9adeba90536…` 相同）。
- 注入隔离：INJ-001 截断波段（E_BAND_CORRUPT）、INJ-002 非有限值（E_BAND_NONFINITE）、INJ-003 幽灵邻接（E_ADJ_UNKNOWN_REF）、INJ-004 负最小连通像素（E_CONFIG_INVALID）全部命中并写入报告，未命中瓦片正常分类。

## 复现命令

```bash
python3.11 -m venv .venv --without-pip --system-site-packages  # 本机 ensurepip 缺失
pip install numpy pillow pytest
python -m workbench.cli gen-samples --out samples
python -m workbench.cli run --samples samples --work work --report work/report.json
python -m workbench.cli run --samples samples --work work_inj --inject samples/inject.json
python -m workbench.cli run --samples samples --work work_stop --stop-after-tiles 3  # 退出码 75
python -m workbench.cli run --samples samples --work work_stop                      # 恢复续跑
python -m workbench.cli run-edits --samples <副本> --work work_edit                 # 增量==全量
python -m pytest tests -q
```
