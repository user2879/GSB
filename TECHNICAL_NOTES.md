# 多视图三维重建诊断原型说明

本目录是一个完全本地、确定性的多视图几何重建诊断原型。它不依赖外部数据集、网络服务或容器；样本 PNG、相机真值、配置、内容寻址账本、原子快照和验收报告均在本地生成。

## 快速运行

```bash
python scripts/check_environment.py
python -m mdr.cli init-sample --root .
python -m mdr.cli run --root .
python -m unittest discover -s tests -v
python scripts/run_acceptance.py
python -m mdr.tk_app --root .
```

应用示例编辑命令流：

```bash
python -m mdr.cli apply commands/acceptance-edits.json --root .
```

验收结果写入：

- `reports/acceptance-report.json`
- `reports/acceptance-report.md`
- `reports/diagnostic-report.json`
- `reports/environment-report.json`

## 目录结构

- `mdr/config.py`：JSON 配置默认值、正整数/枚举/预算校验。
- `mdr/validation.py`：PNG 头、分辨率、像素模式、K/R/t、视图序号和视图引用校验。
- `mdr/samplegen.py`：固定整数种子生成 96×72 RGB PNG、几何标记、纹理块、内参/外参和三维真值。
- `mdr/features.py`：分块颜色均值、梯度统计、定点描述符、稳定 NMS 和内容指纹。
- `mdr/matching.py`：仅匹配声明视图对，按视图序号和 `(x,y)` 字典序稳定遍历/计数。
- `mdr/geometry.py`、`mdr/triangulation.py`：整数射线、定点最小二乘三角化、重投影残差、视差剔除和舍入方向。
- `mdr/depth.py`：全分辨率定点深度图、逐像素舍入码、半步冲突规则和反向投影。
- `mdr/voxels.py`：空间归属、残差/深度一致性合成、八叉树细分、体素置信度和预算待办。
- `mdr/scheduler.py`：`ThreadPoolExecutor`、稳定排序键和全局点/单视图配额分配。
- `mdr/ledger.py`：内容寻址 JSON 产物、`events.jsonl`、临时目录提交、全局版本和配置哈希快照。
- `mdr/pipeline.py`：端到端编排、缓存命中/重建/失效事件和规范状态。
- `mdr/diagnostics.py`：高一位宽、无量化参考路径和四环节误差归因。
- `mdr/acceptance.py`：增量/全量、并发/预算、恢复和失效传播验收。
- `tests/test_pipeline.py`：固定点、冲突、视差、预算扫描、校验和确定性测试。

## 确定性规则

- 定点缩放为 `S = 2 ** frac_bits`；`Q0/Q1/Q2` 对应配置中的量化移位。
- 除法和量化的半步平局统一向零舍入，每个点/像素保存 `exact/up/down` 或数字码 `0/1/2`。
- 深度冲突规则为：至少两个候选且原始深度差满足 `0 < spread <= max(1, quant_step // 2)` 时标记 `flag=1`，按“最小深度 → 视图顺序 → 像素字典序”决断。
- 所有最终列表均显式排序；不使用运行时全局随机状态，不依赖数组归约或线程完成顺序。
- 线程只并发执行无共享状态的计算，预算消耗和结果归并仍在主线程按稳定键完成。

## 缓存与精确失效

每个阶段的 `cache_key` 均由上游内容指纹、相机/参数指纹、位宽和量化档位派生：

- 视图：PNG SHA-256 + 提取参数指纹。
- 匹配对：两个视图特征键 + 匹配参数指纹。
- 三角化点：匹配键 + 两相机指纹 + 视差/定点/量化参数。
- 深度图：相机指纹 + 实际投影/覆盖该视图的点键集合 + 深度参数。
- 体素：父体素键、边界、三角化点键集合、`像素引用 + depth_q + depth_map_key` 集合和体素参数。

`ledger/events.jsonl` 和当次状态中的 `cache_events` 逐条记录 `hit/rebuilt/invalidated` 及依据。`canonical_ledger` 是与最终状态对应的规范化依赖清单；冷全量和热增量的该清单及所有业务产物逐字段一致。物理 hit/miss 历史描述执行过程，冷启动与增量运行天然不同，因此验收比较的是规范化依赖/缓存键清单，而不是把过程事件伪装成相同。

## 预算与待办

配置 `configs/reconstruction.json` 中：

- `budgets.global_point_budget`：全局允许进入深度/体素阶段的三角化点数。
- `budgets.view_point_quota`：每个视图在点对中消耗的稳定配额。
- `budgets.max_subdivisions_per_batch`：单批允许的父体素细分次数。
- `budgets.max_workers`：线程池上限，任何执行路径不会超过该值。

预算按稳定键顺序分配；耗尽后单元写入 `pending`，业务码为 `E_BUDGET_PENDING`。快照 `state.pending` 与 `manifest.pending_count` 持久化待办；下一次运行把旧待办 ID 作为优先键。

## 快照与崩溃恢复

提交过程为：

1. 在 `ledger/snapshots/.staging-*` 写 `state.json`。
2. 计算状态哈希并写 `manifest.json`。
3. 将临时目录原子改名为 `snapshot-vNNNNNN-<confighash>`。
4. 原子替换 `ledger/snapshots/current.json`。

恢复只信任 `current.json` 指向且清单哈希正确的已提交快照。残留 `.staging-*` 会被忽略；当前指针损坏、状态哈希不符或 schema 版本变化时判定 `rebuild`。被 `SIGKILL` 强杀时可能来不及生成退出报告，因此恢复结论只来自已提交快照。

## 误差归因

`mdr/diagnostics.py` 运行 `frac_bits + 1`、`Q0` 无量化参考路径，并按相同体素 ID 比较：

1. `feature_quantization`：定点颜色统计/描述符和得分差。
2. `triangulation_residual`：三维定点坐标和重投影残差归一化差。
3. `depth_consistency`：逐深度值、MAD 和不一致合成分数差。
4. `voxel_subdivision`：合成体素分数、状态和子体素集合差。

每个体素输出四类差值、容差、首个超过容差的环节、是否发生判定翻转以及 `flip_source`。边界探针会确定性选择一个分数边界体素并把阈值设到两路径分数之间，以复现和解释翻转。

## Tkinter 环境说明

当前基础环境缺少系统 `libtk8.6.so`，且无免密 sudo。已验证两种处理：

- 标准方式：`sudo apt-get install -y python3-tk`。
- 无 root 方式：`bash scripts/install_local_tk.sh`，脚本通过 `apt-get download` 解压 Tcl/Tk 到 `vendor/tk`；`mdr/tk_support.py` 会自动预加载。

`python -m mdr.tk_app --smoke` 已在本机完成窗口创建/销毁冒烟。
