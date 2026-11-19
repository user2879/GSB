# 三维网格缺陷生成、并行修复与独立验证

本项目在本地确定性数据上完成：缺陷样本生成、真值清单输出、重复顶点焊接、连通分量分块、预算约束下的块级并行修复、共享边界裁决、确定性合并、质量核对与独立文件级验证。全部步骤由一条命令串行编排，不使用 Docker、容器环境或外部数据源。

当前实现与验收结果已完成。实际计算环境为 WSL2 Ubuntu 22.04 上的 CPython 3.11.0rc1；代码只使用跨平台 Python、NumPy 与文件 API，Windows 命令见下文。

## 环境准备

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import numpy, trimesh; print(numpy.__version__, trimesh.__version__)"
```

本次实际安装版本：

- Python 3.11.0rc1
- NumPy 2.4.6
- trimesh 4.12.2
- pytest 9.2.0

Linux/WSL 对应命令：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -c "import numpy, trimesh; print(numpy.__version__, trimesh.__version__)"
```

## 一键运行

```bash
.venv/bin/python -m mesh_repair run \
  --config configs/default.json \
  --out artifacts/acceptance/default
```

Windows：

```powershell
.\.venv\Scripts\python.exe -m mesh_repair run --config configs\default.json --out artifacts\acceptance\default
```

独立验证器只读取最终网格和真值清单：

```bash
.venv/bin/python -m mesh_repair validate \
  --mesh artifacts/acceptance/default/merged.ply \
  --truth artifacts/acceptance/default/ground_truth.json \
  --out artifacts/acceptance/default/reloaded_verifier.json
```

## 项目结构

- `mesh_repair/generator.py`：三种闭合基础形体与五类缺陷注入。
- `mesh_repair/io_ply.py`：确定性二进制 little-endian 三角 PLY 读写。
- `mesh_repair/preprocessing.py`：容差焊接、连通分量分块、边界顶点清单。
- `mesh_repair/repair.py`：全局只读拓扑分析、线程池块修复、按块键合并。
- `mesh_repair/boundary.py`：共享顶点登记与字典序单写者裁决。
- `mesh_repair/quality.py`：管线侧体积、面积、拓扑、真值动作对齐检查。
- `mesh_repair/verification.py`：只消费 PLY 与真值的独立验证器。
- `mesh_repair/pipeline.py`：固定序列、报告、步骤哈希和错误码编排。
- `configs/`：默认、低预算降级、预算拒绝配置。
- `tests/`：16 项自动化测试。
- `artifacts/acceptance/`：本次实际运行产物、报告和哈希。

## 固定序列

1. 用固定种子生成 `baseline.ply`、`defective.ply`、`ground_truth.json`。
2. 在焊接容差内合并重复顶点，写出 `welded.ply`。
3. 按共享边连通分量分块；超过上限的块按排序后的面索引固定切片，写出 `chunks.json`。
4. 进行只读全局拓扑分析，生成删除原因、自相交对、非流形边处理记录与朝向翻转计划。
5. 工作线程池按块并发执行；结果按块键收集，而不是按完成顺序。
6. 边界仲裁器按 `(vertex, block_key)` 排序提交共享顶点。
7. 按原始面索引排序合并，写出 `merged.ply` 与 `repair_actions.json`。
8. 独立验证器重新读取文件和真值，写出 `verifier.json`。
9. 交叉核对管线与验证器结论。不一致时返回 `E_VERIFIER_MISMATCH`。
10. 写出 `report.json`、`hashes.json`；两次同配置运行逐字节一致。

## 缺陷与真值

基础网格包含三个独立闭合分量：icosphere、圆环和带开放凹槽的挤出立方体。默认样本为 562 顶点、1116 面。

| 缺陷类型 | 注入方式 | 修复动作 |
| --- | --- | --- |
| `non_manifold_edge` | 在既有边上追加外侧鳍面，使边关联面数变为 3 | 保留两条主面片，删除高边界度多余面片 |
| `duplicate_vertex` | 追加与既有顶点完全重合的未引用顶点 | 焊接至最小原索引，登记被移除顶点 |
| `zero_area_face` | 追加三个共线新顶点形成的零面积三角面 | 按面索引删除 |
| `isolated_component` | 追加独立的 4 面四面体小连通块 | 删除低于阈值的小分量 |
| `self_intersection` | 追加一对无共享顶点、真实相交的三角面 | 成对剔除自相交面 |

真值 JSON 中每个注入对应一条记录，包含 `defect_id`、`type`、`faces`、`vertices`、`criterion` 和必要元数据。默认配置 `injection_count = defect_record_count = 7`，逐类计数为 1、3、1、1、1。面与顶点索引均指向注入后的 PLY。

## 确定性与并行设计

### 焊接与分块

- 焊接按原顶点索引递增处理，并查集根始终取最小索引，因此规范顶点位置不依赖输入遍历之外的因素。
- 分块先按共享边连通分量划分，再对超限分量按排序面索引执行固定长度切片。
- 每个块的 `boundary_vertices` 是“同分量内被多个块共同引用”的顶点；`chunks.json` 同时记录全局共享边界集合。

### 预算调度

调度器在提交任务前使用确定性成本模型进行准入：

- 单块内存估算：`4096 + 96 * vertex_count + 192 * face_count` 字节。
- 单块耗时估算：`0.02 + (face_count + 0.5 * vertex_count) / 1000 * (1 + 0.05 * (threads - 1))` 毫秒。
- 固定降级优先级：先检查 `time_budget`，再检查 `memory_budget`。
- 线程数每次减一，并记录从多少降到多少、触发原因、当时的估算耗时和峰值内存。
- 串行执行仍不满足预算时抛出 `E_BUDGET_DENIED`；任何降级路径都必须成功返回全部块结果。

这里的“耗时”是有意的确定性复杂度预算而非墙钟采样，避免机器负载导致两次运行报告不一致；实际块结果仍全部执行并核对块键集合。

### 共享边界

每个块对边界顶点先登记 `VertexProposal(block_key, vertex, position, keep)`。全部块完成后，仲裁器按 `(vertex, block_key)` 排序：

- 位置以最小块键的登记为规范位置；位置差超过容差时记录冲突。
- 保留状态采用逻辑或，即任一保留面使用该顶点则保留。
- 最终合并阶段只由仲裁器写顶点表，块结果不会直接修改共享顶点。

默认样本的 169 个共享顶点产生 347 条登记，位置冲突为 0，并出现 2 个可确定性解决的保留状态分歧。

## 修复正确性与验证

修复顺序为：零面积面、自相交面、非流形边、孤立小分量、朝向一致性。所有删除动作保留原面索引与原因；合法面索引区间为 `[0, baseline.face_count)`，误删会写入 `misdeleted_legal_faces`。默认样本的合法面坐标多重集与基准完全一致。

管线检查：

- 非流形边数为 0。
- 边界边数为 0，连通分量数回到基准的 3。
- 每条双面共享边的绕向相反，面法线一致。
- 自相交面对数为 0。
- 体积、面积相对偏差在声明容差内。
- 输出面数等于基准面数，所有真值缺陷都有对应修复动作。

独立验证器只读取 `merged.ply` 和 `ground_truth.json`，重新实现边关联、连通分量、绕向和文件哈希检查；不使用管线内存中的块、计划或动作对象。若与管线结论或公共检查状态/指标不一致，固定序列返回 `E_VERIFIER_MISMATCH`，报告首个不一致检查及其面、边或顶点定位。

## 业务错误码

| 错误码 | 退出码 | 触发场景 |
| --- | ---: | --- |
| `E_MESH_FORMAT` | 10 | PLY/真值格式错误、截断、越界索引 |
| `E_BLOCK_KEY` | 11 | 块键重复、缺块、面覆盖不完整或块索引越界 |
| `E_BOUNDARY_CONFLICT` | 12 | 边界登记非法或保留面所需的共享顶点被裁决丢弃 |
| `E_BUDGET_DENIED` | 13 | 串行执行也无法满足内存或单块时间预算 |
| `E_VOLUME_DEVIATE` | 14 | 体积相对偏差超容差 |
| `E_RESIDUAL_TOPOLOGY` | 15 | 残余非流形边、自相交、边界、分量或法线异常 |
| `E_VERIFIER_MISMATCH` | 16 | 管线与独立验证器结论或公共指标不一致 |

命令异常链路也会写出 `report.json` 和 `hashes.json`；预算拒绝时保留已生成的生成、焊接和分块产物，不伪造未执行的修复结果。

## 实际验收结果

默认运行：

| 产物/指标 | 值 |
| --- | --- |
| 状态 | `pass`, `OK`, exit 0 |
| 缺陷注入/真值记录 | 7 / 7 |
| 分块 | 18 块，7 个输入分量，169 个共享边界顶点 |
| 修复后 | 562 顶点、1116 面、3 个分量 |
| 非流形边/自相交/不一致绕向 | 0 / 0 / 0 |
| 体积 | 7.333367234260094，相对偏差 0 |
| 面积 | 36.381014996182095，相对偏差 0 |
| `baseline.ply` SHA-256 | `ae38f05d5e5c5dfb614745aad73c8e6c47e5e6969296f31b6c82120717ff3cb1` |
| `defective.ply` SHA-256 | `6942b2c720ae3733390ad3bde4bcdd37e8c175cdecaaf7fde1905e2655ecf6e9` |
| `ground_truth.json` SHA-256 | `99f54a2defb47869aa76a7a981e1c0526b4fe0d12711de784742b099c8050a56` |
| `welded.ply` SHA-256 | `a529f97c591c51896d2dd5b15b5091962c66ad590db7a77e855f17d43b8ce30c` |
| `chunks.json` SHA-256 | `f39760262b8fa1eb61dfac92e4a6eb30773f84717ab0358e09881da854e4950b` |
| `merged.ply` SHA-256 | 与 `baseline.ply` 相同：`ae38f05d...cb1` |
| `verifier.json` SHA-256 | `b044b6db2b017b266d479b3ab09aa30147b235f4fbe98793e7e2a3da0b7969a8` |
| `report.json` SHA-256 | `0b966376cab3847b69b2e1d2beb9dc8cd1fc5b124aae1dffb4d10a3a8dc22d74` |

预算降级运行 `artifacts/acceptance/low_budget/`：请求 8 线程，`time_budget` 依次将 8→7→6→5→4，`memory_budget` 再将 4→3；最终 3 线程完成全部 18 块，状态 `pass`，`merged.ply` 哈希仍为 `ae38f05d...cb1`。报告哈希为 `4147303ae09f33fa15e09c3ed9d1e03ded7d61aa8c36be25d8f9896acd07a43e`。

预算拒绝运行 `artifacts/acceptance/denied_budget/`：最大单块估算 30208 字节，大于 20000 字节预算，返回 `E_BUDGET_DENIED`/exit 13；报告哈希为 `deb55ee6356100078ee974a9fb55aaaa67b0a84de9630245df726b6f7d130561`。

确定性复核：

- `artifacts/acceptance/determinism.json` 记录默认配置两次运行的全部 10 个产物逐字节一致，包括 `report.json` 和 `hashes.json`。
- `artifacts/acceptance/thread_invariance.json` 记录 1、2、8 线程三次运行均通过，`merged.ply` 哈希全部为 `ae38f05d...cb1`。
- `pytest`：16 项测试全部通过。

## 复现验收

```bash
.venv/bin/python -m pytest -q

.venv/bin/python -m mesh_repair run --config configs/default.json --out artifacts/acceptance/default
.venv/bin/python -m mesh_repair run --config configs/low_budget.json --out artifacts/acceptance/low_budget
.venv/bin/python -m mesh_repair run --config configs/denied_budget.json --out artifacts/acceptance/denied_budget
```

每个运行目录中的 `report.json` 给出完整分步结果，`hashes.json` 给出各步骤 SHA-256、大小和报告哈希，`verifier.json` 给出独立复核判据与首个失败定位。
