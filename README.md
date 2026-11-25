# MG-Recon：定点多视图重建增量诊断工具

这是一个可直接在 Linux/WSL Ubuntu 本地运行的多视图几何重建诊断项目。它不依赖外部数据集或网络资源，首个版本包含代码、配置、确定性样本、编辑命令、阶段快照、诊断报告和自动化测试。

项目重点不是交换格式，而是以下三件事的真实落地：

- 三角化点残差与深度图一致性联合驱动的自适应体素细分；
- 三角化点、深度图像素、细分父子体素三类依赖上的精确缓存失效；
- 任意并发度、预算耗尽时机和编辑序列下，增量结果与空白全量重建逐字段一致。

## 快速开始

环境要求：Python 3.10+、NumPy；Tkinter 用于图形界面。

```bash
# WSL Ubuntu 如需 Tk
sudo apt update && sudo apt install -y python3-tk

python3 -m pip install -r requirements.txt

# 生成确定性本地样本与配置
python3 -m mgirecon init --project demo --force

# 单次全量运行
python3 -m mgirecon run --project demo --fresh

# 完整验收演示（基线、连续编辑增量、空白全量对照、异常码样例）
python3 -m mgirecon demo --project demo

# 图形界面（当前无 libtk 的环境会输出 E_GUI_UNAVAILABLE 及文本诊断）
python3 -m mgirecon gui --project demo

# 自动化测试
python3 -m unittest discover -s tests -t . -v
```

验收演示会生成：

- `reports/demo_summary.json` / `reports/demo_summary.txt`
- `reports/demo_report.json`
- `reports/baseline.json`
- `reports/gui_fallback.json`
- `demo/.mgirecon/current/manifest.json` 与各阶段快照

## 目录

```text
mgirecon/
  config.py       配置、矩阵校验、哈希、错误聚合
  pngio.py        自包含 PNG RGB8 读写与容器校验
  synth.py        固定种子本地多视图样本生成
  features.py     分块颜色统计与梯度特征
  matching.py     固定顺序匹配、内点计数、并列决断
  geometry.py     定点投影、整数正规方程三角化、量化
  depth.py        全分辨率深度图、量化舍入方向、边界冲突
  voxel.py        残差驱动细分、父子/邻接关系、复用账本
  strategy.py     视图规模自适应策略表
  scheduler.py    有界线程池与稳定归并
  engine.py       阶段编排、预算、缓存复用、失效传播
  snapshot.py     临时目录写入后整体替换的快照
  commands.py     编辑命令流校验与应用
  report.py       机器可读与文本诊断
  gui.py          Tkinter 诊断界面
  cli.py          命令行与完整演示
demo/             已生成的确定性样本、配置、命令与快照
reports/          实际验收报告
tests/            18 项自动化测试
```

## 当前实际验证结果

本工作区使用 Python 3.11.0rc1、NumPy 2.4.6 实际运行。

```text
$ python3 -m unittest discover -s tests -t . -v
Ran 18 tests in 30.596s
OK
```

完整演示最终结果：

```text
incremental == full: True
incremental digest: a74737ff872804986b96031cc6c8060fa35674e8968267d45008678e2e41f7cb
full digest:        a74737ff872804986b96031cc6c8060fa35674e8968267d45008678e2e41f7cb
voxels=416, subdivided=16, max_depth=4
depth distribution={"0":284,"1":28,"2":28,"3":28,"4":32}
reused entries=455, invalidated entries=8
relations seen across edits: camera, content, depth, feature, hierarchy, new,
  pair, parameter, point, quantization, removed
exception samples: E_PARAM_INVALID, E_RECON_BUDGET_EXCEEDED,
  E_VIEW_UNKNOWN, E_VOXEL_ALIGN
```

当前容器缺少 `libtk8.6.so`，因此 Tk 窗口无法实例化。界面代码已交付，命令行会稳定输出：

```text
[E_GUI_UNAVAILABLE] Tkinter is not available in this runtime:
libtk8.6.so: cannot open shared object file: No such file or directory
```

安装 `python3-tk` 并有可用显示服务后，运行 `python3 -m mgirecon gui --project demo` 即可打开诊断界面。

## 配置与样本

`demo/config.json` 固定保存场景种子、分辨率、每个视图的图像路径、渲染种子、3x3 整数内参、3x4 整数外参，以及匹配、重建和调度参数。

- 图像为 RGB8 PNG，固定 `128x96`；PNG 签名、IHDR、CRC、颜色类型、位深和解压尺寸均校验。
- 内参必须为可逆整数矩阵；外参旋转部分必须满足整数正交条件。
- 视图 ID 与图像路径各自唯一，映射一一对应。
- `min_disparity`、`residual_threshold` 必须为正整数。
- 体素尺寸必须按三轴整除重建范围，否则保留 `E_VOXEL_ALIGN` 并跳过体素阶段。
- 非法视图被隔离，其余视图继续进入后续阶段。

## 编辑命令

命令流是 JSON 数组，或形如 `{"commands": [...]}` 的对象：

```json
[
  {"op": "replace_view", "view": "v02", "render_seed": 9022},
  {"op": "set_extrinsics", "view": "v03",
   "value": [[1, 0, 0, 10], [0, 1, 0, 5], [0, 0, 1, 198]]},
  {"op": "set_param", "name": "min_disparity", "value": 15},
  {"op": "set_param", "name": "residual_threshold", "value": 14},
  {"op": "set_param", "name": "quantization_level", "value": 2},
  {"op": "set_param", "name": "voxel_size", "value": 16}
]
```

可编辑参数包括 `min_disparity`、`residual_threshold`、`quantization_level`、`voxel_size`、`voxel_max_depth`、`min_occupancy_points`、`global_point_budget`、`per_view_point_quota`、`max_workers`、`min_inliers`、`max_pairs`。

未知视图输出 `E_VIEW_UNKNOWN`；非法矩阵、未知参数或非法值输出 `E_PARAM_INVALID`。命令出错后继续处理后续命令。

## 定点与数值规则

- 世界坐标使用 Q(`fixed_bits`) 整数，默认 Q10。
- 投影、三角化、残差、深度、体素评分全部只用 Python 整数。
- 三角化以整数正规方程与 Cramer 法则求解，记录每个坐标的求解舍入方向。
- 量化使用 `quantum = level * 2^(fixed_bits-4)`，记录 `exact/up/down` 方向。
- 三维点量化后重新投影，残差为各视图像素误差平方和。
- 深度图逐像素保存深度整数、舍入方向、边界标记、来源点和像素缓存键。
- 深度冲突在投影距离近似并列且深度差不超过半个量化步长时，固定选择较近深度；深度相同则按稳定点标识决断，并标记边界像素。

## 自适应体素与失效

体素评分：

```text
score = mean(point.residual) + depth_spread / quantum
```

`score >= residual_threshold` 时递归产生 8 个子体素，直到达到残余阈值或策略细分深度。父节点只记录 `SUBDIVIDED`，其 `direct_occupied` 为 `null`；叶节点独立给出 `SOLID/EMPTY`，父子占用不会互相借用，子结论也不会回写父结论。

缓存键覆盖：

- 特征：图像 SHA、分辨率、分块精度、描述子精度、定点位宽；
- 匹配对：左右特征键、最小视差、残差阈值、匹配范围、量化档位；
- 三角化点：匹配对键、左右相机哈希、量化档位、量子；
- 深度像素：视图、坐标、来源点键、量子、冲突规则；
- 体素：点键集合、深度像素键集合、阈值、量子、范围、父路径、子键和细分顺序。

失效关系显式保存在账本中：`content`、`camera`、`feature`、`pair`、`point`、`depth`、`quantization`、`hierarchy`、`parameter` 和 `removed`。未失效体素直接复制子树，并为每个复用节点记录点键摘要、深度键摘要、父子结构摘要和命中缓存键。

## 预算、并发与确定性

- 特征、匹配对和根体素判定均通过有界线程池执行，最大并发取自配置。
- 线程返回结果按视图 ID、匹配对 ID、根体素路径稳定归并。
- 三角化点按 `pair_id + 左特征坐标 + 右特征坐标` 的稳定键消费全局预算和单视图配额。
- 预算或视图配额耗尽时停止消费新点，记录 `E_RECON_BUDGET_EXCEEDED`，并把未完成项保留到待办队列。
- 测试覆盖 1/4 线程结果一致、低预算重复运行一致、增量与空白全量摘要一致。

## 快照与恢复

每次运行先将所有阶段产物、待办队列和清单写入 `.mgirecon/snapshot-tmp-*`，成功后整体替换 `.mgirecon/current`。清单记录：

- 快照 schema 与全局版本号；
- 配置哈希、相机哈希；
- 已完成视图和体素路径；
- 已消耗预算和完整预算摘要；
- 特征、匹配、三角化、深度、体素和账本各阶段摘要；
- 待办数量、错误码和状态。

启动时以 schema 版本、配置哈希和阶段缓存键决定续跑、增量或空白重建。快照损坏时输出 `E_SNAPSHOT_INVALID` 并回退到全量路径。

## 集成 API

```python
from mgirecon.config import load_config
from mgirecon.engine import reconstruct
from mgirecon.snapshot import load_snapshot, write_snapshot

root = "demo"
config = load_config(root)
previous = load_snapshot(root)["state"]
result = reconstruct(root, config, previous_state=previous)
write_snapshot(root, result)
```

`result` 中的 `features`、`match_pairs`、`points`、`depth_maps`、`voxels`、`ledger`、`pending`、`budget`、`scheduler` 和 `errors` 均可直接序列化或供上层集成。

更多算法与验证细节见 `TECHNICAL.md`。
