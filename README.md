# MVRV Diagnostics：多视图三维重建与多分辨率体素占位诊断

一个完全本地、确定性、可复核的 Python 项目。它从固定整数种子合成多视图 RGB PNG、相机内外参和编辑命令流，在定点整数域依次执行特征提取、稳定匹配、三角化、深度图生成、残差驱动体素细分与占位判定，并提供精确缓存失效账本、稳定并行调度、预算/配额控制、原子快照恢复和 Tkinter 诊断界面。

原始任务说明保存在 `docs/requirements.md`。

## 环境准备

WSL Ubuntu：

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-tk
python3 -m venv .venv
source .venv/bin/activate
pip install 'numpy>=1.24'
```

也可以直接使用系统 Python 3.10+，仅额外依赖 `numpy>=1.24`。运行时不访问网络、不使用 Docker、不依赖外部数据集；PNG 读写由标准库 `zlib/struct` 和 NumPy 完成。

## 快速开始

```bash
# 生成 data/ 与 config/ 中的确定性样本（仓库内已生成一份）
python -m mvrv generate --root .

# 加载校验
python -m mvrv validate --root . --verbose

# 全量重建并提交快照
python -m mvrv rebuild --root . --full --workers 2

# 无编辑再次运行：应全部复用，状态哈希不变
python -m mvrv rebuild --root . --workers 4

# 完整验收：样本、增量、随机编辑序列、并行、预算、快照恢复、非法输入
python -m mvrv acceptance --root .

# 单元/集成测试
python -m unittest discover -s tests -v

# Tkinter 界面
python -m mvrv ui --root .
```

一键脚本：`bash scripts/verify.sh`。

## 目录与模块

- `mvrv/synthetic.py`：固定种子 `20261126` 合成 64×48 RGB 视图、4 个相机、33 个候选/接受地标、替换视图和编辑命令流。
- `mvrv/png_codec.py`：零第三方 PNG 编码/解码器，校验签名、IHDR、CRC、8-bit RGB 模式和固定分辨率。
- `mvrv/validation.py`：图像、相机矩阵、视图对应关系、配置和命令流校验，输出稳定字符串/数字业务码与失败清单。
- `mvrv/fixedpoint.py`：Q1024 定点量化、确定性半值向上取整和舍入方向。
- `mvrv/features.py`：分块 luma/梯度/颜色统计、活跃度阈值、特征坐标和内容指纹。
- `mvrv/matching.py`：仅在固定视图对中做互近邻匹配，按视图序号和特征坐标字典序排序并累加内点序号。
- `mvrv/geometry.py`：投影矩阵、整数正规方程/Cramer 求解、定点重投影残差和舍入方向。
- `mvrv/depth.py`：64×48 逐像素深度记录、量化档位、舍入方向、半步冲突决断和边界像素标记。
- `mvrv/voxels.py`：根网格、八叉树子体素、残差均值与深度一致性合成、细分/占位依据和父子引用。
- `mvrv/strategy.py`：按视图数、匹配对数、预计子体素数选择 `coarse/balanced/fine` 策略；块大小、匹配范围和子体素顺序会真实改变计算与账本。
- `mvrv/scheduler.py`：稳定排序任务和线程调度；完成顺序与规范结论分离。
- `mvrv/engine.py`：缓存键、精确失效传播、全局点数预算、单视图配额、增量复用账本和规范状态。
- `mvrv/snapshot.py`：阶段文件先写临时目录，再 `os.replace` 整体提交；清单记录全局版本、配置哈希、相机哈希和阶段版本，并支持校验/回退恢复。
- `mvrv/ui.py`：Tkinter 指标面板、账本树、参数编辑、替换视图、外参编辑、增量重算和恢复判定。
- `tests/test_pipeline.py`：17 个测试，覆盖上述一致性约束。
- `results/`：实际验收 JSON/文本、测试日志、CLI 运行输出和 UI 冒烟记录。

## 确定性与失效模型

定点 scale 为 `1024`。相机 JSON 中的普通数字在加载时通过 `Decimal` 确定性量化为 Q1024；几何求解使用 Python 大整数构造正规方程并用整数行列式求解，避免浮点路径分歧。

主要缓存键：

- 特征：图像 SHA-256 + 策略指纹 + 提取器版本。
- 匹配对：左右视图特征缓存键 + 匹配策略指纹。
- 三维点：匹配指纹 + 两端相机指纹 + 定点求解器指纹；最小视差作为准入门槛单独记录 `gate`，因此提高视差阈值时未重新求解几何的点只记录门槛变化。
- 深度像素：视图/坐标 + 量化参数 + 有序三维点缓存键集合。
- 体素：几何路径 + 体素参数/策略指纹 + 引用三维点键集合 + 引用深度像素键集合 + 父体素路径。

因此：

- 替换一张视图只使该视图特征、涉及它的匹配对、变化/消失的三维点、相关深度像素和引用这些依赖的体素失效。
- 修改一个相机外参只使涉及该视图的三维点及其下游失效；特征和描述子匹配可复用。
- 修改视差门槛只影响准入状态及下游。
- 修改体素尺寸/残差阈值会使体素阶段按键失效；尺寸变化还会产生新路径和删除旧体素。

账本位于每次构建返回值的 `ledger` 字段，包含 `features/pairs/points/depth_pixels/voxels/budget/invalidated` 的重算、复用、门槛变化、删除和待办计数。规范状态 `state` 不包含运行相关的完成顺序、线程数或本次重算计数，因此全量、增量、不同并发度的 `sha256_object(state)` 完全相同。

## 预算与并发

- 候选三维点按 `(视图 i, 视图 j, 左 x, 左 y)` 稳定排序。
- 一个点消耗 1 个全局点数，并分别消耗两个端点视图的配额。
- 调度器按稳定顺序预留预算；首次不能满足的任务标记为 `aborted_budget`，其后任务保留为 `pending_budget` 并写入 `todo_queue`。
- 特征视图任务、匹配对任务、三维点任务和体素任务均带稳定 `sort_key`。
- `run_stable` 可使用多个线程并按完成先后收集结果，但最终一律按稳定键排序归约；测试覆盖 workers `1/2/3/4/8`、0–24 的预算耗尽点、单视图配额和 10 种乱序归约种子。

## 快照与恢复

`SnapshotStore` 把以下阶段产物写入同一个 `staging-*` 临时目录：

- `validation.json`
- `features_matches.json`
- `triangulation.json`
- `depth.json`
- `voxels.json`
- `ledger.json`
- `todos.json`
- `manifest.json`

除清单外每个文件都记录 SHA-256；临时目录完成后通过 `os.replace` 原子发布为 `runtime/snapshots/version-XXXXXX/`，再原子替换 `current.json`。恢复时先验证当前指针；若清单或任一文件哈希损坏，则扫描并恢复到最新的有效版本。

## 业务码

| code | numeric | 含义 |
| --- | ---: | --- |
| `OK` | 0 | 校验通过 |
| `VALIDATION_FAILED` | 4000 | 存在失败项 |
| `E_JSON` | 4001 | JSON 缺失或格式错误 |
| `E_CONFIG_VALUE` / `E_CONFIG_SCHEMA` | 4003 / 4002 | 配置值或结构不合规 |
| `E_PNG_SIGNATURE` | 4101 | PNG 签名错误 |
| `E_PNG_HEADER` / `E_PNG_CONTAINER` | 4102 / 4106 | IHDR/分块容器错误 |
| `E_PNG_DIMENSIONS` | 4103 | 分辨率不是 64×48 |
| `E_PNG_MODE` | 4104 | 不是 8-bit RGB |
| `E_PNG_CRC` | 4105 | PNG 块 CRC 错误 |
| `E_IMAGE_MISSING` / `E_IMAGE_VIEW_MISMATCH` | 4110 / 4111 | 图像缺失或视图不对应 |
| `E_VIEW_INDEX_DUPLICATE` | 4112 | 视图序号重复/非法 |
| `E_CAMERA_SCHEMA` | 4201 | 相机矩阵结构错误 |
| `E_INTRINSIC_SINGULAR` | 4202 | 内参不可逆 |
| `E_EXTRINSIC_ORTHOGONAL` | 4203 | 外参旋转非正交/行列式非正 |
| `E_COMMAND_SCHEMA` / `E_COMMAND_UNSUPPORTED` | 4301 / 4302 | 编辑命令流不合规 |

## 实际验证结果

以下结果来自当前工作区，详细 JSON 见 `results/acceptance_report.json`，测试日志见 `results/unittest_output.txt`。

环境：Python `3.11.0rc1`、NumPy `1.26.4`（满足 Python ≥3.10、NumPy ≥1.24）。

验收摘要：

```text
ok=True
state_hash=78cbced553336381e10d642a98ec180724bf830525b034c3db7b634e956ad64c
strategy=balanced views=4 pairs=6 features=107 matches=140
triangulated=140 depth_pixels=12288 boundary=25 conflict=27
voxels=296 occupied=64 subdivided_parents=19 max_level=2
unchanged_exact=True
scenarios_all_exact=True
stream_all_exact=True
budget_all_exact=True
parallel_all_exact=True
snapshot_recovery=True
invalid_inputs_all_rejected=True
```

精确失效样本：

- 无编辑增量：4 视图、6 对、140 点、12288 深度像素、296 体素全部复用。
- 替换视图 1：仅 1 个特征视图和 3 个关联匹配对重算；3 个消失点进入 removed；4 个深度像素和 2 个根体素重算；增量与空白全量哈希一致。
- 修改视图 2 外参：特征/匹配对零重算；恰有 67 个涉及视图 2 的三维点重算，其余 73 点复用。
- 最小视差 2→5：特征/匹配零重算，24 个点发生门槛状态变化。
- 体素尺寸 4→3、残差阈值 820→650：下游体素键按预期变化，最终状态仍与空白重建逐字段一致。

测试：`17` 个 unittest 用例全部通过，归档日志中的实际结果为 `Ran 17 tests in 60.653s — OK`。

Tkinter：基础 shell 最初缺少 `libtk8.6.so`。标准 WSL 修复为 `sudo apt-get install -y python3-tk`。本次还用 `apt-get download` + `dpkg-deb -x` 将 `libtcl8.6/libtk8.6/libxft2/libxss1` 解到用户目录，并通过 `LD_LIBRARY_PATH/TCL_LIBRARY/TK_LIBRARY` 成功构造 `DiagnosticApp`、触发全量回调；记录见 `results/ui_smoke.txt`。

## 可复核实验入口

```bash
python -m mvrv acceptance --root .
jq '.ok, .full_state_hash, .metrics, [.scenarios[].exact_match]' results/acceptance_report.json
```

验收脚本会把破坏性编辑放在 `runtime/acceptance/` 的临时复制工作区中，不会污染源生 `data/views/*.png` 样本。
