# 增量可微软光栅化与逆向成像组件

这是一个不依赖外部素材或网络服务的确定性可微渲染组件。它包含纯 Python 核心、Windows 可运行示例、15 项自动化测试，以及两次运行逐字节一致的验收输出。

核心运行时只使用 Python 标准库；NumPy 与 PyTorch CPU 作为可选互操作环境单独验证，不参与主渲染路径，因此离线环境也能完整执行渲染、梯度、依赖图、持久化和恢复演示。

## 功能概览

- 48x48 图像，固定 8x8 瓦片，3 个球体、1 个方向光、1 个透视相机。
- 球体按屏幕 impostor 投影并做软覆盖光栅化，输出可见球体 ID、三角形重心坐标、深度、覆盖率与法线。
- 着色拆成环境通道和 Lambert 漫反射通道，粗糙度用简化漫反射增益 `1 / (1 + roughness)` 参与模型。
- 29 个扁平参数：每个球体的位置/半径/反照率/粗糙度、光源方向/环境项、相机位置/欧拉角/焦距。
- 对球体位置、半径、反照率、粗糙度、光源方向、环境项、相机外参和焦距提供解析梯度。
- 参数图到瓦片、着色通道、梯度分块、最终图像和损失的版本化依赖图。
- 参数内容哈希 + 数值容差双重有效性判定；微扰不触发失效。
- 按旧/新覆盖瓦片集合计算精确闭包，闭包外节点复用并记录版本戳与复用来源摘要。
- 原子代际快照、版本指针、内容哈希、受控关闭标记、强制终止孤儿回收与重启续跑。
- 六个稳定业务错误码及其异常链路测试。

## Windows 快速开始

在 Windows 10/11、Python 3.10+ 环境中执行：

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# 可选：从本地已配置的包源安装 CPU 版依赖。
python -m pip install numpy
python -m pip install torch

python -m unittest discover -s tests -v
python -m diffrender.demo --workspace artifacts\demo_workspace --out artifacts\demo
python scripts\verify_demo.py
```

核心演示不需要 NumPy/PyTorch。`scripts\env_check.py` 可检查可选环境，例如本机实际验证结果为：

```json
{"cuda_available":false,"numpy":"1.26.4","python":"3.10.12","torch":"2.14.1+cpu","torch_threads":8}
```

## 代码结构

| 文件 | 职责 |
| --- | --- |
| `diffrender/scene.py` | 场景、扁平参数键、参数校验、确定性场景与逆向成像真值场景 |
| `diffrender/math3d.py` | 相机基向量及其对 yaw/pitch/roll 的解析导数 |
| `diffrender/renderer.py` | 投影、软光栅化、着色、解析梯度、全量参考路径、有限差分与双路径比对 |
| `diffrender/graph.py` | 节点、边、触发参数键、前向失效闭包、版本戳与图序列化 |
| `diffrender/pipeline.py` | 增量调度、瓦片/参数分块缓存、复用账本、状态编解码 |
| `diffrender/persistence.py` | 代际快照、原子写入、版本指针、关闭标记、孤儿回收 |
| `diffrender/demo.py` | 固定编辑序列、单参数优化、硬终止子进程与恢复验证 |
| `tests/` | 确定性、梯度、闭包、复用、持久化、恢复和错误码测试 |
| `scripts/verify_demo.py` | 两次独立运行并逐文件比较字节 |

## 渲染与梯度模型

### 软光栅化

相机基为 `right`, `up`, `forward`，相机坐标 `q = (dot(c-eye, right), dot(c-eye, up), dot(c-eye, forward))`。球心投影与屏幕半径：

```text
cx = W/2 + focal * qx / qz
cy = H/2 - focal * qy / qz
radius_px = focal * radius / qz
```

像素到投影中心的距离为 `d`，软覆盖率为：

```text
t = radius_px + 0.5 - d
coverage = clamp(t, 0, 1)
```

每个球体使用屏幕空间 impostor 法线 `N = right*u + up*v - forward*w`，其中 `u=(x-cx)/r`、`v=-(y-cy)/r`、`w=sqrt(1-u²-v²)`。每个像素选择最近正覆盖球体，记录 ID、`(b0,b1,b2)` 重心坐标、前表面深度和覆盖率。每个瓦片还记录覆盖球体集合，作为依赖图几何上游。

### 着色

```text
L = normalize(light.direction)
lambert = max(0, dot(N, L))
K = 1 / (1 + roughness)
ambient_channel = albedo * ambient * coverage
diffuse_channel = albedo * (1 - ambient) * lambert * K * coverage
intensity = ambient_channel + diffuse_channel + background * (1 - coverage)
```

最终损失为 `mean((image - target)^2)`。解析梯度按瓦片和参数分块缓存，每个分块只在自己关联的瓦片失效时重算；不存在“先重算整图再裁剪”的路径。

### 解析梯度

- 球心分量：通过相机基投影到 `qx/qy/qz`，再链式传播到屏幕中心、屏幕半径、局部坐标、覆盖率、法线与 Lambert 项。
- 半径：影响屏幕半径、局部坐标、覆盖率、法线和深度，不影响球心投影。
- 反照率/粗糙度：直接作用于环境或漫反射通道。
- 光源分量：使用归一化雅可比 `dL/dl_j = (e_j - L*L_j) / ||l||`。
- 相机位置：`dq/deye = -right/up/forward`。
- 相机 yaw/pitch/roll：同时传播 `dq/dθ`、`d(right/up/forward)/dθ` 和法线局部坐标导数。
- 焦距：同时传播屏幕中心、屏幕半径和法线局部坐标。

有限差分使用中心差分和步长 `1e-8`。实际全参数最大误差为 `7.144230168565358e-08`，声明容差为 `2e-4`。

## 依赖图与失效闭包

节点类型：

- `param:<key>`：单个标量参数。
- `raster:tile:r:c`：瓦片 ID、重心、深度、覆盖、法线、覆盖球体集合。
- `shade:tile:r:c`：环境通道、漫反射通道、强度。
- `gradchunk:tile:r:c:<key>`：某瓦片对某参数的损失像素梯度分块。
- `grad:<key>`：某参数的全部瓦片梯度聚合。
- `image:final`、`loss:l2`：最终图像与损失。

参数边携带触发键；一旦节点失效，沿所有下游结构边传播。相机编辑影响全部瓦片。球体几何编辑取旧覆盖瓦片与新覆盖瓦片并集，解决球体移入此前未关联瓦片的问题。材质/光照编辑只标记真正可见的着色瓦片。更新前后调用 `_expected_closure` 自检，不一致返回 `E_CLOSURE_MISMATCH`。

每次更新的账本记录：

- 参数旧/新精确十六进制值、内容哈希、差值、容差、是否真正生效。
- 光栅脏瓦片、着色脏瓦片、完整失效闭包。
- 重算节点及其版本戳。
- 闭包外全部复用节点及其版本戳，以及复用清单哈希。

默认容差：

| 参数类别 | 容差 |
| --- | ---: |
| 球体位置/半径 | `1e-9` |
| 球体材质 | `1e-9` |
| 光源 | `1e-9` |
| 相机 | `1e-9` |

只有内容哈希变化且数值差绝对值大于容差才生效。小于等于容差的编辑记入账本但闭包为空，所有节点版本和哈希保持不变；实现还会保留原有效值，避免微扰累积。

## 持久化与恢复

工作区布局：

```text
<workspace>/state/committed/snapshot-000001.json
<workspace>/state/committed/CURRENT.json
<workspace>/state/committed/CLOSED.json       # 仅受控关闭
<workspace>/state/staging/                    # 未提交缓存
<workspace>/state/orphans/                    # 恢复时回收
```

提交顺序为：写新代际快照，`fsync`，原子替换 `CURRENT.json`。`CURRENT.json` 同时保存版本号、代际、快照文件名和 SHA-256；读取时逐项校验，任何版本或哈希不一致返回 `E_SNAPSHOT_VERSION`，不会回退使用旧代际缓存。

快照包含参数内容哈希、参数版本、节点内容哈希、节点版本戳、依赖边、全部可复用瓦片与梯度分块、账本和关闭状态。重启时从快照重编码所有瓦片/梯度，重建图并逐节点比对提交图；不一致返回 `E_DEPENDENCY_STALE`。

受控关闭写入 `closed=true`、`clean_shutdown=true`、完整节点哈希退出报告和 `CLOSED.json`。强制终止子进程在写入 staging 缓存后直接 `os._exit(97)`，不执行关闭流程。重启只读取最近 `CURRENT.json` 指向的已提交快照，将 staging 文件移动到 `orphans/` 并写孤儿账本与恢复报告；`fabricated_exit_report=false`，不会伪造退出报告。

## 业务错误码

| 错误码 | 触发场景 |
| --- | --- |
| `E_SCENE_INVALID` | 非正半径、非法反照率/环境项、零光源、相机后方的球体等 |
| `E_GRAD_MISMATCH` | 解析梯度/有限差分不一致，或增量与全量梯度不一致；报告分块和参数键 |
| `E_DEPENDENCY_STALE` | 依赖边引用缺失节点、恢复节点/版本戳/缓存与快照不一致 |
| `E_CLOSURE_MISMATCH` | 实际失效闭包与精确旧/新瓦片闭包不一致 |
| `E_SNAPSHOT_VERSION` | 指针版本、快照版本、SHA-256、关闭标记或快照结构不一致 |
| `E_TILE_HASH` | 同一场景重复渲染哈希不同，或增量/全量图像瓦片哈希不一致；报告首个分歧瓦片 |

## 验收结果

当前实际运行结果：

- `python -m unittest discover -s tests -v`：15 项通过。
- 固定编辑与优化账本：16 条；其中 2 条容差内微扰闭包为 0。
- 逆向优化损失：`0.001844470601953254` 降至 `0.0002759652543264671`。
- 每一步增量图像/梯度与全量参考路径哈希一致，最大逐像素梯度差为 `0.0`。
- 全参数有限差分最大误差：`7.144230168565358e-08`（声明容差 `2e-4`）。
- 硬终止返回码：`97`；检测到强制终止并回收 1 个孤儿缓存。
- 恢复后 1176 个依赖图节点哈希与版本戳逐项等于提交状态。
- 受控关闭写出 `CLOSED.json` 与退出报告。

关键哈希：

```text
base parameter hash      9f060df8f36e2c48eb6dc3549b89ebeaa2a5b866ec97b17d543dcb2ac25f5e9c
base raster hash         a5e98276d3a84bb5f76adb830291695446a7576bb8f5f709b6b480608315c348
base shade hash          16745a7d518143b3e96828545b1edca5c2c340147207b0903a00fa1c31a5b528
base render hash         f2bf07370b2d89b1038d849222b7aecb1be2e856d78a778e3df860922d50c2ca
final image hash         de73169aec7c8e088f4d06e68bf8dd9bb4bc09c1f69ea72cc05927d5aebd3fdf
final gradient hash      95ad22e053ee77c060127311c0c3d039b053cd1147d8009af13f2f1790bb0fe9
final node manifest      a5599ee8209b3ae2772beb1b786d41cfbae8e6b9895b6983f3531cec60cb3ee1
```

输出文件：

- `artifacts/demo/index.json`：输出文件哈希索引。
- `artifacts/demo/summary.json`：编辑、优化、闭包规模与双路径比较摘要。
- `artifacts/demo/hash_manifest.json`：所有节点哈希、版本戳和关键渲染哈希。
- `artifacts/demo/invalidation_ledger.json`：完整失效/复用账本。
- `artifacts/demo/recovery_report.json`：强制终止、孤儿回收与恢复一致性。
- `artifacts/verification.json`：两次独立运行的逐文件字节比较结果。

`scripts/verify_demo.py` 在两次全新工作区分别运行完整演示，并比较 `index.json`、`summary.json`、`hash_manifest.json`、`invalidation_ledger.json`、`recovery_report.json` 的原始字节。当前两次运行五份文件全部一致。

