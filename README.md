# 房间空间声学增量诊断工作台

面向 Windows 本地环境的确定性房间冲激响应（RIR）与球谐域声场重建工作台。输入由工作区内程序化生成：48 kHz、16-bit PCM、4 通道 RIR，4 个方向通道按声源与方向支撑范围关联。派生链为：

```text
参数版本戳 -> RIR 通道 -> 球谐系数集 -> 能量网格分块 -> 等距柱状投影瓦片
```

任意参数变更后，引擎只重算依赖图中受影响的节点；闭包外的 RIR、球谐系数、能量块和瓦片从内容寻址存储复用。最终结果与同场景全量重建逐节点哈希一致，RIR PCM、球谐系数、能量值和瓦片字节的实测最大差异均为 `0`。

## Windows 安装

要求 Python 3.10+。在仓库根目录打开 PowerShell：

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

也可以直接运行 `scripts\setup_windows.ps1`。依赖为 NumPy、SciPy、PyOpenGL、GLFW、Pillow；诊断页使用标准库 HTTP 服务，PyOpenGL/GLFW 仅用于可选的离屏 FBO 回读探测，不要求 GPU 计算着色器。

确认 48 kHz、16-bit、4 通道 WAV 可写并逐样本回读：

```powershell
.\.venv\Scripts\python.exe -m acoustic_workbench probe-wav --workspace artifacts\workbench
```

## 快速使用

```powershell
# 初始化并构建默认场景
.\.venv\Scripts\python.exe -m acoustic_workbench init --workspace artifacts\workbench

# 提交参数补丁；expected-revision 是乐观并发令牌
.\.venv\Scripts\python.exe -m acoustic_workbench submit "wall.xmax.alpha=0.37" --workspace artifacts\workbench --expected-revision 1
.\.venv\Scripts\python.exe -m acoustic_workbench submit "source.s0.position=1.15,0.95,1.28" --workspace artifacts\workbench

# 输出逐节点哈希清单
.\.venv\Scripts\python.exe -m acoustic_workbench manifest --workspace artifacts\workbench --output artifacts\workbench\manifest.json

# 重启校验、全量对照与可选 OpenGL 离屏探测
.\.venv\Scripts\python.exe -m acoustic_workbench verify --workspace artifacts\workbench --full-compare --gl-probe

# 只读交互诊断页
.\.venv\Scripts\python.exe -m acoustic_workbench diagnose --workspace artifacts\demo\run_a --port 8765 --open-browser
```

`scripts\run_demo.bat`、`scripts\run_diagnose.bat`、`scripts\run_tests.bat` 提供对应 Windows 双击入口。

## 固定演示

```powershell
.\.venv\Scripts\python.exe -m acoustic_workbench demo --output-dir artifacts\demo
```

固定变更序列：

1. 移动 `s0`。
2. 修改 `xmax` 墙面吸声系数。
3. 同时修改接收球面采样密度与重建阶数。
4. 修改 `ymin` 墙面吸声系数。
5. 移动 `s1`。

脚本执行两次独立序列并输出每一步的失效闭包大小、复用节点数、重算耗时与逐节点哈希。随后把最终增量工作区与从零全量重建比较。两次序列清单必须逐字节相同。

本机实测（Python 3.11，Linux/WSL 验证环境；Windows 使用同一纯 Python/NumPy 路径）：

```text
step 1  rev 2  closure=44  reused=28  tiles_reused=12  tiles_recomputed=20
step 2  rev 3  closure=42  reused=30  tiles_reused=12  tiles_recomputed=20
step 3  rev 4  closure=68  reused=4   tiles_reused=0   tiles_recomputed=32
step 4  rev 5  closure=38  reused=34  tiles_reused=14  tiles_recomputed=18
step 5  rev 6  closure=40  reused=32  tiles_reused=14  tiles_recomputed=18
sequential manifests identical: True
incremental equals full rebuild: True
tile hash mismatches: 0; node hash mismatches: 0
numeric max abs differences: sph_float64=0.0 energy_float64=0.0 tile_bytes=0.0
acceptance: PASS
```

输出位置：

- `artifacts/demo/manifest_run_a.json`、`manifest_run_b.json`：两次序列清单。
- `artifacts/demo/manifest_incremental_final.json`、`manifest_full.json`：最终增量与全量清单。
- `artifacts/demo/verification.json`：字节一致性、哈希差异、数值容差与耗时。
- `artifacts/demo/run_a/outputs/room_rir_48k_16bit_4ch.wav`：可直接播放/分析的 4 通道 RIR。
- `artifacts/demo/run_a/outputs/rir_metadata.json`：各通道峰值、到达时间与反射墙面。
- `artifacts/demo/run_a/outputs/equirectangular_projection.png`：等距柱状能量投影。

## 依赖图与失效规则

节点共 72 个：

- `rir:c0..c3`：4 个确定性 RIR 通道。
- `sh:c0..c3`：每个通道的实球谐系数集。
- `energy:r{0..3}c{0..7}`：固定 8×4 能量网格分块。
- `tile:r{0..3}c{0..7}`：固定边界投影瓦片。

边带 `trigger_keys`：

- 参数节点 `param:<key>` 指向其直接影响节点。
- RIR→SH 边带声源位置、该通道路径实际经过的墙面吸声键和噪声种子键。
- SH→能量块边带采样密度、阶数及所关联通道的上游参数键。
- 能量块→瓦片边带上游参数键。

失效闭包算法：

1. 每个节点保存计算时的局部参数版本戳 `param_stamps`。
2. 当前版本与戳不一致的节点成为种子；另用持久化旧场景值做版本戳缺失时的防御性比对。
3. 沿 `kind=artifact` 的依赖边做确定性拓扑传播。
4. 闭包外节点逐块校验 blob 哈希、节点内容哈希和上游输入哈希后复用。
5. 只有受影响能量块及瓦片重写；新 blob 不可变，旧状态直到 `state.json` 原子替换前一直有效。

墙面吸声只进入该墙面实际反射路径的通道路径集合。例如默认场景 `xmax` 只影响 `rir:c0`，因此闭包为 1 个 RIR、1 个 SH、20 个受影响能量块与对应 20 个瓦片，其余 30 个节点复用。

## 球谐与投影

- 采样：确定性 Fibonacci 全球覆盖，默认 128 点，可调 32–512。
- 拟合：用每条路径的量化峰值/能量与延迟衰减，在采样方向构造能量观测，再用实球谐基做最小二乘。
- 阶数上界：`order <= min(8, floor(sqrt(sample_density)) - 1)`；快照与 SH 节点均写入覆盖声明、截断阶数和上界。
- 求值：球谐场在非负截断后乘以固定方向支撑掩码；支撑范围外的通道贡献严格为零，因此支撑外瓦片可精确复用。
- 投影：全球等距柱状 `256×128` 默认网格，固定 `4×8` 块；块边界写入节点载荷与哈希。色调映射使用绝对 dB 标尺，避免全局最大值变化污染未受影响瓦片。

## 持久化与原子性

所有节点产物存放在内容寻址 `blobs/<sha256>.bin`，写入使用临时文件、`fsync` 与 `os.replace`。`state.json` 包含依赖图、节点内容哈希、产物哈希、输入哈希、参数版本戳与最近构建报告，并整体原子替换。`scene.json` 与状态分离，崩溃后重启会比较场景版本和旧状态，重新计算失效闭包。

跨版本复用前逐节点校验：

- 非瓦片产物或依赖输入哈希不符：`E_DEPENDENCY_STALE`。
- 瓦片产物或内容哈希不符：`E_TILE_HASH`。

诊断快照同样使用唯一文件名、原子写入并设为只读。交互视图只读取该快照；HTTP 服务仅允许 GET/HEAD，任何提交方法返回 405。

## 业务错误码

| 错误码 | 触发示例 | 行为 |
|---|---|---|
| `E_DEPENDENCY_STALE` | 球谐/能量/输入哈希跨版本不符 | 拒绝复用，抛出错误；原工作区不受影响 |
| `E_SPH_ORDER` | 阶数 99，或密度 32 时阶数 5 | 校验在状态写入前失败 |
| `E_GRID_BOUNDS` | 宽度 7、非整除分块、非全球边界 | 校验失败 |
| `E_TILE_HASH` | 已持久化瓦片字节被篡改 | 打开副本时报错，原始工作区仍可继续 |
| `E_PARAM_RANGE` | 吸声系数 1.5、非法声源位置 | 状态不变 |
| `E_SNAPSHOT_CONFLICT` | 两个提交使用相同旧 revision 令牌 | 文件锁串行化；后到者冲突，不覆盖先到者 |

实际注入：

```powershell
.\.venv\Scripts\python.exe -m acoustic_workbench init --workspace artifacts\workbench
.\.venv\Scripts\python.exe -m acoustic_workbench inject-error E_SPH_ORDER --workspace artifacts\workbench
.\.venv\Scripts\python.exe -m acoustic_workbench inject-error E_GRID_BOUNDS --workspace artifacts\workbench
.\.venv\Scripts\python.exe -m acoustic_workbench inject-error E_PARAM_RANGE --workspace artifacts\workbench
.\.venv\Scripts\python.exe -m acoustic_workbench inject-error E_SNAPSHOT_CONFLICT --workspace artifacts\workbench
.\.venv\Scripts\python.exe -m acoustic_workbench inject-error E_TILE_HASH --workspace artifacts\workbench
.\.venv\Scripts\python.exe -m acoustic_workbench inject-error E_DEPENDENCY_STALE --workspace artifacts\workbench
```

`E_TILE_HASH` 与 `E_DEPENDENCY_STALE` 注入在隔离副本中进行，原始工作区在注入后仍可重新打开并继续提交。

## 测试

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

测试覆盖：WAV 往返；4 通道合成；依赖图分层；墙面与声源定向失效；阶数/密度只失效下游；重启版本戳重建；瓦片复用哈希；全量重建等价；并发冲突；六类错误码；只读快照与 HTTP；固定演示两次清单字节一致。当前验证结果：`18 passed`。

## 目录

```text
acoustic_workbench/   RIR、球谐、投影、依赖图、持久化、服务、CLI
tests/                单元与端到端验收测试
scripts/              Windows 安装/演示/诊断/测试入口
docs/TECHNICAL.md     数据结构、算法、哈希与原子事务说明
artifacts/demo/       固定演示工作区、清单与验证结果
```
