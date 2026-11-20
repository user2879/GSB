# PointCloud Studio

面向 Windows 的确定性点云编辑、局部重建、误差账本与离屏快照应用。项目只依赖 Python 3.11、NumPy 与 PyOpenGL，不依赖容器、数据库或远程服务。

## 快速开始（Windows PowerShell）

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 生成 120,000 点固定种子样本
python -m pointcloud_studio generate --out artifacts\sample

# 完整验收；同时写入 validation_report.json
python -m pointcloud_studio validate `
  --sample artifacts\sample `
  --workspace artifacts\run_workspace `
  --report artifacts\validation_report.json
```

也可直接运行 `scripts\setup_windows.ps1` 与 `scripts\run_validation.ps1`。

当前交付环境是 WSL/Linux，已使用同样的 Python 3.11 代码完成 CPU 全链路验证。离屏接口在该环境返回 `E_GL_CONTEXT`，同时点集快照、网格、账本和恢复路径保持可用；Windows 下使用隐藏 Win32 窗口 + WGL + 不可见 FBO。

## 命令

```text
python -m pointcloud_studio generate --out DIR
python -m pointcloud_studio validate --sample DIR --workspace DIR --report FILE
python -m pointcloud_studio verify --workspace DIR
python -m pointcloud_studio verify-ledger --points FILE --normals FILE --ledger FILE
python -m pointcloud_studio render --workspace DIR --width 960 --height 640
python -m pointcloud_studio inject-faults --sample DIR --workspace DIR --out DIR
```

`verify-ledger` 是第三方口径：只读取当前提交目录的 `points.bin`、`normals.bin` 和 `ledger.json`，独立重算每个曲面片、内容哈希及全部误差上界。

## 样本与格式

固定配置 `SampleConfig(seed=20261120)` 生成恰好 120,000 点：

- `cube_room`：72,000 点，六个房间面；
- `ground`：48,000 点，独立地面；
- 每点含单位法线和 `[0,1]` 法线置信度；
- 输出 `points.bin`、`normals.bin`、`camera.json`、`edits.json`、`manifest.json`；
- 两次生成的所有文件逐字节一致。

二进制头为 `<8sIIQ`：magic、版本、flags、点数。点记录为 little-endian `float32 (N,3)`；法线记录为 little-endian `float32 (N,4)`，依次是 `nx,ny,nz,confidence`。

## 编辑与增量重建

内置操作序列覆盖球域点剔除、框选剔除、两次法线重估半径变更、一次重建参数调整。局部操作只做以下工作：

1. 用体素栅格选择受影响点；
2. 把点所在曲面片、支撑半径命中的曲面片及其共享边界片加入失效闭包；
3. 只从完整点集重算闭包内曲面片；
4. 复用其他曲面片的同一不可变缓存对象；
5. 用编辑后完整点集另做一次显式全量重建，仅用于一致性对照，不参与生产结果。

一次 120k 样本的局部删除实测只重算 34 个曲面片，复用 965 个；第二次框选删除重算 80 个，复用 921 个。闭包清单与实际重算清单完全相等，复用片顶点逐点不变，增量/全量最大顶点偏差为 `0.0`。

### 实际完整验收结果

2026-11-20 在 WSL2/Linux、Python 3.11.0、NumPy 2.4.6、PyOpenGL 3.1.10 上执行完整五操作验收：

| 操作 | 选中点 | 生产重建模式 | 重算片 | 复用片 | 最大顶点偏差 |
|---|---:|---|---:|---:|---:|
| 球域点剔除 | 205 | 局部增量 | 34 | 965 | 0.0 |
| 框选剔除 | 1,173 | 局部增量 | 80 | 921 | 0.0 |
| 法线重估半径 0.34 | 586 | 局部增量 | 43 | 954 | 0.0 |
| 法线重估半径 0.42 | 542 | 局部增量 | 73 | 923 | 0.0 |
| 三角化参数调整 | 0 | 显式全局参数替换 | 1,026 | 0 | 0.0 |

最终状态为版本 6，995 个曲面片、16,891 个顶点、18,884 个三角形。独立账本复算 `maximum_numeric_difference=0.0`，网络位置误差上界 `0.17053386145341778`；31/31 检查项通过。强杀子进程退出码为 23，重启报告 `clean_previous_exit=false`、恢复版本 6、回收 1 个未提交 staging，随后独立复算仍通过。六类业务错误码全部命中预期。完整记录见 `artifacts/validation_report.json`。

本机是 WSL/Linux，因此真实离屏调用按设计返回 `E_GL_CONTEXT`；这同时验证了 CPU 网格/账本校验路径不依赖 GL。Windows 隐藏窗口 WGL/FBO 路径已完成实现，但需在目标 Windows 图形驱动上执行 `python -m pointcloud_studio render ...` 做现场驱动确认。

## 误差账本

每个曲面片的统一几何位置误差上界为：

```text
quantization_bound
+ 2 * neighbor_radius * sin(normal_angle_bound)
+ chord_height
```

其中：

- 量化项由点文件中的 float32 半 ULP 计算；
- 法线角界由局部平面拟合残差和量化项合成；
- 弦高由源点到相邻三角面局部平面的最大垂直偏差计算；
- 各项均以世界空间欧氏距离为单位，通过三角不等式相加；
- 网络误差上界取所有曲面片上界的最大值。

账本逐片记录输入内容哈希、源点哈希、各阶段误差、总误差和缓存文件；账本自身也有 SHA-256。实现见 `pointcloud_studio/ledger.py`。

## 快照、持久化与异常

- 点集与网格组成一个不可变 `SceneSnapshot`，通过单指针原子替换；版本过期的读取返回 `E_SNAPSHOT_STALE`。
- 重建服务在工作线程中执行；视口只读取已发布快照中的点集与网格。
- 每个提交状态先写入 staging，逐个文件临时写入并 `os.replace`，再原子目录改名，最后原子更新 `CURRENT`。
- 正常退出写 `session_manifest.json` 与 `closed.json`；强杀后不产生伪造的正常退出报告。
- 未提交 staging、无效状态和先于 `CURRENT` 改名的新状态会移入 `orphans/`，并记录在 `recovery/open-*.json`。

业务错误码：`E_POINT_CLOUD_FORMAT`、`E_NEIGHBOR_RADIUS`、`E_SNAPSHOT_STALE`、`E_MESH_MISMATCH`、`E_CACHE_HASH`、`E_GL_CONTEXT`。故障矩阵通过损坏输入副本或可序列化测试参数触发，配置随 `fault_matrix_report.json` 和验收报告保存。

## 代码结构

```text
pointcloud_studio/
  sample.py       固定种子样本、相机轨迹、编辑序列、manifest
  formats.py      二进制点/法线格式
  spatial.py      体素栅格邻域查询
  normals.py      区域法线重估与逐点残差/误差上界
  mesh.py         局部曲面片、三角化、缓存与内容哈希
  ledger.py       误差传播与独立复算
  consistency.py  增量/全量双口径核对
  engine.py       编辑事务、失效闭包、缓存复用
  snapshots.py    不可变联合快照与原子版本替换
  runtime.py      后台重建线程
  render.py       Windows WGL/FBO 与 E_GL_CONTEXT 降级
  viewport.py     视口屏幕框选到点集/编辑操作
  camera.py       相机姿态与投影矩阵
  persistence.py  原子提交、关闭清单、强杀恢复
  injections.py   六类业务异常的可复现注入矩阵
  validation.py   端到端验收与运行报告
```

详细设计、公式和恢复时序见 `docs/TECHNICAL.md`，实际验证结果见 `artifacts/validation_report.json`。
