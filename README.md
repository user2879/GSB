# 运动捕捉重定向与姿态约束工作台

纯 Python + NumPy 的确定性多机位动捕重定向工作台，面向 Windows 与本机离线运行。内置固定种子生成器产生 3 台虚拟相机对同一演员骨架的观测，完成多机位融合、源到目标骨架重定向、逐帧约束修正、预算自适应求解策略、跨策略一致性核对和增量编辑失效闭包。

## 环境与安装

- Python 3.10 及以上
- NumPy 1.24 及以上
- 不使用 GPU、Docker、外部动捕资产或网络数据

Windows 命令行：

```bat
py -3.10 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m mocap_workbench --help
```

Linux/macOS 对应命令：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## 一键演示

```bat
.venv\Scripts\python.exe -m mocap_workbench demo --out workdir
```

或：

```bat
run_demo.cmd
```

固定序列为：

1. 按种子 `20261119` 生成 300、3000、30000 帧三档场景；同一场景在进程内再次生成并比较摘要。
2. 保存观测数组、可见性位图、相机参数、骨架、约束和真值来源清单。
3. 执行多机位最小二乘融合、源→目标重定向、预算策略选择与约束求解。
4. 针对预算允许的其余策略重复求解，并逐帧比较姿态与约束残差。
5. 在 300 帧场景上依次修改单帧约束、相机观测权重、目标骨长，记录精确失效闭包和复用版本戳。
6. 主动触发并核对五个业务错误码。
7. 将结构化的 `demo_report.json` 同时写入目录并输出到标准输出。

演示产物每套文件合计约 78.4 MB，主要来自 30000 帧观测、真值和逐帧残差数组。完整运行约 35–36 秒，峰值内存约 458 MB。

## 输出目录

```text
workdir/
  demo_report.json
  scenes/300/
    scene.json
    visibility_manifest.json
    arrays/observations.npy
    arrays/visibility.npy
    arrays/sigma.npy
    arrays/truth.npy
  solutions/300/
    solution.json
    frame_hashes.json
    consistency.json
    arrays/fused_source_pose.npy
    arrays/fusion_residual.npy
    arrays/retarget_pose.npy
    arrays/retarget_joint_rotation_vectors.npy
    arrays/retarget_roundtrip_residual.npy
    arrays/strategy_pose.npy
    arrays/corrected_target_pose.npy
    arrays/constraint_residuals.npy
  incremental/
    incremental_ledger.json
    arrays/*_version.npy
    arrays/incremental_corrected_pose.npy
    arrays/incremental_constraint_residuals.npy
```

`visibility_manifest.json` 逐帧逐关节列出可见相机位图；位 `i` 表示相机 `i` 可见。不可见相机的观测统一为 `NaN`，融合时不会读取。

## CLI

所有命令向标准输出写单行规范化 JSON，并用退出码表示业务结果。

```bat
python -m mocap_workbench generate --frames 300 --seed 20261119 --out workdir/scene300
python -m mocap_workbench solve --scene workdir/scene300 --out workdir/solve300 --strategy auto
python -m mocap_workbench consistency --scene workdir/scene300 --out workdir/consistency300.json
python -m mocap_workbench errors --out workdir/errors.json
```

可通过 `--memory-mb`、`--time-ms`、`--window-size`、`--window-overlap`、`--anchor-step` 修改预算和策略参数。

## 数据模型

- 17 个关节的人体骨架，根节点为 `hips`，其余节点按父子拓扑排列。
- 源骨架为 T-pose 风格；目标骨架更高、手臂更短且静止姿态下垂，腿部更长。
- 每帧每个关节每台相机包含三维观测点、噪声尺度 `sigma`；可见性由相机视锥和确定性遮挡分数共同决定。
- 观测噪声在相机局部坐标中按 `sigma` 生成并旋转回世界坐标。
- 参考真值只来自固定种子生成器，路径和来源写入场景清单，不参与融合。
- 固定约束包括肘/膝角度限位、左右脚周期接触点、根高度区间、左右角度和位置对称容差。

## 核心算法

### 多机位融合

对可见相机使用噪声倒数平方权重：

```text
x_hat = sum_i(w_i * x_i) / sum_i(w_i),  w_i = 1 / sigma_i^2
```

残差为加权 RMS 距离。隐藏相机的观测值即使被恶意赋值也不会进入权重。无直接观测的关节在骨架层级内按父子关系与声明骨长补全；若关节在孤立帧对全部相机不可见，则用相邻可见帧插值出方向时间先验，再以当前父节点为锚恢复骨长，避免静止方向造成的单帧跳变。全部隐藏值均不参与补全。

### 可逆重定向

每根骨定义静止姿态对齐旋转 `C_i`，将源骨架静止骨方向对齐到目标骨架静止骨方向。当前帧目标骨方向为：

```text
d_target_i = C_i * d_source_i
joint_target_i = joint_target_parent + target_bone_length_i * d_target_i
```

该映射同时使用骨长比例和逐骨静止姿态差，根位移按骨架高度比例缩放。逆映射使用 `C_i^T` 和源骨长，逐帧复现融合后的源姿态。输出包含目标姿态、每关节往返残差、目标骨长误差、逐关节 Rodrigues 旋转向量和全局偏差报告。

方向空间映射不臆造仅凭三维位置无法观测的骨轴自旋；位置与骨长往返是精确的。

### 约束求解

逐帧使用确定性近端投影迭代，硬约束不放松：

- 关节角限位：绕关节法轴旋转子链，改变夹角且保持骨长。
- 双脚接触：解析双骨 IK，将接触踝精确放到固定点；不可达时返回 `E_CONSTRAINT_UNSAT`。
- 根高度：整体平移骨架后继续投影脚接触。
- 左右对称：角度对按容差收紧；位置对优先围绕共同父关节做等半径对称投影，以避免破坏骨长。
- 每轮末尾恢复目标骨长与脚接触，最终逐约束检查残差。

输出 `constraint_residuals.npy`（帧 × 约束）和约束名称清单；角度类残差单位为度，其余为米。不可行时错误详情包含首个不可行帧、冲突约束组合和不可达接触帧。

## 策略与预算

策略针对根轨迹做时间联合平滑，然后进入逐帧约束求解：

- `full_segment`：构造全段稠密法方程，只在 300 帧档使用。
- `windowed`：192 帧窗口、48 帧重叠，重叠权重累加，窗口局部三对角求解。
- `sparse_anchor`：每 2 帧设锚点，锚点轨迹三对角求解后线性插值。

默认预算为 `512 MB / 12000 ms`，估算的是策略求解工作集和单次求解时间。选择结果和完整估算账本写入 `solution.json`。30000 帧的 `full_segment` 会明确返回 `E_STRATEGY_BUDGET`，不会被静默执行。

一致性核对默认使用 5 cm 关节位置容差和 0.02 约束残差容差。报告包含最大偏差、首个分歧帧、参与关节、约束残差差值和最近的窗口边界/锚点区间。

## 增量失效闭包

`IncrementalWorkbench` 使用重叠窗口策略并缓存四个阶段：

```text
fusion -> retarget -> strategy -> constraints
```

- 单帧约束修改：只重算该帧 `constraints`。
- 单相机权重修改：重算该帧 `fusion` 与 `retarget`，再重算包含该帧的窗口及其并集帧的 `strategy` 与 `constraints`。
- 目标骨长修改：复用全部 `fusion`，重算全部 `retarget/strategy/constraints`。

闭包集合必须与实际重算集合完全相等；不等时返回 `E_CLOSURE_MISMATCH`。每个阶段每帧保存版本数组，账本逐项记录 `computed` 或 `reused`、来源版本和当前账本版本。

## 业务错误码

| 业务码 | 退出码 | 触发条件 |
|---|---:|---|
| `E_OBS_MISSING` | 10 | 没有可见观测，或层级补全无法到达全部关节 |
| `E_RETARGET_TOL` | 11 | 严格模式下重定向往返超过声明容差 |
| `E_CONSTRAINT_UNSAT` | 12 | 约束不可行、接触不可达或骨长无法保持 |
| `E_STRATEGY_BUDGET` | 13 | 请求/自动策略均超出内存或时间预算 |
| `E_CLOSURE_MISMATCH` | 14 | 声明失效闭包与实际重算集合不一致 |
| `E_INPUT` | 20 | 参数、形状或场景格式错误 |
| `E_IO` | 21 | 文件读写错误 |

`python -m mocap_workbench errors` 会串联触发并报告五个核心业务码。

## 测试与复算

```bash
.venv/bin/python -m unittest discover -s tests -v
```

测试覆盖确定性生成与落盘、可见性位图、三种相机子集、隐藏观测隔离、层级补全与时间连续补全、可逆重定向、约束可行/不可行、预算选择、策略分歧定位、增量闭包和全部业务错误码。

完整双跑验证记录见 `VERIFICATION.md`。
