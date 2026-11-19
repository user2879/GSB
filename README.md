# 光谱材质与颜色管理工作区

这是一个仅依赖 Python 3.11+ 与 NumPy 的本地确定性应用，覆盖光谱材质数据校验、CIE 观察者积分、色适应、目标色彩空间输出、版本化操作、阶段级增量重算、原子提交、强制终止恢复和可序列化异常注入。

项目按 Windows 本地应用设计：路径、原子替换、进程终止和批处理入口均兼容 Windows；本次实际验证在任务提供的 Linux/Python 3.11 环境中执行。

## 快速运行

Windows PowerShell / CMD：

```bat
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python app.py generate --out samples\sample_scene.json --operations samples\demo_operations.json
.venv\Scripts\python app.py demo --out runtime\demo --report reports\demo_report.json --force
.venv\Scripts\python -m unittest discover -s tests -v
```

也可直接运行：

```bat
run_demo.bat
run_tests.bat
```

Linux/macOS 等价命令：

```bash
python3 app.py demo --out runtime/demo --report reports/demo_report.json --force
python3 -m unittest discover -s tests -v
```

主要入口是 `app.py`，包内也支持 `python -m spectral_workspace`。

## 交付内容

- `spectral_workspace/model.py`：材质、光源、场景配置及 JSON 校验。
- `spectral_workspace/standards.py`：内置 CIE 1931 2° / 1964 10° 5 nm 色匹配函数、D65/D50、Bradford/von Kries 基矩阵与目标色彩空间矩阵。
- `spectral_workspace/color.py`：光谱积分、光源/反射率项分离、白点适应、sRGB/DisplayP3/XYZ 输出。
- `spectral_workspace/pipeline.py`：阶段图、依赖签名、内容哈希、增量复用和全量重建校验。
- `spectral_workspace/operations.py`：带顺序版本号的场景编辑操作。
- `spectral_workspace/persistence.py`：暂存事务、快照、`CURRENT` 原子提交指针、关闭标记、孤儿回收与快照版本规则。
- `spectral_workspace/injection.py`：参数层和阶段层异常注入配置。
- `spectral_workspace/synth.py`：固定种子本地合成材质/光源与固定演示序列。
- `spectral_workspace/demo.py`：真实子进程强杀、恢复比对和完整验收演示。
- `tests/`：23 项 `unittest` 测试。
- `reports/demo_report.json`：逐阶段哈希、完整阶段产物、颜色、注入与恢复报告。
- `reports/unittest.log`：实际测试输出。

## 数据与颜色管线

每条材质包含材质 ID、380-780 nm 每 5 nm 的 81 个反射率采样、粗糙度与金属度；每个光源包含同波长网格上的 SPD 与混合权重。场景配置包含观察者、固定相机白点和适应方法。

每个“材质 × 光源”组合有两层阶段：

1. `spectral_xyz:<material>:<light>`：按当前观察者积分。产物同时记录光源积分项、反射率×光源积分项、归一化系数 `k`、积分步长、观察者、CMF 哈希和光源白点。
2. `adapted_xyz:<material>:<light>`：从光源白点适应到相机白点。产物记录完整适应矩阵、基矩阵/逆矩阵、锥响应缩放、源白点和目标白点。

每个材质再有一个 `material_color:<material>` 阶段，按光源权重聚合适应后 XYZ，并转换到目标色彩空间。sRGB/DisplayP3 同时输出线性 RGB、未裁剪编码值、裁剪后 RGB 和是否在色域内。粗糙度和金属度作为材质表面元数据进入最终阶段依赖，不伪称为完整 BRDF 渲染。

积分采用 `normalize_light_Y_to_100`：`k = 100 / (sum(SPD * ybar) * 5nm)`。完美漫反射体在光源下得到 Y=100。观察者名称与 CMF 表内容哈希都进入依赖签名，因此 2° 与 10° 观察者不会错误复用。

## 增量重算

阶段是否重算由依赖签名自动决定，而不是按整工作区刷新：

- 反射率或 SPD 改变：重算该材质/光源对的 `spectral_xyz`、`adapted_xyz` 和受影响材质的最终阶段。
- 粗糙度或金属度改变：只重算该材质的 `material_color`。
- 替换一个光源：只重算该光源的材质对阶段及最终聚合阶段。
- 切换观察者：全部光谱阶段因观察者/CMF 哈希变化而重算。
- 切换适应方法：光谱阶段复用，只重算适应阶段及最终聚合。

每次操作的 `history` 记录 `dirty_closure`、`recomputed` 和 `reused`。复用项带有原始 `source_version` 与 `content_hash`。正常编辑路径只做依赖闭包重算；报告、恢复和显式 `verify` 会执行无缓存全量重建对比，首个分岔阶段通过 `E_STAGE_STALE` 返回。阶段失效注入会自动强制该次对比且不会提交。

本次固定演示中的实际重算数量：

| revision | 操作 | 重算 | 复用 |
|---:|---|---:|---:|
| 1 | initialize | 15 | 0 |
| 2 | 编辑 mat_oxide 反射率 | 5 | 10 |
| 3 | 编辑 mat_polymer 粗糙度/金属度 | 1 | 14 |
| 4 | 替换 light_warm | 9 | 6 |
| 5 | 切换 CIE 1964 10° | 15 | 0 |
| 6 | 切换 von Kries | 9 | 6 |
| 7 | 未提交的 mat_metal 编辑 | 1 | 14 |

## 持久化与恢复

工作区布局：

```text
workspace/
  CURRENT                 # 唯一提交指针，原子替换
  CLOSED.json             # 正常退出关闭标记
  SESSION_OPEN.json       # 会话打开标记，强杀后残留
  snapshots/snap-*/       # state.json、manifest.json、operations.jsonl
  staging/txn-*/          # 未提交事务和逐阶段产物
  orphaned/txn-*/         # 恢复时回收的未提交事务
  recovery/               # 孤儿记录与恢复事件
```

提交顺序是：先写完整暂存事务与阶段产物，再写完整快照目录，最后原子替换 `CURRENT`。只有 `CURRENT` 指向的快照才算已提交；崩溃或强杀发生在指针替换前时，旧稳定状态不变。

恢复只读取 `CURRENT` 指向的快照，校验 manifest、状态哈希和每个阶段内容哈希，并执行“缓存协调 + 全量重建”双重验证。未提交暂存目录被移动到 `orphaned/`，同时写入 `recovery/orphan-records.jsonl`。强杀不会写 `CLOSED.json`，恢复报告明确标记 `exit_report_fabricated=false`。

快照整体替换比较 `snapshot_version`：旧版本和同版本冲突内容返回 `E_SNAPSHOT_VERSION`，不会覆盖新状态；更高版本可原子替换；同版本同哈希为幂等。

## 异常注入

注入仅由输入和 JSON 配置控制，不修改源码。支持：

- `reflectance_out_of_range`：指定材质、采样序号和越界值。
- `white_point_zero`：替换相机白点为含零分量值。
- `observer_invalid`：替换为非法观察者。
- `stage_failure`：在指定操作版本强制破坏指定阶段，用于验证 `E_STAGE_STALE` 和首个分岔定位。

注入配置会规范化、计算哈希并写入报告。参数层注入会保留原始 JSON Pointer，例如：

- `/materials/mat_oxide/reflectance/17`
- `/scene/camera_white_point_xyz/0`
- `/scene/observer`

## 业务错误码

| 错误码 | 含义 |
|---|---|
| `E_SPECTRUM_RANGE` | 采样缺失、波长非 380-780/5 nm、反射率越界/非有限、SPD 非法 |
| `E_WHITE_POINT` | 白点形状、非有限或零/负分量；光源白点不可计算 |
| `E_OBSERVER_INVALID` | 观察者不受支持 |
| `E_STAGE_STALE` | 增量阶段与全量重建不一致，或持久化阶段哈希失效 |
| `E_SNAPSHOT_VERSION` | 旧快照覆盖、同版本冲突或非法快照版本 |
| `E_INJECT_CONFIG` | 注入配置非法 |
| `E_ADAPTATION_INVALID` | 适应方法非法 |
| `E_COLOR_SPACE_INVALID` | 目标色彩空间非法 |
| `E_OPERATION_INVALID` | 操作版本/载荷/顺序非法 |
| `E_STATE_CORRUPT` | 稳定状态、`CURRENT` 或快照完整性损坏 |
| `E_INPUT_JSON` | 输入 JSON 无法读取 |
| `E_SCENE_INVALID` | 场景结构、ID、权重或表面参数非法 |

## CLI

```text
app.py generate           生成固定种子样本和演示操作
app.py validate           校验场景 JSON
app.py demo               执行固定编辑、注入、强杀恢复和报告
app.py apply              执行版本化操作并生成报告
app.py report             打开已有工作区并生成报告
app.py restore-snapshot   按版本规则整体恢复快照
```

例如：

```bash
python3 app.py apply \
  --workspace runtime/apply \
  --operations samples/demo_operations.json \
  --report reports/apply_report.json
```

## 实际验证结果

验证日期：2026-11-19。环境：Python 3.11、NumPy 2.4.6、本地文件系统，无容器、无外部服务。

- `python3 -m unittest discover -s tests -v`：23 项测试全部通过，耗时 12.580 s。
- 固定样本 SHA-256：`sha256:c3b163141cc7465b218bc6ae4eae02e3ddd6aec151fe884f21672f1b5da8420b`。
- 操作序列规范化哈希：`sha256:80887e84c1c9a5e480f4a6aa2d867c699e1757b9ede7cc757eb6106bdbce43c6`。
- 稳定 revision 6 状态哈希：`sha256:c5e8e03670c89b88df1be48343b0a9d3f4c69c26c4fad4a78c261eb16c36e03b`。
- 强杀前未提交 revision 7 的暂存事务被回收为 `txn-000000000007-d3fd2956fe83`。
- 恢复后 revision/snapshot 均为 6，15 个阶段哈希与基线逐字节一致。
- 恢复后颜色与基线逐字节一致，颜色最大绝对差 `0.0`，阶段最大绝对差 `0.0`。
- 两次独立演示报告 SHA-256 均为 `858172cea5805f0f637aca8161d4855d1b0aadc3c8acaf3848741a8f823a18e3`，`cmp` 返回 0。
- 注入链路实测返回：`E_SPECTRUM_RANGE`、`E_WHITE_POINT`、`E_OBSERVER_INVALID`、`E_STAGE_STALE`、`E_INJECT_CONFIG`。

可直接查看 `reports/demo_report.json` 复算各阶段积分项、矩阵、哈希和最终颜色。

## 设计边界

- 颜色结果按任务要求完成光谱到 XYZ、白点适应和目标色彩空间变换；粗糙度/金属度作为表面参数参与版本与最终阶段依赖，但不实现完整 BRDF/几何渲染。
- 多光源按场景权重对适应后 XYZ 做加权平均，权重和必须为正。
- 当前应用按单进程工作区模型设计；多进程并发写入不在本任务范围内。
- 稳定状态和阶段哈希使用 15 位有效数字规范化，以消除跨平台浮点末位噪声并保持报告可重复。
