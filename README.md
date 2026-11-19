# VGSynth：确定性矢量路径合成组件

VGSynth 是一个可在 Windows 本地运行的 Python 领域组件与完整示例。它从固定种子生成可复算的矢量路径场景，完成扁平化、扫描线填充、描边轮廓、预乘 Alpha 抗锯齿合成、分层哈希、参数级异常注入、独立验证和最小反例输出。

运行时不访问网络或外部素材。场景全部由固定种子程序化生成，输出为规范化 JSON、BGRA 原始缓冲、PNG 预览、覆盖率位图、层哈希清单和结构化报告。

## 快速开始

Windows PowerShell：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m vgsynth demo --out artifacts\demo
.\.venv\Scripts\python.exe -m vgsynth verify --run-dir artifacts\demo\run_a
```

本仓库已在 Linux/WSL 的同版本 Python 3.11 环境中实际验证；代码只使用标准库、NumPy 与 PySide6，不包含平台专用路径。Windows 详细步骤见 [docs/RUNNING.md](docs/RUNNING.md)。

## 固定演示序列

`demo` 默认执行两次完全相同的运行，分别写入 `run_a` 与 `run_b`，并生成 `comparison.json` 逐文件比较哈希。固定序列覆盖：

- 简单交叉、自相交五角星、大面积重叠路径、含退化段路径。
- 填充色 Alpha 越界、非法填充规则、坐标越界、负容差、零描边宽度。
- 两个注入操作写同一字段产生的 `E_INJECTION_CONFLICT`。
- 参数级像素缓冲变异与层哈希清单变异，用于触发 `E_LAYER_HASH` / `E_VERIFIER_MISMATCH`。

每次运行输出：

```text
artifacts/demo/
├─ comparison.json
├─ run_a/
│  ├─ report.json
│  ├─ run_manifest.json
│  ├─ hashes.json
│  └─ cases/<case_id>/
│     ├─ scene.json
│     ├─ injection.json
│     ├─ generation.json
│     ├─ flatten.json
│     ├─ layers.json
│     ├─ verification.json
│     ├─ issues.json
│     ├─ final.bgra
│     ├─ final.png
│     ├─ layers/*.bgra|*.png
│     ├─ coverage/*.raw|*.png
│     └─ minimal_counterexample/
└─ run_b/...
```

PySide6 预览：

```powershell
.\.venv\Scripts\python.exe -m vgsynth view --run-dir artifacts\demo\run_a
```

## 可集成 API

```python
from vgsynth import generate_scene, flatten_scene, composite_scene, verify_case_dir

scene = generate_scene("self_intersect_loop", seed=0xC0FFEE2026, scene_id="demo")
flat = flatten_scene(scene, tolerance=0.25)
result = composite_scene(scene, flat)

print(result["canvas_hash"])
print([(layer["layer_id"], layer["layer_hash"]) for layer in result["layers"]])
```

完整磁盘级运行与验证使用 `run_case`、`run_demo`、`verify_case_dir`。输入、输出与错误模型见 [docs/DESIGN.md](docs/DESIGN.md)。

## 核心约定

- 场景 JSON 使用排序键、紧凑分隔符和固定换行编码；随机决策记录子种子与确定性 PRNG 算法。
- 扁平化对三次贝塞尔采用确定性自适应二分；每段记录误差上界、顶点数及退化处理动作。
- 填充同时生成非零环绕与奇偶两张 8 位覆盖率位图；每像素使用 8 条子扫描线，每个采样行按像素中心水平射线求交。
- 描边先生成圆角连接/端帽多边形轮廓，再用非零环绕规则光栅化。
- 合成为 `BGRA8_PREMULTIPLIED`，固定顺序为填充在下、描边在上，最终画布按层 ID 升序继续预乘 Alpha 合成。
- 层哈希覆盖层头、双填充规则覆盖率哈希、描边覆盖率哈希和完整 BGRA 字节。
- 验证器从磁盘重新读取场景、注入配置、层清单及 BGRA 缓冲，重新计算并优先按层 ID、再按行优先像素顺序报告首个不一致。
- 最小反例在保留目标违规的前提下缩减到路径/子路径/段/注入操作，并逐项执行删除测试证明不可再删。

## 业务错误码

`E_SCENE_INVALID`、`E_PATH_DEGENERATE`、`E_FILL_RULE`、`E_COLOR_RANGE`、`E_TOLERANCE_RANGE`、`E_LAYER_HASH`、`E_VERIFIER_MISMATCH`、`E_INJECTION_CONFLICT`。

注入异常在参数层生效，不修改源码。非法值使用明确记录的恢复策略继续生成诊断输出，例如颜色钳制、坐标钳制、非法填充规则回退到 `nonzero`、负容差取绝对值、零宽描边跳过描边层。

## 实际验证

最终验证记录见 [docs/VERIFICATION_REPORT.md](docs/VERIFICATION_REPORT.md)。最近一次完整执行中：

- 12 个固定样本全部完成生成、扁平化、双规则填充、描边、合成、验证与最小反例定位。
- `run_a` 与 `run_b` 的全部输出文件哈希一致。
- 10 项自动化测试通过。
