# 运行与安装说明

## 1. 环境要求

- Windows 10/11，64 位。
- Windows 版 Python 3.11 或更高版本。
- 可写的本地工作区。
- 不需要 Docker、WSL、外部素材或网络服务。
- 运行时依赖：NumPy、PySide6。PySide6 只用于桌面预览，合成与验证核心不依赖 GUI。

本次实现实际验证环境：

```text
Python 3.11.0rc1
numpy 2.4.6
PySide6 6.12.0
Linux/WSL x86_64（同一套 Python 代码，路径均为 pathlib）
```

Windows 上建议使用当前最新稳定版 Python 3.11+ 与上述依赖版本。`requirements.txt` 固定了本次验证使用的版本。

## 2. 创建独立虚拟环境

PowerShell：

```powershell
cd <工作区目录>
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

如果系统没有 `py` 启动器：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

确认版本：

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -c "import numpy, PySide6; print(numpy.__version__, PySide6.__version__)"
```

离线安装时可先在联网机器下载与 Windows/Python 版本匹配的 wheel，再执行：

```powershell
.\.venv\Scripts\python.exe -m pip install --no-index --find-links .\wheels -r requirements.txt
```

## 3. 运行自动化测试

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

测试覆盖：

- 场景 JSON 与种子审计逐字节确定性、子种子复算。
- 扁平化重复一致性、容差收紧顶点数单调不减、退化动作。
- 非零/奇偶覆盖率差异首点、描边覆盖、层哈希敏感性。
- 全部要求的异常注入分类。
- 独立验证器正常通过与首个像素不一致定位。
- 最小反例不可再删证明。
- 两套完整运行目录的逐文件哈希一致性。

## 4. 执行固定演示序列

```powershell
.\.venv\Scripts\python.exe -m vgsynth demo --out artifacts\demo
```

该命令会：

1. 创建 `artifacts\demo\run_a` 与 `artifacts\demo\run_b`。
2. 对每个样本执行生成、注入、扁平化、双规则填充、描边、合成、验证、最小反例。
3. 比较两轮所有输出文件的 SHA-256。
4. 写出 `artifacts\demo\comparison.json`。

关键断言：

```text
equivalent = true
only_first = []
only_second = []
differing = []
run_manifest_equal = true
```

如只生成一轮用于调试：

```powershell
.\.venv\Scripts\python.exe -m vgsynth demo --out artifacts\single --single
```

## 5. 验证输出

验证整个运行目录：

```powershell
.\.venv\Scripts\python.exe -m vgsynth verify --run-dir artifacts\demo\run_a
```

验证单一样本目录：

```powershell
.\.venv\Scripts\python.exe -m vgsynth verify --case-dir artifacts\demo\run_a\cases\normal_cross
```

退出码：

- `0`：验证通过；整轮验证中的参数声明型像素/哈希变异按预期识别为 `E_VERIFIER_MISMATCH`。
- `2`：单样本验证发现不一致，或整轮中出现未声明的不一致。
- 验证失败时 JSON 的 `first_mismatch` 给出层 ID、行优先像素坐标、通道、期望值与实际值。

## 6. 生成单个场景

```powershell
.\.venv\Scripts\python.exe -m vgsynth generate --kind simple_cross --seed 0xC0FFEE2026 --out artifacts\one_scene.json
```

可选场景：

- `simple_cross`
- `self_intersect_loop`
- `overlapping_large`
- `degenerate_segments`

## 7. 比较两次运行

```powershell
.\.venv\Scripts\python.exe -m vgsynth compare --first artifacts\demo\run_a --second artifacts\demo\run_b
```

比较单位为 `hashes.json` 中记录的全部产物，不依赖文件时间戳或输出绝对路径。

## 8. 桌面预览

```powershell
.\.venv\Scripts\python.exe -m vgsynth view --run-dir artifacts\demo\run_a
```

预览窗口左侧为样本列表，右侧为最终合成 PNG。预览仅用于人工检查，不参与确定性哈希。

## 9. 输出目录

程序会自动创建目录。每次 `demo` 前会清理目标 `run_a` / `run_b`，避免旧产物污染比较结果。典型输出：

```text
cases/<case_id>/scene.json
cases/<case_id>/injection.json
cases/<case_id>/generation.json
cases/<case_id>/flatten.json
cases/<case_id>/layers.json
cases/<case_id>/verification.json
cases/<case_id>/issues.json
cases/<case_id>/final.bgra
cases/<case_id>/final.png
cases/<case_id>/layers/L0000.bgra
cases/<case_id>/layers/L0000.png
cases/<case_id>/coverage/L0000.fill_nonzero.raw
cases/<case_id>/coverage/L0000.fill_evenodd.raw
cases/<case_id>/coverage/L0000.stroke.raw
cases/<case_id>/minimal_counterexample/
```

## 10. 常见问题

- `ModuleNotFoundError: numpy`：确认使用了 `.venv\Scripts\python.exe`，而不是系统 Python。
- PySide6 安装失败：确认 Python 为 64 位，并优先安装 wheel；无需系统级 Qt。
- 无法打开预览：在交互式桌面会话中运行；无桌面服务器时仍可执行全部合成与验证命令。
- 权限错误：确认工作区可写，或将 `--out` 指向用户目录。
- 验证失败且出现 `E_VERIFIER_MISMATCH`：检查 `verification.json.first_mismatch`，其中包含首个不一致层与像素坐标。
