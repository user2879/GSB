# 透明介质折射合成组件（NumPy / Python 3.11）

面向图形渲染的透明介质折射合成领域组件：确定性程序化场景、光线与透明面求交、
斯涅尔折射与全反射、介质吸收路径衰减、递归深度控制、逐像素路径报告，
以及只经输入/参数生效的可复现异常注入与像素隔离。

## 快速开始

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt   # numpy，版本见 requirements.txt
.venv/bin/python -m refrac --workspace . suite        # 端到端验收流水线
.venv/bin/python -m unittest discover -s tests -v     # 40 个单元/集成测试
```

Windows 使用 `scripts/setup_windows.bat` 或 `scripts/run_all.ps1`；
Linux/macOS/WSL 使用 `scripts/run_all.sh`。

## 命令

- `refrac clean`：清理并重建 `samples/reports/artifacts`。
- `refrac generate`：固定种子生成场景 JSON 与元数据清单。
- `refrac render --scene ... [--injection ...]`：渲染一帧并落盘 RGBA/报告/哈希。
- `refrac suite`：双跑字节比对、采样/深度变体、注入套件、最小反例验证。
- `refrac replay --scene ... --injection ... --row R --col C`：单像素复算。

## 目录

- `refrac/`：领域组件源码（模块职责见 `docs/TECHNICAL_NOTES.md`）。
- `tests/`：unittest 测试与确定性 TIR 探针场景。
- `samples/ reports/ artifacts/`：程序生成的场景、报告与哈希产物。
- `scripts/`：Windows / POSIX 环境搭建与运行脚本。

## 验收要点

- 同输入两次合成逐字节一致（float32 RGBA 与路径报告 SHA-256 相同）。
- 比尔—朗伯衰减与解析式在 1e-10 容差内一致；采样数变化只影响吸收像素。
- 全反射按临界角判定并标注位置/次数；退化临界角归类 `NO_CRITICAL_ANGLE`，不丢光线。
- 七个业务错误码覆盖异常链路；单像素失败被隔离且不终止整帧。
- 首个受影响像素按行优先、列升序定位；最小反例可单独复现且移除任一字段即失效。
