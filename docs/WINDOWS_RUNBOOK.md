# Windows 本地运行手册

## 1. 环境准备
1. 安装 Python 3.11 或 3.12 的 Windows 官方安装包，安装时勾选
   “Add python.exe to PATH”。
2. 打开 PowerShell 验证：
   ```powershell
   python --version
   pip --version
   ```
3. 安装唯一第三方依赖（SQLite 与 Tkinter 均为标准库，无需额外安装）：
   ```powershell
   pip install -r requirements.txt
   ```
4. 本组件不使用 Docker、不访问网络服务、不需要外部账号。

## 2. 目录约定
- `sample/`：输入声明 JSON（自带小、中、大三份确定性样本）。
- `ledger/`：SQLite 账本文件输出目录（运行时自动创建）。

## 3. 运行测试
在项目根目录执行：
```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

## 4. 端到端演示
```powershell
$env:PYTHONPATH = "src"
python scripts/run_demo.py
```
演示会校验两次生成的告警逐字节一致、提交账本、重放并输出零差异结果，
同时打印两条路径的取舍理由、误差界、抑制数、根因排序与租约占用峰值。

## 5. 强制终止恢复演示
```powershell
$env:PYTHONPATH = "src"
python scripts/crash_recovery_demo.py
```
脚本先提交第 1 轮、把未提交的第 2 轮标记为 `unknown`，再模拟重启：
仅依据已完整提交轮次恢复，未知轮次要求重跑且不生成退出报告，重放逐项比对为零差异。

## 6. CLI 用法
```powershell
$env:PYTHONPATH = "src"
python -m arcore.cli --declaration sample/declaration.json `
  --ledger ledger/cli.sqlite3 --rounds 3
```
也可安装为命令：`pip install -e .` 后直接使用 `arcore`。

## 7. 路径与跨平台说明
所有文件路径通过 `pathlib.Path` 处理；时钟仅使用 `time.monotonic_ns`；
线程与 SQLite 均为标准库能力，相同声明在 Windows 与其他系统上得到相同摘要与关联结论。
