# Windows 本地环境准备(无需容器/外网包)
# 用法: powershell -ExecutionPolicy Bypass -File scripts/setup_windows.ps1
$ErrorActionPreference = "Stop"

Write-Host "Python 版本:"
python --version

if (-not (Test-Path ".venv")) {
    Write-Host "创建虚拟环境 .venv ..."
    python -m venv .venv
}

Write-Host "校验标准库与 Tkinter ..."
& ".\.venv\Scripts\python.exe" -c "import sqlite3, hashlib, json, struct; print('stdlib ok')"
try {
    & ".\.venv\Scripts\python.exe" -c "import tkinter; print('tkinter ok', tkinter.TkVersion)"
} catch {
    Write-Warning "tkinter 缺失: 请重新运行 CPython 安装包并勾选 Tcl/Tk and IDLE 组件。"
}

Write-Host "运行测试 ..."
& ".\.venv\Scripts\python.exe" run_tests.py
