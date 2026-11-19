# 端到端验收流水线（Windows PowerShell）。
# 前置：已安装 Windows 版 Python 3.11 并加入 PATH。
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    py -3.11 -m venv .venv
    .\.venv\Scripts\python.exe -m pip install --upgrade pip
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
}

.\.venv\Scripts\python.exe -m refrac --workspace . clean
.\.venv\Scripts\python.exe -m refrac --workspace . suite
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
