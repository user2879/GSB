# Windows 环境准备：创建独立虚拟环境并安装锁定依赖（无需 GPU/在线账号）
# 用法：在项目根目录执行  powershell -ExecutionPolicy Bypass -File setup.ps1
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
Write-Host "环境就绪：.\.venv\Scripts\python -m syncbench.cli demo --workspace workspace"
