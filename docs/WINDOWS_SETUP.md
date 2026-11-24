# Windows 环境准备（Python 3.11+）

1. 安装 Python 3.11 或更高版本，安装时勾选 *Add python.exe to PATH*。
2. 在仓库根目录打开 PowerShell，创建并激活虚拟环境：
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install numpy pytest
   ```
3. 自检标准库组件：
   ```powershell
   python -c "import sqlite3; print(sqlite3.sqlite_version)"
   python -c "import tkinter; print('tk ok')"
   ```
   若 `tkinter` 不可用，工作台会自动降级为无头交互控制台，其余功能不受影响。
4. 确认高位回环端口未被占用（样本使用 19001-19002、19101-19103）：
   ```powershell
   Get-NetTCPConnection -State Listen | Where-Object { $_.LocalAddress -eq '127.0.0.1' }
   ```
5. 运行可重复演示：
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts/demo.ps1
   ```

说明：平台不使用 Docker 或容器，不访问外部网络目标，所有仲裁均发生在本机回环接口。
