@echo off
REM Windows 独立虚拟环境搭建与依赖安装。
where py >nul 2>nul
if errorlevel 1 (
  echo Python launcher ^(py^) not found. Install Python 3.11 from python.org first.
  exit /b 1
)
py -3.11 -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
echo Setup complete. Activate with: .venv\Scripts\activate.bat
