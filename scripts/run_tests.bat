@echo off
cd /d "%~dp0\.."
python -m unittest discover -s pool_workbench\tests -p "test_*.py" -v
pause
