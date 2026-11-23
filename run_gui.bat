@echo off
chcp 65001 >nul
cd /d "%~dp0"
python -m imv.cli gui --data-dir "%~dp0artifacts\gui-workbench" %*
if errorlevel 1 pause
