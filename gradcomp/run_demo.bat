@echo off
REM ============================================================
REM  gradcomp 可重复演示脚本（Windows 本地回环 TCP）
REM  顺序执行：未压缩步 -> 压缩步 -> 压缩快照恢复续训重放
REM  用法: run_demo.bat [loopback^|tcp]
REM ============================================================
setlocal
cd /d %~dp0
set MODE=%1
if "%MODE%"=="" set MODE=loopback
python run_demo.py %MODE%
endlocal
