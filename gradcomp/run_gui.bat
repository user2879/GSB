@echo off
REM 启动 Tkinter 可视化界面（需要安装时勾选 tcl/tk）
cd /d %~dp0
python -m gradcomp gui --config samples/compressed_tcp.json --workdir workdir/gui
