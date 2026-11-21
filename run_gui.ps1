# Windows 启动专业桌面界面（需官方 Python 安装时勾选 tcl/tk）
param([string]$Workspace = ".\gui_workspace")
$ErrorActionPreference = "Stop"
python -m tier_mover.gui
Write-Host "GUI 已退出，工作区: $Workspace"
