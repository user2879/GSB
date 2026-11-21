# Windows PowerShell 一键演示（无需 Docker / 外部服务）
# 用法: .\run_demo.ps1 [-Workspace .\demo_workspace]
param(
    [string]$Workspace = ".\demo_workspace",
    [int]$Seed = 20261122
)
$ErrorActionPreference = "Stop"
Write-Host "==> Python 版本:"
python --version
Write-Host "==> 初始化工作区 $Workspace"
python -m tier_mover.cli init $Workspace --seed $Seed
Write-Host "==> 推导迁移计划"
python -m tier_mover.cli plan $Workspace
Write-Host "==> 执行批次迁移"
python -m tier_mover.cli run $Workspace
Write-Host "==> 账本视图"
python -m tier_mover.cli status $Workspace
Write-Host "==> 重放校验"
python -m tier_mover.cli replay $Workspace
Write-Host "==> 全场景演示（含三类注入与崩溃恢复）"
python -m tier_mover.cli demo "$Workspace-demo" --seed $Seed
