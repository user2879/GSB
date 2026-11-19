# 固定序列演示（含强制终止恢复）。无显示器时自动 offscreen，仅 CLI 不依赖 GUI。
param([string]$Workspace = "workspace")
.\.venv\Scripts\python -m syncbench.cli demo --workspace $Workspace --crash
