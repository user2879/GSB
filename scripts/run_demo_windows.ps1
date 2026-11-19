# 一键确定性演示: 写入->崩溃->恢复->视图重建校验->快照->重放
$ErrorActionPreference = "Stop"
& ".\.venv\Scripts\python.exe" -m dwb_diag.cli demo --root "data\windows-demo"
Write-Host "启动 Tkinter 诊断界面(可选) ..."
& ".\.venv\Scripts\python.exe" -c "from dwb_diag.gui import launch; launch(r'data\windows-demo')"
