# 跨故障域租约仲裁与配额预算诊断工作台 · 可重复演示（Windows / PowerShell）
# 用法：在仓库根目录执行  powershell -ExecutionPolicy Bypass -File scripts/demo.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Test-Path .venv)) {
    python -m venv .venv
}
. .venv\Scripts\Activate.ps1
python -m pip install --quiet numpy pytest

Write-Host "== 1. 环境自检 =="
python -c "import sys, sqlite3, numpy; print('python', sys.version.split()[0]); print('sqlite3', sqlite3.sqlite_version); print('numpy', numpy.__version__)"
python -c "import tkinter" 2>$null; if ($LASTEXITCODE -ne 0) { Write-Host "tkinter 不可用：工作台自动降级为无头模式" }

Write-Host "== 2. 生成确定性样本（小/中两档）=="
python -m faultisolation.cli gen-samples --out samples

foreach ($scale in @(
    @{name="small"; spec="samples/small/spec-small.json"; sess="samples/small/session.json"; ledger="artifacts/small-live.sqlite"},
    @{name="medium"; spec="samples/medium/spec-medium.json"; sess="samples/medium/session.json"; ledger="artifacts/medium-live.sqlite"}
)) {
    Write-Host "== 3.$($scale.name) 强校验与路径选择 =="
    python -m faultisolation.cli --spec $scale.spec --session $scale.sess validate
    python -m faultisolation.cli --spec $scale.spec --session $scale.sess plan

    Write-Host "== 4.$($scale.name) 真实回环实时仲裁（子进程+线程池+TCP）=="
    python -m faultisolation.cli --spec $scale.spec --session $scale.sess live --ledger $scale.ledger --replace

    Write-Host "== 5.$($scale.name) 独立校验：三路径逐字节一致 + 槽位唯一 + 账本重放 =="
    python -m faultisolation.cli --spec $scale.spec --session $scale.sess verify --ledger $scale.ledger

    Write-Host "== 6.$($scale.name) 整数定点配额预算诊断 =="
    python -m faultisolation.cli --spec $scale.spec --session $scale.sess diagnose
}

Write-Host "== 7. 强制终止与账本恢复演示 =="
python -c "from faultisolation.recovery_demo import run_recovery_demo; import json; print(json.dumps(run_recovery_demo('samples/small/spec-small.json','samples/small/session.json','artifacts/recovery-ledger.sqlite'), ensure_ascii=False, indent=2, default=str))"

Write-Host "== 8. 测试套件 =="
python -m pytest tests -q
Write-Host "演示完成。"
