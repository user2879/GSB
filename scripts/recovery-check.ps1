[CmdletBinding()]
param(
    [string]$Sample = "samples/demo",
    [string]$State = ".runtime/recovery-demo"
)

$ErrorActionPreference = "Continue"
python -m lease_arbitrator init-sample --dir $Sample --force
if (Test-Path $State) { Remove-Item -Recurse -Force $State }

python -m lease_arbitrator run --sample $Sample --state-dir $State --crash-batch 4
if ($LASTEXITCODE -ne 97) { throw "expected injected hard-kill exit code 97" }

python -m lease_arbitrator recover --sample $Sample --state-dir $State
python -m lease_arbitrator run --sample $Sample --state-dir $State
if ($LASTEXITCODE -ne 0) { throw "recovery rerun failed" }

python -m lease_arbitrator verify --sample $Sample --state-dir $State
if ($LASTEXITCODE -ne 0) { throw "post-recovery verification failed" }
