param(
    [int[]]$Ports = @(19081, 19082, 19083)
)

$ErrorActionPreference = "Stop"
Write-Host "Python:"
python --version
Write-Host "NumPy:"
python -c "import numpy; print(numpy.__version__)"

foreach ($Port in $Ports) {
    $listener = Get-NetTCPConnection -LocalAddress 127.0.0.1 -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if ($listener) {
        throw "Loopback port $Port is already in use."
    }
    Write-Host "Loopback port $Port is free."
}

Write-Host "Ensure Windows Defender Firewall allows loopback traffic for python.exe if prompted."
