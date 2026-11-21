$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

py -3 -c "import sys, numpy; assert sys.version_info >= (3, 11); print(sys.version.split()[0], 'numpy', numpy.__version__)"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

py -3 -m unittest discover -s tests -v 2>&1 | Tee-Object -FilePath validation_tests.txt
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

py -3 -m aqeval demo --workspace validation_workspace 2>&1 | Tee-Object -FilePath validation_output.txt
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

py -3 -m aqeval generate --workspace sample_workspace 2>&1 | Tee-Object -FilePath validation_output.txt -Append
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

py -3 -m aqeval run --workspace sample_workspace 2>&1 | Tee-Object -FilePath validation_output.txt -Append
exit $LASTEXITCODE
