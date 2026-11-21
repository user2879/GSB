param([string]$Workdir = "demo_output")
python scripts/run_demo.py --workdir $Workdir
exit $LASTEXITCODE
