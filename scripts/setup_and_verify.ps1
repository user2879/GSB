$ErrorActionPreference = "Stop"

python --version
python -m pip install numpy pyyaml
python -m pip show numpy pyyaml

python -m vibpipe.cli generate --out examples/local_sample --injections examples/injections
python -m unittest discover -s tests
python -m vibpipe.verify

python -m vibpipe.cli run `
  --config examples/local_sample/pipeline_config.json `
  --samples examples/local_sample/samples.jsonl `
  --out out/demo

Write-Host "Verification passed. See artifacts/verification_report.json and out/demo/reports/pipeline_report.json"
