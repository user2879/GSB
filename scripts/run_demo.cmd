@echo off
setlocal
python -m vibpipe.cli generate --out examples\local_sample --injections examples\injections || exit /b 1
python -m vibpipe.cli run --config examples\local_sample\pipeline_config.json --samples examples\local_sample\samples.jsonl --out out\demo || exit /b 1
python -m vibpipe.verify || exit /b 1
echo Verification passed. See artifacts\verification_report.json
