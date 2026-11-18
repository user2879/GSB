@echo off
REM Windows 本地确定性演示（仅依赖 Python 3.11 标准库，无需 Docker）
setlocal
cd /d %~dp0
set SRC=samples\workspace
set MAN=samples\manifests\sample_manifest.json
set WD=workdir\demo
python -m layerbench.cli validate --manifest %MAN% --workdir %WD% --source-root %SRC% || exit /b 1
python -m layerbench.cli verify   --manifest %MAN% --workdir %WD% --source-root %SRC% || exit /b 1
for /f "delims=" %%d in ('dir /b /ad /o-n "%WD%\snapshots"') do (
  python -m layerbench.cli replay --snapshot "%WD%\snapshots\%%d" --workdir %WD% --source-root %SRC%
  goto :done
)
:done
endlocal
