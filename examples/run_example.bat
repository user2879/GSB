@echo off
REM ============================================================================
REM 多视点光度立体 x 可微渲染交替优化 —— Windows 本地完整示例
REM 仅依赖标准库与 NumPy；无需 GPU、Docker 或任何外部服务。
REM 建议 Python 3.10+，先在项目根目录创建并激活虚拟环境：
REM     py -3 -m venv .venv
REM     .venv\Scripts\activate
REM     python -m pip install -r requirements.txt
REM ============================================================================
setlocal
set PY=python
set WS=workspace

echo [1/7] 生成 256x256 确定性观测集（6 光源 x 2 相机 = 12 视点）
%PY% -m photometric_altopt.cli generate --workspace %WS% --size 256
if errorlevel 1 goto fail

echo [2/7] 解析梯度有限差分核对
%PY% -m photometric_altopt.cli gradcheck --workspace %WS% --samples 32
if errorlevel 1 goto fail

echo [3/7] 交替优化基线运行（含真值核对）
%PY% -m photometric_altopt.cli run --workspace %WS% --run-name baseline --max-iterations 8 --check-truth
if errorlevel 1 goto fail

echo [4/7] 强制终止模拟（第 3 轮后崩溃），再恢复并续写
%PY% -m photometric_altopt.cli run --workspace %WS% --run-name crash_demo --max-iterations 8 --crash-after 2
%PY% -m photometric_altopt.cli resume --workspace %WS% --run-name crash_demo --max-iterations 8

echo [5/7] 参数层异常注入：非有限反照率 / 零向量光源 / 退化视点集合
%PY% -m photometric_altopt.cli run --workspace %WS% --run-name f_nonfinite --inject-nonfinite
%PY% -m photometric_altopt.cli run --workspace %WS% --run-name f_zero --inject-zero-light
%PY% -m photometric_altopt.cli run --workspace %WS% --run-name f_degenerate --inject-degenerate

echo [6/7] 管线层异常注入：渲染阶段跳过视点 3 与 7
%PY% -m photometric_altopt.cli run --workspace %WS% --run-name f_skipview --skip-view 3,7 --max-iterations 3

echo [7/7] 打印工作区内容哈希清单（恢复前后可对照）
%PY% -m photometric_altopt.cli manifest %WS%\trajectories\baseline
echo DONE
exit /b 0

:fail
echo EXAMPLE_FAILED
exit /b 1
