"""本地命令执行器：Windows 走 cmd 内置，POSIX 走 sh，仅使用标准库。

关键约定
--------
- 命令字符串支持 ``${PYTHON}`` 占位符，执行前替换为当前解释器，
  使同一份清单在 Windows 与 Linux 下可移植，且占位符替换属于规范化参数。
- 标准输出按字节捕获后做换行标准化（CRLF/CR -> LF），再参与层摘要，
  保证 Windows 与 POSIX 的同一逻辑命令得到相同 stdout 摘要。
- 超时使用显式机制：先 ``terminate()``（Windows 为 TerminateProcess，
  POSIX 为 SIGTERM）宽限期后仍存活则 ``kill()``（SIGKILL / 强制结束），
  并回收子进程。返回 ``timed_out=True`` 与残留状态说明，绝不把超时描述成
  「已终止失控命令」——我们只保证终止由本工作台直接派生的进程句柄。
- 环境变量基于父进程环境叠加阶段环境，字典按名稳定序构造。
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass

from . import digest

# terminate() 后的宽限毫秒；超时收尾所用，不参与层摘要
_GRACE_MS = 200


@dataclass
class CommandResult:
    command: str
    exit_code: int
    stdout_normalized: str
    stdout_digest: str
    elapsed_ms: int
    timed_out: bool
    timeout_ms: int
    residual: str          # 残留状态的显式说明
    mechanism: str         # 实际采取的终止机制说明


def expand_command(command: str) -> str:
    """替换可移植占位符；未知占位符原样保留并由 shell 自行处理。"""
    return command.replace("${PYTHON}", sys.executable)


def run_command(command: str, timeout_ms: int, env: dict[str, str] | None = None,
                cwd: str | None = None) -> CommandResult:
    """执行本地命令并捕获退出码、标准化 stdout、耗时与超时残留。"""
    expanded = expand_command(command)
    is_windows = os.name == "nt"
    if is_windows:
        argv = ["cmd", "/c", expanded]
    else:
        argv = ["/bin/sh", "-c", expanded]

    # 独立进程组，便于在 POSIX 上整组终止；Windows 上终止进程句柄
    posix_kwargs = {}
    if not is_windows:
        posix_kwargs["start_new_session"] = True

    merged_env = dict(os.environ)
    if env:
        for key in sorted(env):
            merged_env[key] = env[key]

    start = time.monotonic()
    proc = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=cwd,
        env=merged_env,
        **posix_kwargs,
    )
    timed_out = False
    residual = "无残留：子进程在超时前正常退出并被 wait 回收"
    mechanism = "正常等待退出，未触发终止"
    try:
        stdout_bytes, _ = proc.communicate(timeout=timeout_ms / 1000.0)
        exit_code = proc.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        mechanism, residual = _terminate_tree(proc, is_windows)
        stdout_bytes, _ = proc.communicate()
        exit_code = proc.returncode if proc.returncode is not None else -1
    elapsed_ms = max(0, int(round((time.monotonic() - start) * 1000)))
    stdout_text = digest.normalize_newlines(
        stdout_bytes if stdout_bytes is not None else b""
    )
    return CommandResult(
        command=expanded,
        exit_code=exit_code,
        stdout_normalized=stdout_text,
        stdout_digest=digest.sha256_hex(stdout_text),
        elapsed_ms=elapsed_ms,
        timed_out=timed_out,
        timeout_ms=timeout_ms,
        residual=residual,
        mechanism=mechanism,
    )


def _terminate_tree(proc: subprocess.Popen, is_windows: bool) -> tuple[str, str]:
    """显式两级终止，返回 (机制说明, 残留状态说明)。"""
    mechanism_parts = []
    if is_windows:
        # Windows：terminate 调用 TerminateProcess(handle, 1)
        proc.terminate()
        mechanism_parts.append("TerminateProcess 结束派生进程句柄")
    else:
        # POSIX：先向整个进程组发 SIGTERM
        import signal
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            mechanism_parts.append("SIGTERM 进程组")
        except ProcessLookupError:
            mechanism_parts.append("SIGTERM 时进程已退出")
    try:
        proc.wait(timeout=_GRACE_MS / 1000.0)
    except subprocess.TimeoutExpired:
        if is_windows:
            proc.kill()
            mechanism_parts.append("宽限期后 kill() 强制结束")
        else:
            import signal
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                mechanism_parts.append("宽限期后 SIGKILL 进程组")
            except ProcessLookupError:
                pass
        proc.wait()
    residual = (
        "工作台直接派生的进程（POSIX 为其进程组）已终止并回收；"
        "若命令自行派生出脱离该进程组的孤儿进程，本工作台不承诺其生命周期，"
        "该不确定性仅影响环境清理、不参与层摘要"
    )
    return " -> ".join(mechanism_parts), residual
