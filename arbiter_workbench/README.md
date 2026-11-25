# 跨环境配置仲裁诊断工作台

单机（WSL Ubuntu）、无容器、无外部服务的多环境部署配置仲裁系统。环境仲裁服务以子进程
方式在 `127.0.0.1` 高位端口提供真实 TCP 服务，样本由整数种子确定性生成，账本使用
标准库 SQLite。

## 环境准备

```bash
python3 --version            # 需要 3.11+
pip install -r requirements.txt
# Tkinter：优先 sudo apt install -y python3-tk；
# 无 sudo 时使用预置 Tcl/Tk 库的包装脚本 scripts/env_wsl.sh
./scripts/env_wsl.sh python3 -c "import tkinter; print(tkinter.TkVersion)"
```

## 常用命令

```bash
# 全量仲裁（启动 4 个子进程服务 → 套接字收发 → 合并 → SQLite 原子提交）
python3 -m arbiter.cli run

# 基于变更集的增量仲裁与失效传播
python3 -m arbiter.cli incremental --changed-envs env-canary-c
python3 -m arbiter.cli incremental --policy-changes '{"view-dr-quorum": ["pool.max_size"]}'

# 快照导出 / 整体替换恢复 / 重启重放校验
python3 -m arbiter.cli snapshot --out decls/snapshots/expected/snapshot.json
python3 -m arbiter.cli restore  --snapshot decls/snapshots/expected/snapshot.json
python3 -m arbiter.cli recover

# 强制终止恢复演示
python3 scripts/crash_recovery_demo.py

# 诊断界面（无显示自动无头降级）
python3 -m arbiter.gui --headless
./scripts/env_wsl.sh python3 -m arbiter.gui

# 测试
./run_tests.sh
```

## 业务码

`E_SPEC` 声明非法 · `E_PORT` 端口问题 · `E_TIMEOUT` 套接字超时 · `E_ORDER` 确定性破坏 ·
`E_QUANT` 定点规则违反 · `E_STATE` 快照/账本状态冲突 · `E_REPLAY` 重放不一致 ·
`E_INTERNAL` 内部不变量失败。详见 `docs/DESIGN.md`。
