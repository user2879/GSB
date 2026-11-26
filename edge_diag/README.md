# 边缘缓存分片下发诊断工作台

演示同一批缓存分片在多节点租约水位编辑、配额版本换代、子进程强制终止后，
结论能否被原样复核。节点为独立子进程，回环 TCP + JSON 行协议，NumPy 整数
定点校验，SQLite 原子账本，独立只读重放，Tkinter 只读水位界面。

## 快速开始

```bash
# 在仓库根目录准备环境（系统缺 python3-venv 时可用 virtualenv .venv）
python3 -m venv .venv && source .venv/bin/activate
pip install -r edge_diag/requirements.txt

cd edge_diag
bash scripts/demo.sh                       # 一键：样本->运行->崩溃恢复->重放
```

单独执行：

```bash
python -m edge_cache.cli gen-sample
python -m edge_cache.cli run
python -m edge_cache.cli crash-demo --crash-seq 3 --crash-node node-b
python -m edge_cache.cli verify
python -m edge_cache.cli ui
```

## 测试

```bash
python -m pytest edge_cache/tests/          # 无 Tk 时 UI 用例自动跳过
source scripts/tk-env.sh && python -m pytest edge_cache/tests/
```

## 文档

- `docs/DESIGN.md`：架构、确定性口径、双路回收裁决、账本与恢复设计
- `docs/VALIDATION.md`：实际验证结果
- `artifacts/`：演示输出与账本证据

## Tk 说明

系统未安装 `python3-tk` 时，项目内置 `.tk-python`（tkinter 包与扩展）和
`.sysroot`（libtk/tcl 脚本）。`source scripts/tk-env.sh` 后启用；无显示时
可用 `scripts/start-xvfb.sh` 在用户命名空间启动本地 Xvfb（`DISPLAY=unix/:77`）。
