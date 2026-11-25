# ReconFlow

ReconFlow 是一套可集成本地 Python 包与可重复演示，用于回答：滚动追加、连续编辑的训练样本流，其加权、去重和配比结论能否在 Tkinter 诊断界面中即时查看，并在事后通过 SQLite 账本和权威重算唯一复核。

核心特性：

- 固定整数种子、BLAKE2b/SHA-256 驱动的确定性领域文档与 NDJSON 批次流。
- JSON/NDJSON 强校验，首个非法记录以 `E_SPEC` 给出行号和字段名。
- 加权采样、精确整数相似度去重、配比归一的万分之一定点管线。
- 快速材料化视图与权威全量重算分离；不一致以 `E_DIVERGE` 定位。
- 权重、阈值、分片大小、单篇文档编辑后的引用图失效传播和仅子集重算。
- 127.0.0.1 真实 TCP NDJSON 算力子进程，全局/领域双点预算和排队计时。
- SQLite v1 账本、原子事务、快照、SIGKILL 后 unknown 标记、重跑和 `E_REPLAY`。
- 定点误差归因、稳定候选裁决和 Tkinter 四页诊断台。

## 快速开始

```bash
python3 --version
python3 -m pip install numpy==1.26.4

export PYTHONPATH=src
python3 -m reconflow demo --root artifacts/demo
python3 -m reconflow ui --root artifacts/demo
python3 -m unittest discover -s tests -v
```

也可以安装后使用控制台命令：

```bash
python3 -m pip install .
reconflow demo --root artifacts/demo
reconflow ui --root artifacts/demo
```

一键验证脚本：

```bash
./scripts/verify.sh
```

## 常用命令

```bash
reconflow generate --root artifacts/demo
reconflow validate --root artifacts/demo
reconflow init-cache --root artifacts/demo --tcp
reconflow view --root artifacts/demo --record
reconflow verify-view --root artifacts/demo

reconflow edit-weight alpha 4000 --root artifacts/demo --tcp
reconflow edit-threshold 8800 --root artifacts/demo --tcp
reconflow edit-shard-size 8 --root artifacts/demo
reconflow edit-doc alpha:d00003 \
  --term-counts '[12,1,1,1,1,1,1,1,1,1,1,1]' \
  --missing-bitmap '[0,0,0,0,0,0,0,0,0,0,0,0]' \
  --root artifacts/demo --tcp

reconflow ledger-init --root artifacts/demo
reconflow process-batches --root artifacts/demo
reconflow recover --root artifacts/demo
reconflow persist-view --root artifacts/demo
reconflow replay --root artifacts/demo
reconflow snapshot --root artifacts/demo
reconflow snapshots --root artifacts/demo
```

崩溃恢复演示由 `reconflow demo` 自动执行：子进程在 SQLite 提交前被 `SIGKILL`，重启后无完整提交结局的批次标记为 unknown，随后重跑并逐项 replay。

## 文档

- `docs/TECHNICAL.md`：架构、定点舍入/饱和规则、失效图、TCP 协议、账本恢复、误差归因和 UI。
- `docs/VERIFICATION.md`：本次环境中的实际命令、测试结果、哈希、预算峰值和 Tk 冒烟结果。
- `artifacts/demo/verification_report.json`：机器可读的端到端验收报告。
- `artifacts/demo/diagnostic_view.txt`：诊断界面的稳定文本投影。

## Tkinter

Ubuntu/WSL 正常安装方式：

```bash
sudo apt-get install -y python3-tk
```

若 Tk 不可用，CLI 返回结构化 `E_UI` 或 `E_UI_DISPLAY`；数据层、demo 和文本诊断投影仍可完整运行。
