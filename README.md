# RTS 资源征收与单位调度回放工作台

固定种子确定性生成即时战略对局事件流，逐 tick 合成权威资源与单位状态，在交互式时间轴上
回放，并输出可复算的逐 tick 状态哈希、账本与预算降级报告。在线帧驱动回放与离线全量合成
共用同一引擎，保证任意 tick 状态逐字段一致。

## 环境准备

- Python 3.11+，pip 可用
- 依赖：`numpy`、`PySide6`（见 `requirements.txt`，可用离线 wheel 或已配置源安装）
- 无容器、无远程服务、无外部样本：事件流/规则/样本全部由生成器程序化产出
- 可写目录 `workspace/` 持久化快照、清单与报告，可被外部强杀以验证恢复语义

```bash
python -m pip install -r requirements.txt
```

## 快速开始

```bash
# 固定演示序列：生成 -> 离线合成 -> 在线回放(含跳转/倍速) -> 核对 -> 预算报告
python -m rts_replay --workspace workspace demo
python -m unittest discover -s tests          # 34 个测试
python -m rts_replay --workspace workspace gui # 交互界面
python -m rts_replay --workspace workspace hashcheck
```

Windows 可直接运行 `run_demo.bat` 与 `run_gui.bat`。

## 样本集

`workspace/samples/` 下五个确定性场景：正常局、资源越界局、同 tick 并发争用局、倍速跳转
局、序号断裂局。每个场景含 `events.jsonl`、`rules.json`、`manifest.json`（事件计数与
内容 SHA-256）。相同种子两次生成逐字节相同。

## 产物

- `workspace/snapshots/<scenario>/snapshot_*.json`：原子写快照（GUI 用 `gui_snapshots/`）
- `workspace/reports/<scenario>/tick_hashes.jsonl`：逐 tick 状态哈希与软错误
- `workspace/reports/<scenario>/ledger.jsonl`：逐事件账本（含每事件 RNG 值）
- `workspace/reports/<scenario>/verification.json`：在线/离线核对、跳转等价、错误审计
- `workspace/reports/<scenario>/online_budget.json`：逐帧耗时与降级统计
- `workspace/reports/master_report.json|.txt`：跨场景汇总

## 进一步阅读

设计原理、确定性/无前瞻论证、降级语义、错误码表与实测结果见 [DESIGN.md](DESIGN.md)。
