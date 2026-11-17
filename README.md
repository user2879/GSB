# 本地确定性混沌工程故障注入与可复现验证组件

面向 Windows 本地回环环境的混沌工程领域组件。用一份本地 JSON 声明实验，编排器以整数
随机种子推进可复现的注入时间轴，通过四种**真实机制**对回环端口上的确定性模拟进程注入
故障，采集注入前后的成功率与延迟观测，写入 SQLite 账本，并支持强制终止后仅依据已提交
账本恢复、整体替换与重放差异定位。

* 仅依赖 Python 3.11+ 标准库：`tkinter`、`sqlite3`、`subprocess`、`socket`、`threading`、`ctypes`
* 不使用 Docker/容器，不访问外网，不需要账号或第三方包
* 所有目标均为 `127.0.0.1` 高位端口上的本地子进程

## 目录结构

```
chaoskit/                可集成 Python 包
  spec.py                声明加载与强校验（E_SPEC，整体拒绝，输出字段路径）
  rng.py                 自研 SplitMix64 确定性整数 PRNG
  timeline.py            种子驱动时间轴、摘要与 E_TIMELINE 首偏差校验
  mock_service.py        确定性目标服务（真实子进程，行协议）
  process_control.py     Windows/POSIX 真实挂起、恢复、强杀
  target.py              子进程托管、孤儿回收、延迟代理、状态区分
  observe.py             单调时钟采样、定点成功率/分位数与量化误差界
  ledger.py              SQLite 账本、事务、unknown、版本识别、整体替换
  engine.py              编排器与四种注入执行
  replay.py              重放校验与 E_REPLAY 字段级差异定位
  viewmodel.py / gui.py  无 Tk 视图模型 + Tkinter 诊断界面
  cli.py                 命令行入口
examples/                样本声明、漂移声明、演示脚本
tests/                   单元与端到端集成测试（29 个）
docs/DESIGN.md           设计与技术说明（确定性证明、口径、Windows 要点）
artifacts/demo/          演示产物（账本、只读备份、汇总）
```

## 快速开始

```bash
# 强校验声明
python -m chaoskit.cli validate examples/sample_experiment.json

# 查看确定性时间轴（两份乱序声明摘要一致）
python -m chaoskit.cli timeline examples/sample_experiment.json
python -m chaoskit.cli timeline examples/sample_experiment_shuffled.json

# 运行实验（生成 artifacts/run/ledger.sqlite3）
python -m chaoskit.cli run examples/sample_experiment.json --workdir artifacts/run

# 中断后恢复续跑（未提交的在途注入标记 unknown 并重跑）
python -m chaoskit.cli recover examples/sample_experiment.json --workdir artifacts/run

# 重放校验（结构 + 可选观测容差比对）
python -m chaoskit.cli replay examples/sample_experiment.json --workdir artifacts/run --observations

# 诊断界面（Windows 带显示环境）；无显示环境用 --headless 输出同一数据的文本报告
python -m chaoskit.cli gui examples/sample_experiment.json --workdir artifacts/run --headless
```

## 一键演示（正常 / 强杀恢复 / 重放差异 三场景）

```bash
python examples/demo_run.py
```

演示顺序执行：

1. **可复现性**：乱序声明时间轴摘要逐字节一致；两次独立运行的成功率结论严格相等。
2. **强杀恢复**：父进程 `kill -9` 工作进程，重启后仅依据已提交事务恢复，在途注入标
   `unknown` 并重跑，报告已恢复数、unknown 序号与剩余预算（不承诺退出报告）。
3. **重放差异定位**：用漂移声明（种子与目标配置变化）触发 `E_REPLAY`，定位首个不一致
   注入序号、字段与两侧取值；随后演示账本整体替换、只读备份与替换记录。
3b. 两次独立运行的观测摘要在量化/调度容差内通过重放校验。

## 声明示例（节选）

```json
{
  "version": 1,
  "experiment_id": "loopback-chaos-demo",
  "seed": 20261118,
  "budget": {"max_injections": 8, "max_parallel": 1},
  "observation": {"sample_interval_ms": 20, "min_samples": 4},
  "services": [
    {"service_id": "orders", "host": "127.0.0.1", "data_port": 18081,
     "proxy_port": 19081, "fixed_cost_ms": 12, "base_error_bps": 0}
  ],
  "injections": [
    {"injection_id": "inj-orders-delay", "type": "network_delay",
     "target_id": "orders", "duration_ms": 160, "params": {"delay_ms": 60}},
    {"injection_id": "inj-orders-errors", "type": "error_rate",
     "target_id": "orders", "duration_ms": 160, "params": {"error_bps": 5000}},
    {"injection_id": "inj-payments-suspend", "type": "suspend",
     "target_id": "payments", "duration_ms": 140, "params": {}},
    {"injection_id": "inj-payments-kill", "type": "terminate",
     "target_id": "payments", "duration_ms": 120, "params": {}}
  ]
}
```

错误比例以基点 `error_bps`（0–10000）整数定点表示；服务端口仅允许 `127.0.0.1` 且
1–65535 不重复；任一字段非法即以 `E_SPEC` 整体拒绝并给出全部 JSON 路径，绝不部分生效。

## 包 API

```python
from chaoskit import (
    load_spec, build_timeline, timeline_digest,   # 声明 + 时间轴
    ChaosEngine,                                   # 注入执行
    aggregate, collect_phase,                      # 观测聚合
    Ledger, open_or_replace_ledger,                # 账本保存/恢复/替换
    verify_identity, verify_timeline_entries, verify_observations,  # 重放
)
```

## 业务码

| 码 | 含义 |
| --- | --- |
| `E_SPEC` | 声明非法，`details.paths` 列出全部字段路径，整体拒绝 |
| `E_TIMELINE` | 重生成时间轴与账本摘要偏差，含首个偏差注入序号与两侧值 |
| `E_TARGET` | 服务不可达/端口被未知进程占用/挂起后仍可达等 |
| `E_LEDGER` | 账本格式未知或内容摘要不匹配（疑似损坏/外部篡改） |
| `E_REPLAY` | 重放不可复现，含依据、注入序号、字段名与两侧取值 |

## 测试

```bash
python -m unittest discover -s tests -v
```

确定性原理、观测口径与量化误差界、事务恢复与 Windows 原语细节见 `docs/DESIGN.md`。
