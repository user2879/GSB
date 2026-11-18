# QoS 流量整形规则诊断工作台

面向本地 Windows（同时兼容 Linux/macOS）环境的 QoS 规则编译、失效传播、整形验证、
账本恢复与会话快照重放工作台。**纯 Python 3.11 标准库实现，无容器、无第三方网络依赖、
无真实账号或线上目标**，所有网络通信仅发生在 `127.0.0.1` / `::1` 回环地址。

## 能力总览
- 规则集与依赖图加载、强校验（整体拒绝，错误定位到规则标识与字段路径）。
- 增量变更集原子应用，沿依赖图推导直接/间接下游失效集合与稳定传播轨迹。
- 规则到整形计划的确定性编译、单调版本号、计划整体替换与持久化摘要。
- 仅重编译失效子集，未失效产物逐字节复用、产物版本号不变，并提供增量=全量校验。
- 回环 TCP 子进程流量发生器/整形器/接收端，令牌桶限速与吞吐违规 `E_SHAPE` 报告。
- SQLite 账本持久化，强制终止后只恢复已提交记录，未提交规则标注 UNKNOWN。
- 会话快照版本识别、整体替换恢复、变化规则清单与恢复后一致性重放校验。
- Tkinter 可视化：依赖关系、失效传播集合、生效计划、诊断业务码（缺 Tkinter 时优雅降级）。

## 目录结构
```
uid178/
├── qos/                  # 核心包
│   ├── errors.py         # 业务码异常：E_RULE/E_DEPEND/E_REF/E_INCR/E_SHAPE/E_SNAP/E_REPLAY
│   ├── canonical.py      # 确定性 JSON 序列化与 SHA-256 摘要
│   ├── models.py         # Rule 数据模型与字段级强校验
│   ├── graph.py          # 依赖图、稳定拓扑、成环检测、下游传播与轨迹
│   ├── ruleset.py        # JSON 加载与整体拒绝语义
│   ├── changeset.py      # 变更集原子应用（新增/修改/删除/删边）与失效推导
│   ├── compiler.py       # 确定性编译器、版本化整体替换、增量/全量一致性
│   ├── shaper.py         # 虚拟时钟整形引擎、令牌桶、回环帧协议
│   ├── traffic 相关      # 见根目录 traffic_sim.py
│   ├── ledger.py         # SQLite 账本、pending 操作、崩溃恢复
│   ├── snapshot.py       # 快照持久化、版本识别、整体替换与变化清单
│   ├── replay.py         # 恢复后重放校验与首个差异定位
│   ├── workbench.py      # 门面：串联完整业务流
│   ├── cli.py / __main__.py  # 命令行入口
│   ├── gui.py            # Tkinter 界面（懒加载）
│   └── applog.py         # 结构化 JSONL 日志
├── samples/              # 确定性样本与异常注入样本
├── tests/                # unittest 测试（28 个）
├── traffic_sim.py        # 回环发生器+整形器+接收端子进程脚本
├── demo.py               # 一键诊断演示，产出 data/diagnostic_report.json
├── CONFIG.md             # 环境、路径、端口与命令配置说明
└── ARCHITECTURE.md       # 设计原理、版本/恢复/重放语义
```

## 快速开始
```bash
python --version            # 需 3.11+
python -m unittest discover -s tests -v
python demo.py              # 端到端 14 项验收检查并生成诊断报告
python -m qos --help        # 命令行帮助
python -c "from qos.gui import WorkbenchGUI; WorkbenchGUI().run()"  # 有 Tkinter 时
```

## 业务码
| 业务码 | 触发场景 |
| --- | --- |
| `E_RULE` | 字段校验失败、JSON 损坏、上游引用不存在（整体拒绝加载） |
| `E_DEPEND` | 依赖图成环 |
| `E_REF` | 删除仍被其他规则引用的规则（附引用者） |
| `E_INCR` | 增量编译与全量编译摘要不一致（附首个差异规则与字段） |
| `E_SHAPE` | 实际吞吐超过承诺速率（附承诺/实际整数对比） |
| `E_SNAP` | 快照版本字段缺失或快照不可识别 |
| `E_REPLAY` | 恢复后重放与快照记录不一致（附首个差异规则、字段与依据） |

## 确定性保证
- 所有规范化表示经 `canonical.canonical_bytes` 输出：`sort_keys=True`、固定分隔符、LF、UTF-8，
  不依赖 dict/set 遍历顺序与输入数组顺序。
- 拓扑序与计划优先级均为纯函数：同层按规则标识字典序，计划按最长端口前缀等确定性键排序。
- 虚拟时钟整形使用整数微秒与固定到达间隔，同输入逐字节可重放；真实回环模式额外提供进程级验证。

详见 `ARCHITECTURE.md` 与 `CONFIG.md`。
