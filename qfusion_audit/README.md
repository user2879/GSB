# 多路召回融合量化溯源组件（qfusion_audit）

面向融合打分器**定点化部署前验收**的本地确定性组件。三路召回（词面 lexical、
协同过滤 collaborative、热度 hotness）的候选得分先做通道内归一化，再在定点整数域
按通道权重与温度合成融合分、叠加排序特征、裁剪并排序。交付重点不是召回效果，而是
**逐候选、逐环节、逐通道的可溯源量化审计账本**，以及跨小数位宽/量化档位的一致性判定。

## 目录结构

```
qfusion_audit/
├── qfusion/
│   ├── errors.py          # E_RECORD_INVALID / E_CONFIG_INVALID /
│   │                      # E_RECOMPUTE_BUDGET_PENDING / E_SNAPSHOT_STALE
│   ├── fixedpoint.py      # 定点量化原语、舍入、阶段常量、稳定排序键
│   ├── serialization.py   # 规范化 JSON 与 sha256 内容哈希
│   ├── config.py          # 融合配置解析与校验
│   ├── loader.py          # 候选流 JSONL 逐行解析与校验（失败不中断）
│   ├── normalization.py   # minmax / zscore 归一化与零分母固定规则
│   ├── engine.py          # 定点融合、特征叠加、裁剪、决断排序
│   ├── audit.py           # 跨档位账本、环节归因、翻转与边界对诊断
│   ├── scheduler.py       # 重算点预算、待办保留、线程池并发
│   ├── snapshot.py        # 临时写入 + 符号链接整体替换 + 失效判定
│   ├── pipeline.py        # 端到端主管线、增量指纹复用、审计报告
│   ├── sample_generator.py# 固定种子样本生成（不依赖网络/账号）
│   └── cli.py             # 命令行入口
├── tests/                 # unittest 套件（34 用例）
├── samples/               # 确定性样本、配置版本副本、快照目录、报告
├── run_demo.py            # 一键验收演示
├── DESIGN.md              # 技术说明与定点口径
└── VERIFICATION.md        # 实际验证结果记录
```

## 环境与运行

要求 Python 3.10+ 与 NumPy（仅样本生成用到）。

```bash
# 1. 生成固定种子样本（已随仓提供，可重复生成）
python3 -m qfusion.sample_generator

# 2. 运行完整审计并输出报告（默认预算取自配置）
python3 -m qfusion.cli --report samples/audit_report.json

# 3. 小预算演示待办续跑
python3 -m qfusion.cli --batch-budget 13 --report samples/audit_report.json

# 4. 一键验收（账本/翻转/预算续跑/快照恢复）
python3 run_demo.py

# 5. 测试
python3 -m unittest discover -s tests -v
```

## 定点口径（摘要，详见 DESIGN.md）

- 定点约定：整数 `q` 表示实值 `q * 2^-f`，`f` 为小数位宽。
- 量化档位 `gear`（正整数）在最小步长上再放大：真实步长 `step = gear * 2^-f`；
  `q = round_half_up(x/step)`，残差 `r = x - q*step`，舍入方向由残差符号给出。
- 不同位宽的整数不在同一尺度，审计统一换算到公共单位 `2^-F`（F 为最大位宽）
  后再比较，避免把尺度差异误判为量化误差。
- 跨档位名次比较同样在公共单位下用统一排序键重算。

## 输入数据契约

候选流每行字段：`request_id, channel, item_id, raw_score, features,
channel_rank, shard`。配置字段见 `samples/input/fusion_config.json`。
非法记录写入失败清单并继续加载，不抛出中断。
