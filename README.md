# fppdiag — 定点特征管线漂移诊断工具

面向 Windows 本地环境的训练数据流式特征管线诊断工具。管线全程使用**整数定点
运算**，追踪量化/分桶/交互误差沿阶段链的传播上界，并据此对观测窗口相对参考
窗口的漂移做“确超 / 确未超 / 误差不可区分”三态裁决。纯 Python 3.11 标准库实现，
无容器、无网络、样本由整数种子确定性生成。

## 目录
- `fppdiag/` 核心包
  - `fixedpoint.py` 显式整数定点原语（half-away 舍入、int32 饱和、整数对数）
  - `spec.py` 规格/数据源/窗口强校验（E_SPEC，定位首个违规对象与字段）
  - `sources.py` 确定性 LCG 样本 + 摘要保护的列缓存（损坏→E_DATA）
  - `stages.py` 标准化/分桶/对数/交互乘积/截断五阶段整数变换
  - `engine.py` 分块流式执行（两种调度，逐字节一致）
  - `errorchain.py` 误差上界传播、贡献占比与主导段（阈值 40%）
  - `stats.py` 定点统计、漂移分量与稳定三态裁决
  - `ledger.py` SQLite 账本 + 版本化快照（批次事务、内容摘要）
  - `orchestrator.py` 失效传播、增量/全量比对、崩溃恢复、重放
  - `cli.py` 命令行与三类异常注入
- `tests/` 52 个 unittest 用例
- `samples/pipeline.json` 本地样本管线（`init-sample` 可再生成）
- `docs/DESIGN.md` 技术设计与全部显式口径
- `docs/VERIFICATION.md` 实测验证记录

## 快速开始
```
pip install -r requirements.txt
python run_demo.py init-sample --outdir samples
python run_demo.py validate --spec samples/pipeline.json
python run_demo.py run      --spec samples/pipeline.json --workdir work/demo
python run_demo.py verify   --spec samples/pipeline.json --workdir work/demo
python -m unittest discover -s tests
```

## 业务码
E_SPEC(10) 规格非法 · E_DATA(11) 数据源损坏 · E_PARAM(12) 参数越界 ·
E_DIVERGE(13) 增量/全量不一致 · E_REPLAY(14) 恢复重放不一致 ·
E_SNAPSHOT(15) 快照摘要/版本不符。

详见 `docs/DESIGN.md` 与 `docs/VERIFICATION.md`，原始任务书见
`docs/REQUIREMENTS.md`。
