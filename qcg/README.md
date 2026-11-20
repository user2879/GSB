# QCG — 低精度量化卷积算子图编辑与调度应用

面向 v1（扁平定宽键值行）与 v2（JSON 命名节点/边）两种交换版本的量化卷积
算子图，提供双版本解析互转、整数域定点卷积、确定性反量化、内容指纹失效
传播、三重预算约束并行调度与阶段快照恢复。全程纯本地、无网络、无 GPU。

## 目录结构

```
qcg/
  qcg/                 应用包
    errors.py          业务错误码与故障记录
    models.py          统一中间表示 Node/Graph、稳定拓扑序
    config.py          YAML 调度配置与配置哈希
    v1_codec.py        v1 定宽 24 字符行解析/序列化
    v2_codec.py        v2 JSON 解析/序列化、浮点缩放与精度标签
    converter.py       双向互转、旁路元数据、降级损失清单
    quant_pack.py      小端二进制量化包读写、固定种子样本
    kernel.py          整数定点卷积（SATURATE/WRAP/ERROR 溢出策略）
    dequant.py         反量化归约与五种确定性舍入、边界样本标记
    fingerprint.py     参数/张量指纹、缓存键、上游集合、下游闭包
    editor.py          位宽/缩放/舍入编辑、节点增删、子图替换
    product_store.py   产物持久化、块驻留、LRU 淘汰与重算审计
    snapshot.py        四阶段临时目录+原子替换、续跑身份判定
    injection.py       独立注入配置与四类故障
    engine.py          确定性规划器 + 线程池执行器 + 编排
    cli.py             命令行入口（convert/edit/run）
  tests/               pytest 测试套件（46 用例）
  scripts/
    generate_samples.py  固定种子逐字节可复现样本生成
    verify.py             端到端验证并写 reports/verification.json
data/
  graphs_v1/ graphs_v2/ quant_pack/ sidecar/ injection/ snapshots/
reports/               运行报告与验证证据
```

## 快速开始

```bash
pip install numpy pyyaml pytest
cd qcg
python3 scripts/generate_samples.py     # 重新生成样本（逐字节一致）
python3 -m pytest -q                    # 46 个测试
python3 scripts/verify.py               # 端到端验证
```

### CLI

```bash
# v2 降 v1，不可表达信息写入旁路元数据
python3 -m qcg.cli convert --input ../data/graphs_v2/sample_graph.v2.json \
  --output ../data/graphs_v1/x.v1.txt --target v1 \
  --sidecar ../data/sidecar/x.sidecar.json
# v1 升回 v2，依据旁路精确还原舍入模式与写回标记
python3 -m qcg.cli convert --input ../data/graphs_v1/x.v1.txt \
  --output ../data/graphs_v2/x.v2.json --target v2 \
  --sidecar ../data/sidecar/x.sidecar.json
# 编辑与运行（重复运行自动按缓存键失效，仅重算受影响下游）
python3 -m qcg.cli edit bitwidth --graph ../data/graphs_v2/x.v2.json \
  --node n0002 --value 16
python3 -m qcg.cli run --graph ../data/graphs_v2/x.v2.json \
  --config ../data/scheduler_config.yaml \
  --pack ../data/quant_pack/sample_pack.le.bin \
  --snapshot-root ../data/snapshots/demo --report ../reports/demo.json
```

## 关键设计

### 确定性的核心：规划与执行分离
`Planner` 单线程按稳定拓扑序（位次 `%04d:节点ID` 排序键）一次性决定每个
节点的复用/计算、预算扣减顺序、LRU 淘汰与待算队列；`Engine` 仅把已规划
任务放入线程池，任务无副作用、上游就绪才入波次，结果再按拓扑序提交。
因此线程调度时序、并发度、预算耗尽时机都无法影响最终账本，串行与并发
逐字段一致（见 `test_serial_and_concurrent_identical`、
`test_diamond_graph_parallel_wave`、`test_deterministic_budget_deduction_order`）。

### 降级旁路
v1 无法表达舍入模式、原地写回与任意浮点缩放。降级时把原值、千分比截断
残差、精度标签、位宽枚举、写回标记、边界样本全部写入 sidecar，并对每项
损失输出 `E_LOSSY_DOWNGRADE`；升回时按 sidecar 精确还原，无法还原项进入
损失清单。

### 失效传播
缓存键 = 量化参数指纹（含算子实现版本）+ 上游张量内容指纹 + 舍入模式 +
写回标记。编辑后以参数指纹变化为种子，沿下游闭包传播；未失效节点直接
复用持久化产物，复用/重算清单写入诊断报告。删除节点的陈旧产物会被
`retain_only` 清理，保证增量结果与空白全量重建逐字段一致。

### 三重预算
按节点配额 → 全局预算 → 单批复算预算的固定顺序判定与扣减；超配额分别记
`E_NODE_BUDGET_EXHAUSTED` / `E_GLOBAL_BUDGET_PENDING` /
`E_RECOMPUTE_PENDING`，下游因 `upstream_pending` 保留在待算队列，下一轮
（预算刷新）按同一稳定顺序续跑直至收敛。

### 阶段快照与注入隔离
加载校验、整数卷积、反量化归约、汇总四阶段均先写 `staging/*.tmp` 再
`os.replace` 原子提交；清单记录全局版本号、图指纹、配置哈希、版本来源、
旁路哈希、已完成节点与产物版本。启动续跑比对三项身份，不一致输出
`E_CONFIG_MISMATCH` 拒绝续跑。注入按节点+阶段命中，编号写入报告，单节点
故障隔离不影响其余节点。
