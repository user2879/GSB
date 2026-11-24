# opdiag — 本地确定性张量 激活/转置重排 算子链诊断组件

面向本地确定性张量的算子链（激活节点 ReLU/GELU/Sigmoid/Tanh + 转置/维度
置换节点）诊断组件。纯 Python + NumPy 2.x，不依赖网络、数据库或容器。

## 1. 环境准备

```bash
python3 --version            # 需要 3.11+
pip install -r requirements.txt   # 安装 numpy 2.x
```

## 2. 一键可重复演示

```bash
python -m opdiag.demo
```

该命令会按固定种子在 `demo/sample/` 生成算子链、输入张量描述、执行配置与
异常注入配置，然后：

1. 在小重算预算下多批执行，验证待办保留、快照恢复与预算扣减序列；
2. 施加参数修改、节点顺序调整、子图替换、精度档位切换、原地写回标记变更、
   定点位宽/量化档位切换等编辑序列，做增量运行；
3. 在独立空白目录按最终配置一次性全量重建，逐字段比对；
4. 独立注入 4 类真实故障，验证命中、业务码映射与故障隔离。

结果写入 `demo/demo_summary.json`，`consistent=true` 即逐字段一致。

也可分步手工执行：

```bash
python -m opdiag.samplegen --out demo/sample      # 生成确定性样本
python -m opdiag --config demo/sample/run_config_healthy.json   # 健康链
python -m opdiag --config demo/sample/run_config.json           # 带注入
# 小预算下重复执行同一配置即可观察“待办 -> 续跑 -> 全部复用”
```

## 3. 目录结构

```
opdiag/
  errors.py       业务错误码 E_xxx 与失败清单条目
  dmath.py        确定性 exp/sqrt 近似与四种激活（固定运算顺序、舍入计数）
  fixedpoint.py   定点量化（位宽 8/16/32，uniform/pow2/per_element）
  hashing.py      规范化 JSON 哈希、张量内容指纹
  config.py       执行配置枚举校验与默认值
  graph.py        加载、拓扑校验、形状推导、原地冲突、融合候选识别
  ops.py          转置/置换、融合双路径、原地写回
  cache.py        节点缓存键派生
  storage.py      阶段化快照（暂存目录 + 整体替换 + 原子清单）
  engine.py       调度/预算/并发/注入/失效传播/快照/报告 主引擎
  verify.py       增量 vs 全量重建 逐字段一致性验证
  samplegen.py    固定种子样本生成
  demo.py         端到端演示
  cli.py/__main__.py  命令行入口 python -m opdiag --config <path>
tests/            35 个 unittest 用例
demo/             生成的样本、快照输出与演示汇总
```

## 4. 关键设计与业务要求映射

- **拓扑校验（要求1）**：节点 ID 唯一、边端点、无环（Kahn 稳定拓扑）、
  单生产者、激活参数合法性、置换为 0 起始完整排列且秩一致、定点枚举校验。
  每个问题产出独立 `E_xxx`，节点失败不阻断其余合法节点。
- **确定性激活与定点（要求2）**：`dmath.approx_exp` 用范围归约 + 固定阶
  Taylor（Horner 求值），`approx_sqrt` 用固定次数 Newton 迭代；近似形式、
  阶数、迭代次数写入每个产物 `meta.approximations`。定点量化逐元素记录
  步长 `step` 与残差 `residual`，fp32/fp64 与不同位宽/档位产出不同整数
  与反量化值。
- **转置与原地写回（要求3）**：输出索引按字典序升序（C order）遍历并记录
  `order_ids`。fan-out 共享引用或转置变形触发 `E_INPLACE_CONFLICT`；安全
  原地写回在私有缓冲上就地更新，不污染外部输入或其他节点引用。
- **融合策略（要求4）**：激活直连转置识别为融合候选，分别跑非融合
  （先激活后转置）与融合（单节点按输出字典序发射，GELU 采用替代结合顺序）
  两条路径，各自记录累积顺序标识与逐元素中间舍入次数；逐元素差值超容差时
  产出 `E_FUSION_TOLERANCE`，列出超差元素并归因到 `fused_path`，不静默
  选择任一结果。
- **内容指纹失效传播（要求5）**：缓存键 = 实现版本号 + 算子参数 + 置换 +
  原地标记 + 输入张量内容指纹 + 精度/位宽/量化档位。编辑后仅受影响节点及
  其下游失效重算，未受影响节点直接复用且不重算；复用/重算清单按拓扑序位次
  与节点 ID 稳定排列。
- **重算预算调度（要求6）**：预算按节点输出元素数、严格按拓扑序串行扣减，
  与线程调度无关；超预算节点进待办并标记 `E_RECOMPUTE_BUDGET_PENDING`，
  下一批优先处理待办。
- **逐字段一致性（要求7）**：`verify.run_consistency_probe` 对编辑序列的
  增量结果与空白全量重建比较输出/整数/残差/步长/反量化/顺序标识/舍入次数/
  融合差值/失败清单顺序。
- **快照、并发与注入（要求8）**：线程池仅在同层互不依赖节点间并发，不超过
  `max_workers`，任务带 `topo|node|stage` 稳定排序键；产物先写暂存目录再
  整体替换，清单含全局版本号、配置哈希、已完成节点与各产物版本。注入配置
  独立于主配置，支持非有限元素、重复置换、激活参数越界、实现版本缺失，
  注入项编号回写诊断报告并映射业务码。

## 5. 业务错误码

| 代码 | 含义 |
| --- | --- |
| E_NODE_DUPLICATE | 节点 ID 重复 |
| E_EDGE_ENDPOINT | 边端点不存在 |
| E_GRAPH_CYCLE | 存在有向环 |
| E_MULTIPLE_PRODUCER | 输入张量存在多个生产者 |
| E_INPUT_UNBOUND | 节点缺少上游且无外部输入引用 |
| E_ACTIVATION_PARAM | 激活参数非法/越界 |
| E_PERM_INVALID | 置换不是与秩一致的 0 起始完整排列 |
| E_ENUM_INVALID | 精度/位宽/量化档位等枚举非法 |
| E_UNKNOWN_OP | 未知算子类型 |
| E_INPLACE_CONFLICT | 原地写回会破坏其他节点输入引用或形状改变 |
| E_UPSTREAM_FAILED | 上游失败导致的级联跳过 |
| E_FUSION_TOLERANCE | 融合/非融合路径差值超容差 |
| E_RECOMPUTE_BUDGET_PENDING | 超节点级重算预算，保留待办 |
| E_INJ_NONFINITE / E_INJ_DUPLICATE_PERM / E_INJ_ACTIVATION_PARAM / E_INJ_IMPL_MISSING | 异常注入命中码 |

## 6. 运行测试

```bash
python -m unittest discover -s tests -v
```

## 7. 备注

- 本机预装的其他第三方包可能固定依赖 numpy 1.x；本组件要求 numpy 2.x。
  升级 numpy 不影响 opdiag 自身运行，如需隔离可使用独立虚拟环境。
- 所有随机样本由 `numpy.random.default_rng(固定种子)` 生成，重复运行逐元素
  一致；报告与快照默认写入工作区 `output/`（或配置指定目录）。
