# 定点整数嵌入/聚合算子链 — 技术说明

## 1. 目标与范围

在纯本地（无 GPU、无网络模型、无容器）环境下，实现一条 LLM 推理前的嵌入与聚合
算子链，全部数值在定点整数域完成，并支持：

- e1（定宽文本行）与 e2（JSON）双版本交换及双向转换、降级旁路；
- 基于内容指纹的算子级/令牌级精确失效与复用账本；
- 单批令牌点数预算、单算子配额约束下的线程池并发调度与待办保留；
- 阶段化快照的临时目录构建 + 原子整体替换、配置哈希恢复判定；
- 任意编辑序列后的增量结果与从空白全量重建逐字段一致。

## 2. 目录结构

```
fxpipe/
  config.py      枚举、QuantSpec/OpSpec/Chain/Config、schema 校验
  errors.py      业务码 Code、非致命 Issue/IssueLog
  exchange.py    e1 定宽解析/编码、e2 解析、双向转换与 bypass
  kernel.py      定点上下文与 embed/pool/attn/layer_norm 整数内核
  fingerprint.py 规范化 JSON 与 sha256 内容指纹/缓存键
  ledger.py      缓存条目、失效/复用事件、账本持久化
  engine.py      拓扑计划、各阶段产物、reconcile 精确失效
  scheduler.py   稳定顺序预算准入 + ThreadPoolExecutor 受控并发
  snapshot.py    临时目录、原子替换、manifest、配置哈希恢复
  state.py       编辑命令应用，输出受影响令牌/算子集合
  loader.py      图/令牌/段/编辑流加载与逐条目校验跳过
  orchestrator.py 端到端编排、批次循环、诊断报告
  samplegen.py   确定性样本与含坏条目的样本
  demo.py        端到端验收脚本
tests/           unittest 测试套件
main.py          CLI 入口
```

## 3. 定点口径

- 小数位宽 `frac_bits ∈ {8,16,24}`，量化档 `gear ∈ {g0..g3}`，
  档位以整数分子/分母缩放（如 g1=5/4、g2=3/2、g3=7/4），g0 为恒等。
- 整数容器位宽 `width ∈ {8,16,32}`，映射命名方案 qint8/qint16/qint32。
- 所有乘法为定点乘法 `qmul`，带依赖 frac_bits 的舍入项并右移 frac_bits，
  再按位宽饱和（qint8 饱和到 [-128,127] 等）。
- 嵌入查表显式叠加 `unit//16` 的小数 LSB，保证切换 frac_bits/gear 时
  中间整数、逐令牌输出与残差真实变化（见 `test_kernel.py`）。
- 注意力权重为确定性整数千分比，最大余数法保证每个 query 权重和恰为 1000。
- 层归一化以整数 `isqrt` 计算方差平方根，按行固定升序累加均值/方差。
- 全程使用 Python 任意精度整数，结果与宿主机字长、线程数无关。

## 4. e1 / e2 交换

- e1 记录固定 28 字符：chain_id(6)+op_seq(4)+permille(10)+width(8)，
  左补零；宽度不符或字段含非法字符报 `E_FIELD_TRUNCATED` 并跳过该条。
- 显式缺失以 `NA` 表示，生成 `missing:<field>` 完整性标签，区别于非法字符。
- e1→e2：permille/1000 精确换浮点，width 按码表映射命名方案，精度标签 `e1-exact`。
- e2→e1：浮点四舍五入还原 permille，残差、精度标签、被覆盖引用写入 bypass；
  再升 e2 时凭 bypass 还原原精度标签与残差（`restore_e2_from_bypass`）。
- 两版本等价量化口径在内部 `QuantSpec` 上 `equivalent()` 为真，
  来源版本分别保留为 e1/e2。

## 5. 内容指纹与精确失效

- 缓存键由「上游缓存键（排序）+ 算子参数指纹 + 档位 + frac_bits + 交换版本号」
  经 sha256 派生；每个条目记录引用令牌集与引用算子集。
- `Engine.reconcile` 在编辑后对每个已存条目重算「当前状态下应有的缓存键」，
  与已存键比对；并结合受影响算子/令牌集合给出逐条失效依据：
  `op-edit / token-edit / quant-or-version-changed / content-fingerprint-mismatch`。
- 因此即使编辑流把档位/版本最终切回基线（终态配置相同），只要算子参数或
  令牌内容变化，相关条目仍被精确失效，未受影响条目（如另一条链）直接复用。

## 6. 调度、预算与配额

- 计划按算子依赖的最长路径拓扑层级排序，再按 chain_id/op_seq/段起点稳定排序，
  保证上游先于下游。
- 调度器在单线程上按稳定顺序做点数准入（记账与并发无关），仅把已准入的
  纯计算任务交给 `ThreadPoolExecutor`，并发数取 `max_workers` 且不超过。
- 单算子累计点数超 `operator_quota`：中止该算子本批并记
  `E_OPERATOR_QUOTA_EXCEEDED`；预算不足：任务保留待办并记
  `E_POINT_BUDGET_PENDING`。就绪回调确保上游未提交的任务不会在同批误执行。
- 多批次循环在预算补充后续跑，直到待办清空。

## 7. 快照与恢复

- 每阶段产物、账本、待办、已耗预算与 manifest 先写入 `tmp/v<n>.staging`，
  再 `os.replace` 整体替换，最后原子切换 `current` 符号链接。
- manifest 记录全局版本号、配置哈希、交换版本、已完成算子、各阶段产物版本、
  已耗预算、配额账本与待办。
- 启动以「配置哈希 + 交换版本」判定续跑或重建；版本号基于磁盘最大版本递增，
  避免目录碰撞。强制终止只可能留下未引用的 staging，恢复仅依赖已提交快照。

## 8. 验证结果

- 测试：`python -m unittest discover -s tests`，共 33 个用例全部通过，
  覆盖双版本互操作、定点确定性、档位/位宽敏感性、增量==空白重建、
  串行==并发、快照恢复、配置哈希重建、预算/配额/校验业务码。
- 端到端：`python main.py acceptance workspace` 输出
  增量失效/复用计数、增量与全量重建一致性、e1/e2 数值相同而缓存键按版本区分，
  机器可读结果写入 `workspace/run/acceptance.json`。
