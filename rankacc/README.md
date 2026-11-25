# rankacc — 推荐排序采样参数本地验收工具

逐请求候选打分流的可复核验收：一轮运行内完成通道内归一化、定点 TopK 截断、
分层抽样与确定性打散，结论写入可复核账本，并支持 v1 定宽行 / v2 JSON 双版本
往返、增量失效复用、逐级误差归因、预算调度与异常注入隔离。

环境：Python 3.11/3.12，仅依赖 numpy（第三方），其余为标准库。不使用容器。

## 目录结构

- `rankacc/core.py` — E_xxx 业务码、定点助手（Q8/Q12/Q16 枚举）、迷你 YAML 解析、
  候选 JSONL 与配置的加载校验层（失败入清单、其余请求继续）。
- `rankacc/pipeline.py` — 归一化（minmax/zscore，通道名升序 + 候选标识升序的固定
  累加顺序）、TopK 截断与半量子步长边界标记、分层配额（最大余数法，余数按通道名
  升序补齐）、辅助特征分组 + 哈希 PRNG 打散、高一位宽参考的逐级误差归因。
- `rankacc/formats.py` — v1 定宽行（38 字符：kind2|req12|序号4|候选12|千分比7|选择1）
  与 v2 JSON 的解析/序列化；旁路元数据（精度、边界、选择码映射、方案名）支撑的
  往返还原，降级条目入账。
- `rankacc/store.py` — 内容寻址缓存、追加式账本 JSONL、临时目录整体替换的原子提交。
- `rankacc/runner.py` — 预算调度、故障注入、请求/层级缓存失效、回执裁决、编辑流。
- `rankacc/samplegen.py` + `scripts/gen_samples.py` — 固定种子样本生成。
- `tests/test_acceptance.py` — 9 个验收测试（unittest，无第三方测试依赖）。

## 关键语义口径

- 定点：配置档位 frac 位；内部参考位宽为 frac+8。边界候选 = 与末位入选者参考值之差
  ≤ 半个量化步长（参考单位 2^7）的截断内候选；输出参考单位差值与档位来源。
- 打散种子 = SHA256(配置指纹 ‖ 请求标识 ‖ 通道 ‖ 分组键)；配置指纹只含结果相关参数
  （topk/比例/档位/口径/通道权重），线程数与预算等运行参数不参与，故并发与串行一致。
- 缓存键按阶段派生：norm(通道内容,权重,档位,口径) → trunc(norm键,topk) →
  layer(trunc键,通道,比例,配置指纹,请求标识,成员aux) → ser(层键,版本)。替换单候选仅使
  其通道归一化与下游失效；未受影响层命中持久化产物直接复用。
- 预算：按请求标识稳定顺序消耗单批预算；超单请求配额的请求以
  E_POINT_BUDGET_PENDING 入待办，本批中止、随后以独立批次优先处理（待办队列随快照
  持久化，崩溃后续跑优先）。
- 多版本回执：高版本覆盖低版本为权威条目，裁决依据与整数耗时毫秒写入账本。
- 强制终止：注入 crash 故障时本次运行不提交任何暂存产物，仅凭上次持久化快照恢复。

## 命令行用法

```
python3 scripts/gen_samples.py data                 # 生成确定性样本
python3 -m rankacc.cli run --data data --store store
python3 -m rankacc.cli run --data data --store store_f --faults data/faults.jsonl
python3 -m rankacc.cli apply-edits --store store --edits data/edits.jsonl
python3 -m rankacc.cli convert --store store        # v1/v2 旁路往返
python3 -m rankacc.cli verify --data data --work verify_work
python3 -m rankacc.cli report --store store
python3 -m unittest discover -s tests               # 验收测试
```

## 实际验证结果（本机，Python 3.11，numpy 1.26.4，16 核 / 约 6GB 可用）

- `unittest`：9/9 通过（边界决断、往返还原、归因分类、增量=全量、层级失效、
  并发=串行、预算待办、故障隔离、崩溃恢复）。
- 干净运行：12 请求全部出结论，5 批次，0 失败；req0007（30 候选 > 配额 25）记
  E_POINT_BUDGET_PENDING 后以独立批次完成。
- 异常注入运行：req0003→E_BAD_SCORE、req0005→E_UNKNOWN_CHANNEL 入失败清单，
  其余 10 请求正常出结论。
- 双版本往返：`convert` 输出 `roundtrip_restore_ok: true`，12 条降级条目入账；
  边界标记/差值与精确分数自旁路还原，非由千分比折叠值重推（测试断言折叠确实丢精度）。
- 增量 vs 全量：8 个编辑（追加/替换/权重/比例/上限/版本×2/档位）后
  `outputs_identical: true`、`cache_keys_identical: true`，增量路径复用 105 个阶段产物。
- 并发一致：threads=1 与 threads=4 的 outputs 逐字段相同（测试断言）。
- 崩溃恢复：注入 crash 后账本与上次提交逐字节一致，重跑结论与干净参考运行逐字段相同。
