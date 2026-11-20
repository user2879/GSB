# 架构与一致性设计

## 模块

```text
config.py       配置校验、规范化、阶段参数、配置哈希输入
formats.py      raw/v1/v2 解析与编码
conversion.py   双向转换、半入取整、残差与旁路
features.py     归一化、子词切分、固定哈希、桶计数
scoring.py      固定顺序成对打分与边界判别
strategy.py     pilot 指标、候选评分、档位决策
cache.py        逐阶段缓存结构、桶引用、失效摘要
snapshot.py     完整代目录、manifest 原子替换、恢复校验
injections.py   注入 schema、业务码映射、命中报告
engine.py       编排、线程池、增量/全量、报告
verify.py       编辑序列、并发、恢复、互转回归
cli.py          命令行入口
sample.py       固定种子受控词表样本生成
```

## 执行阶段

1. 读取配置和三类输入，解析失败独立记录。
2. 依据 `source_version` 合并原文与交换记录；缺失或非法项进入失败清单。
3. 在基础桶空间上跑 compact pilot，得到规模指标并选择档位。
4. 特征阶段：逐对查询 `features` 内容键；未命中才进入线程池。
5. 打分阶段：依赖特征内容指纹、权重、阈值、档位参数和来源版本。
6. 转换阶段：先由 v2 生成 v1 与旁路，再由旁路驱动 v1 -> v2 恢复。
7. 评估阶段：依赖全部最终记录、失败、桶引用和指标参数。
8. 构建桶引用、失效摘要、边界清单与报告核心指纹。
9. 将四阶段产物写入临时代目录，完整后改名；最后原子替换 manifest。

## 内容键

所有键都是 SHA-256 规范 JSON 指纹，输入对象按键排序、浮点禁止 NaN/Infinity。实现 schema 号参与键，避免代码语义变化后误用旧产物。

- 特征键：原文、切分版本、精度档、归一化、桶数、特征维度、来源版本、注入指纹。
- 打分键：特征内容、权重、阈值、有效桶数、精度档、来源版本、注入指纹。
- 转换键：交换记录、方向、标签枚举、该对旁路条目指纹、注入指纹。
- 指标键：最终记录、失败、桶引用、指标参数。

未失效缓存值会直接反序列化复用，不重新执行阶段函数。失败任务不缓存成功产物；注入键隔离注入与普通运行。

## 失效传播

```text
原文/交换内容变化
  -> 受影响 pair 的 feature key 变化
  -> 该 pair 的 scoring key 变化
  -> metrics key 变化

权重/阈值变化
  -> feature key 不变，特征复用
  -> scoring key 变化
  -> metrics key 变化

档位/来源版本变化
  -> 有效桶数、精度或来源版本进入上游键
  -> 相关特征、打分、指标失效

单对替换/删除
  -> 仅该对的逐对产物失效
  -> 其他对直接复用
```

桶引用由最终特征向量导出：`bucket -> sorted(pair_ids)`。删除/替换对时，从旧引用与新引用对称差导出受影响桶。

## 增量与全量一致

全量运行使用空缓存，增量运行使用上代缓存。两者共享同一纯函数阶段实现和稳定排序。缓存只决定“是否执行”，不改变计算结果。因此任何编辑序列后的最终产物必须等于对同一最终输入、配置和注入从空白状态全量执行的结果。

`verify.py` 在每个编辑步骤后复制最终输入到独立目录，用全新 state/output 全量运行，并比较 `deterministic_core_fingerprint`。

## 快照协议

代目录：

```text
state/generations/gen-<8位全局版本>-<12位配置哈希>/
  features.json
  scoring.json
  conversion.json
  cache.json
  COMPLETE
state/manifest.json
```

提交顺序：

1. 在同一文件系统创建 `.tmp-snapshot-*`。
2. 写四个阶段 JSON、fsync，并写 `COMPLETE`（含各文件指纹）。
3. 目录原子改名为最终代目录。
4. 写 manifest 临时文件并 `os.replace` 到 `manifest.json`。

恢复顺序：

1. 只读取 manifest 引用的代。
2. 要求 `COMPLETE` 与四个阶段文件都存在。
3. resume 时校验配置哈希、可选全局版本、可选来源版本。
4. 不读取 `.tmp-*`、未被 manifest 引用的代或只写了一半的目录。

强杀测试在写入 `.tmp-crash-*` 后抛出 `SimulatedTermination`，不更新 manifest。后续 `resume=True` 只能恢复此前的完整代，并在报告输出 `recovery_point`。

## 并发确定性

任务排序键为 `sha256(pair_id + ":" + stage)`。虽然线程完成顺序不确定，但结果按 `pair_id` 汇总；所有 dict 输出按键排序，所有失败按业务码、阶段、pair、来源、行号、字段、消息稳定排序。聚合使用固定顺序的 `math.fsum`。

## 报告指纹

`deterministic_core_fingerprint` 覆盖：

- 策略决策；
- 统一记录和边界清单；
- 特征桶汇总与桶引用；
- 失败清单；
- 双版本转换记录、损失、旁路与交叉比较；
- 指标；
- 注入报告。

不覆盖缓存计数、快照版本、恢复点和输出目录，因而可跨运行历史比较结果一致性。过程审计信息仍保留在报告其他字段。
