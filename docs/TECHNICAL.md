# 技术说明：推荐特征管道多版本互操作应用

## 架构

```
scripts/gen_samples.py   固定种子(20261118)样本生成：data/sample.{v1.txt,v2.json,v3.bin}
config/step_name_map.json 步骤名 <-> v3 枚举整数映射表（可配置）
config/inject_config.json 异常注入配置（独立文件，注入项带编号）
app/formats.py           三版本解析器/序列化器（E_FIELD_WIDTH/E_PARTITION_OOB/E_STEP_UNKNOWN）
app/ir.py                统一中间表示：实体特征、缺失标记、显式空值、精度标签、有序步骤
app/converters.py        双向转换器 + 降级损失清单 + 旁路元数据（output/sidecar/）
app/steps.py             确定性步骤执行（10^6 定点整数域）
app/edits.py             五类编辑 + 版本语义约束（E_VERSION_UNSUPPORTED）
app/cache.py             内容指纹缓存（output/cache/）
app/snapshot.py          版本化快照：临时文件写盘 + 整体替换（output/snapshot/）
app/inject.py            异常注入（定宽行超长/未知枚举/精度标签缺失/边界相等/合并冲突）
app/report.py            互操作诊断报告（output/report.json）
app/runner.py            编排：解析-转换-编辑-增量执行-快照-报告
```

## 显式声明的规则

- 缺失传播：任何依赖缺失输入的步骤输出亦为缺失；步骤不为缺失条目造值；
  缺失标记沿步骤链传播，直到被“替换实体特征值”编辑显式覆盖。
- 运算顺序：实体 ID 升序、特征名升序；全部运算在 10^6 定点整数域。
- 分桶边界相等：归入较低分桶（bisect_left），记录 `bucket_boundary_equal` 决断。
- 交叉哈希冲突：同名交叉特征保留先出现的一对（keep_first），记录
  `cross_hash_collision` 决断。
- 实体合并冲突：target_wins，记录 `merge_id_conflict` 决断并输出 E_MERGE_CONFLICT。

## 缓存键派生（增量失效）

- parse_key = H(来源内容指纹 | 版本号)
- 第 i 步 key = H(第 i-1 步 key | 步骤参数指纹 | 版本号)
- 最终特征集 key = H(末步 key | "final" | 版本号)

任一上游变化自动使该步骤及全部下游键失效；复用/重算清单按步骤序位稳定排序。

## 快照协议

每次成功编辑或转换后：ir.json / sidecar.json / ledger.json 先写 .tmp 再整体替换，
最后提交 manifest.json（全局版本号、配置哈希、名称表哈希、IR 内容指纹、旁路哈希）。
启动时校验清单版本号与各文件内容指纹；任何不符输出 E_SNAPSHOT_STALE 并整体重建。

## 实际验证结果（2026-11-18 本机执行）

- `python -m unittest discover -s tests`：18 项测试全部通过，覆盖
  三版本解析与错误码、转换保真（千分比精确换算/残差旁路/枚举映射缺项）、
  确定性执行与边界决断、编辑拒绝（E_VERSION_UNSUPPORTED）、
  增量失效范围（改第 2 步参数 → 复用 [0] 重算 [1,2]）、
  v1/v2/v3 三目标版本增量 vs 全量逐字段一致、快照命中/损坏重建/版本落后、
  异常注入五项全部命中（INJ-001..INJ-005）。
- `./run.sh` 端到端演示：快照 stale_rebuild → v1/v3/v2 转换（降级损失 166 条）
  → 六类编辑（复用/重算清单符合预期）→ 增量 vs 全量重建一致 →
  output/report.json 含偏差直方图、最大偏差特征、降级损失、复用/重算与失败清单。
