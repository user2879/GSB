# 确定性样本

默认参数：

- 规模：48 个文档
- 固定种子：20261120
- 分段容量：5
- 集合：`orders`
- 主键：`id`
- 索引路径：`metrics.amount`
- 初始集合版本：7

文件：

- `collections.json`：含 `customer`、`metrics`、`attributes`、`tags` 的嵌套文档集合。
- `changes.json`：`incremental`、`stale`、`isolation` 三个变更批次。
- `plan.json`：容量、批次清单和完整命令行模板。

重新生成：

```powershell
python samples\generate.py --scale 48 --seed 20261120 --capacity 5 --output-dir samples
```

修改 `--scale`、`--seed`、`--capacity` 可生成不同整数规模；输出顺序、数值与批次内容由固定种子和参数决定。

批次说明：

- `incremental`：字段迁移、删除、插入混合的普通增量批次。
- `stale`：继续制造陈旧键历史。
- `isolation`：建议配合 `--retain-stale` 使用，触发查询时隔离窗口。

`SHA256SUMS` 记录默认参数下三个生成文件的摘要，可用于核验重新生成结果。
