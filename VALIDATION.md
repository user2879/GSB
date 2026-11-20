# 实际验证记录

验证日期：2026-11-20  
环境：Python 3.11.0rc1、numpy 2.4.6、PyYAML 5.4.1、Linux 本地文件系统  
数据：`samples/` 中固定种子合成表，初始 84 行、最终 100 行，含数值列、类别列、缺失值、冗余列和二分类目标。

## 自动测试

执行：

```bash
python3 -m unittest discover -s tests -v
```

结果：多次执行均通过；最终记录为 `Ran 13 tests in 17.672s`，`OK`。

覆盖项：

- 八类独立业务码与失败报告；
- 分箱重复运行一致、分箱数变化真实改变边界/索引/内容指纹；
- 缓存失效矩阵：扩树、只调学习率、深度调整、分箱数变更、单行修正；
- 集成与全量逐字段等价、并发 1/4 一致；
- 单树分裂异常隔离、失败树不进入集成/贡献；
- 贡献总额与 `raw_score - base_score` 对齐；
- `gradient_boosting` 与 `random_forest` 两种模型的并发确定性；
- SIGTERM 返回 0、恢复后复用已完成树；
- SIGKILL 后只加载完整快照并成功续训。
- 提交前使用过期期望版本会返回 `SNAPSHOT_VERSION_CHANGED`。

## 增量/全量等价性

执行：

```bash
python3 -m tree_workbench verify \
  --config samples/experiment.yaml \
  --data-spec samples/data_spec.json \
  --edits samples/edit_plan.json \
  --work-dir artifacts/final-validation \
  --parallel-workers 4
```

结果：

```json
{"all_equal": true, "scenario_count": 5, "concurrency_equal": true}
```

五个场景（`initial`、`append-and-grow`、`deeper-and-slower`、`rebucket-and-correct`、
`final-append-and-grow`）的差异数均为 0。比较范围是整个分箱产物、全部树结构/指纹、
逐轮损失、集成预测/指标、全部样本贡献、有效配置（并发数作为运行控制除外）、数据摘要和失败清单。

最终场景：

- 有效配置：14 棵树、深度 5、学习率 0.12、分箱数 7、并发 4；
- 训练/验证行数：75 / 25；
- 验证准确率：0.68；
- 验证对数损失：0.5041890752；
- 失败清单：空；
- 最终快照版本：22，已完成树：`[0..13]`；
- 并发 1 与并发 4 的完整计算产物差异数：0。

完整结果：`artifacts/verification.json`。

## 恢复验证

- SIGTERM：子进程训练中发送 SIGTERM，完成在途树、提交快照后整数退出码为 0；重启报告
  `recovery.loaded.completed_trees` 非空，已完成树直接复用。
- SIGKILL：首批树提交后强制终止进程；重启只加载 `manifest.complete = true` 的完整快照并完成剩余树。
- 未完成临时快照：在快照目录放入无效 `.tmp-snapshot-*` 后，加载器仍返回最后一个完整版本。
