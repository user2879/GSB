# 实际验证结果

验证日期：2026-11-18（Asia/Shanghai）

## 环境

- Python 3.11.0rc1
- NumPy 2.4.6
- scikit-learn 1.9.1
- PyYAML 5.4.1
- 本地合成数据：180 行、4 个潜在组、4 个数值列、2 个普通类别列、1 个 18 类高基数列、2 个冗余数值列，并带有确定性缺失值。

## 运行命令与退出码

```bash
python3.11 -m tcw --config config/experiment.yaml --no-injections \
  --snapshot-dir artifacts/snapshots-clean --report artifacts/runs/clean_report.json
# exit 0, status=ok

python3.11 -m tcw --config config/experiment.yaml --injections config/injections.yaml \
  --snapshot-dir artifacts/snapshots --report artifacts/runs/diagnostic_report.json
# exit 1, status=completed_with_failures

python3.11 -m tcw --config config/experiment.yaml --injections config/injections.yaml \
  --snapshot-dir artifacts/snapshots --report artifacts/runs/diagnostic_report_resume.json --resume
# exit 1, status=completed_with_failures, 三阶段均从快照恢复
```

## 候选排名

排名使用三项指标在候选集合内归一化后的加权 composite，权重为轮廓系数 0.40、CH 0.30、对数似然 0.30。

| 排名 | 方法 | 超参数 | composite | silhouette | CH | log-likelihood | 噪声 |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | KMeans | k=2 | 0.455523735 | 0.112024 | 26.595138 | -40.056126 | 0 |
| 2 | DBSCAN | eps=8.5, min_samples=5 | 0.430348802 | 0.268260 | 7.024240 | -36.250254 | 3 |
| 3 | KMeans | k=3 | 0.366393605 | 0.122031 | 17.776294 | -30.050421 | 0 |
| 4 | KMeans | k=4 | 0.357836617 | 0.103478 | 14.398694 | -4.115818 | 0 |
| 5 | GMM | n=4, diag | 0.317524510 | 0.062427 | 10.166517 | 28.496173 | 0 |
| 6 | GMM | n=4, full | 0.317323293 | 0.031706 | 7.906264 | 58.542687 | 0 |
| 7 | KMeans | k=5 | 0.292243725 | 0.082806 | 13.894191 | -12.213884 | 0 |
| 8 | DBSCAN | eps=7.0, min_samples=5 | 0.284094247 | 0.152804 | 7.915528 | -24.300584 | 3 |
| 9 | GMM | n=3, full | 0.252678105 | 0.030233 | 7.914797 | 36.753972 | 0 |
| 10 | GMM | n=3, diag | 0.227479666 | 0.038290 | 6.933092 | 28.444451 | 0 |
| 11 | DBSCAN | eps=1.0, min_samples=8 | 0.000000000 | 不可用 | 不可用 | -46.377918 | 180 |
| 12 | DBSCAN | eps=10.0, min_samples=5 | 0.000000000 | 不可用 | 不可用 | -46.377918 | 0 |

排名 11 为全噪声候选：`noise_count=180`、`all_noise=true`，没有被丢弃，仍出现在排名和指标表中。排名 12 只形成单簇，因此轮廓系数和 CH 不可用，同样被保留。单元测试另用重复点数据触发 KMeans 空簇，并验证空簇候选继续参与排名。

所有 12 个候选均执行两次独立训练：所有指标绝对差为 `0.0`，逐样本分配差异数为 `0`，GMM 责任度最大绝对差为 `0.0`。

## 注入命中

| 注入编号 | 故障 | 期望/实际业务码 | 命中 |
| --- | --- | --- | --- |
| `INJ-NONFINITE-001` | 样本 7 / `income` 非有限值 | `E_INPUT_NON_FINITE` | 是 |
| `INJ-MISSING-CATEGORY-002` | 样本 11 / `region` 未映射类别 | `E_CATEGORY_UNMAPPED` | 是 |
| `INJ-KMEANS-ZERO-CLUSTERS-003` | KMeans 簇数改为 0 | `E_HYPERPARAM_OUT_OF_RANGE` | 是 |
| `INJ-DBSCAN-NEGATIVE-RADIUS-004` | DBSCAN eps 改为 -0.5 | `E_DBSCAN_RADIUS_INVALID` | 是 |
| `INJ-GMM-COVARIANCE-005` | GMM 协方差类型改为 `banana` | `E_HYPERPARAM_OUT_OF_RANGE` | 是 |

重复执行同一注入配置得到的 `injection_hits` 与 `failure_list` 逐字段一致；5 个注入编号唯一。

## 快照与恢复

- 全局版本：1
- 配置哈希：`11e5883c30de0828faddadab13eca4113a5da372200a1c510b5b413aa223a2cc`
- 注入哈希：`c75a0a08f6f9eb5b660f988aaef07d7b1bbc19344f29b6525f79b1043b658af0`
- 数据哈希：`9ae4efd59bc62f99342f337c6bfd9df01637ea13681b6c0c4191bef4a23ada70`
- 预处理版本：`preprocess-32ef9108e5e21f97`
- 聚类版本：`clustering-702bce15ba77d35b`
- 评估版本：`evaluation-6b58adce4c215042`

恢复运行记录 `restored_stages=["preprocess", "clustering", "evaluation"]`。首跑与恢复报告的以下字段完全相同：候选排名、指标表、诊断、逐样本分配与责任度、注入命中、失败清单、配置哈希、注入哈希、快照版本和预处理产物。

另外验证了快照版本、配置哈希、注入哈希任一不符均抛出 `E_CONFIG_MISMATCH` 并拒绝续跑。

## 耗时与预算

最近一次注入实验的实际阶段耗时约为：数据 11.7 ms、预处理 105.5 ms、聚类 1769.3 ms、评估 397.7 ms、注入 0.7 ms，总计 2297.7 ms。配置预算为 3000 ms，预处理/聚类/评估均未超预算。耗时字段是观测值，不作为跨机器逐字段一致性的断言。

## 测试

```text
test_concurrency_does_not_change_results ... ok
test_empty_cluster_and_all_noise_candidates_are_retained ... ok
test_injection_hits_are_unique_and_repeatable ... ok
test_invalid_config_entries_fail_but_valid_remain ... ok
test_preprocessing_is_elementwise_repeatable ... ok
test_snapshot_resume_and_mismatch_rejection ... ok

Ran 6 tests
OK
```
