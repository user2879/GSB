这些异常文件由 scripts/make_demo.py 生成。

- missing_parameter_version.json: 直接解析触发 E_VERSION
- zero_buckets.json: 直接解析触发 E_BUCKET
- sample_size_exceeded.json: 完成过滤后触发 E_SAMPLE；上限为过滤后较小表的可用行数
- invalid_spec.json: 多个声明错误聚合为 E_SPEC
- corrupted_sample.npz: 截断后的 NPZ；配合样本清单摘要触发 E_SAMPLE

推荐命令：
python -m gsb run --spec examples/anomalies/missing_parameter_version.json --workdir var/anomalies/version
python -m gsb run --spec examples/anomalies/zero_buckets.json --workdir var/anomalies/bucket
python -m gsb run --spec examples/anomalies/sample_size_exceeded.json --workdir var/anomalies/sample
python -m gsb run --spec examples/anomalies/invalid_spec.json --workdir var/anomalies/spec
python -m gsb run --spec examples/demo_skewed.json --workdir var/anomalies/corrupt --inject-sample-corruption
