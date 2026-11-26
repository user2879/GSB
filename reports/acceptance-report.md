# 多视图重建原型验收报告

- 总结果：通过
- Python：3.11.0rc1 (Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.35)
- NumPy/Pillow：1.26.4 / 10.3.0
- Tkinter：available
- 主运行规范指纹：`d15bce9354d1aa369f90a2fb44ae6b223af7b1f0c64725eb66a72c61f5b2d71d`

## 增量与全量一致性

- 结果：通过
- 增量指纹：`f1ad562cb5444cb10811d90f33d75e3105a5fe1e9959757a50e17afd05015262`
- 全量指纹：`f1ad562cb5444cb10811d90f33d75e3105a5fe1e9959757a50e17afd05015262`
- 命令数：12；输入错误码：3
- 规范化账本条目：293

## 精确失效传播

- 结果：通过
- `unrelated_feature_v0_hit`：True
- `replaced_feature_v1_rebuilt`：True
- `affected_pair_v0_v1_not_hit`：True
- `unrelated_pair_v2_v3_hit`：True

## 并发与预算

- 不同线程数指纹一致：True
- `point_budget`：passed=True，pending=32
- `voxel_subdivision_budget`：passed=True，pending=1
- `resume_priority`：passed=True，pending=34

## 恢复判定

- 结果：通过
- 健康快照：`resume_with_snapshot`；残留临时目录数：1
- 损坏当前指针：`rebuild` / `E_SNAPSHOT_CORRUPT`

## 误差归因

- 默认档位统计：`{'within_tolerance': 92, 'voxel_subdivision': 2, 'decision_flip': 4, 'flip_source:voxel_subdivision': 2, 'depth_consistency': 2, 'isolated_exceeded:depth_consistency': 2, 'isolated_exceeded:voxel_subdivision': 2, 'flip_source:depth_consistency': 2}`
- 边界探针触发：True
- 探针体素：`v0_0_1`；阈值：1925
- 翻转归因统计：`{'within_tolerance': 65, 'depth_consistency': 4, 'isolated_exceeded:depth_consistency': 12, 'decision_flip': 5, 'flip_source:depth_consistency': 3, 'feature_quantization': 11, 'isolated_exceeded:feature_quantization': 11, 'isolated_exceeded:triangulation_residual': 9, 'flip_source:feature_quantization': 2, 'isolated_exceeded:voxel_subdivision': 2}`
