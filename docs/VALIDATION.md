# 实测验证结果

环境：WSL2 Ubuntu 22.04，Python 3.10.12（venv），依赖版本见 requirements.txt；
离屏 GL：Mesa 23.2.1 D3D12 后端，OpenGL 4.2 Compatibility，FBO 创建成功。

## 样本生成确定性

- 点数：120000；六面各 20000；两次生成 points/normals/manifest/camera/edits 逐字节一致。
- 包围盒与置信度分布见 data/manifest.json。

## 增量重建与一致性

| op | 类型 | 失效闭包 | 重算 | 复用 | 闭包=重算 | 顶点最大误差 | 法线最大夹角(rad) | 状态 |
| -- | -- | -- | -- | -- | -- | -- | -- | -- |
| 0 | cull_region | 25 | 25 | 575 | True | 0.0 | 3.332e-08 | match |
| 1 | reestimate_normals | 0 | 0 | 600 | True | 0.0 | 3.332e-08 | match |
| 2 | cull_region | 36 | 36 | 564 | True | 0.0 | 3.332e-08 | match |
| 3 | rebuild_params | 600 | 600 | 0 | True | 0.0 | 3.332e-08 | match |

## 帧时预算（预算 33.3 ms，确定性时钟）

- 总帧数 16，降级帧 10，密度档位 [0.25, 0.5, 1.0]，延迟刷新 4 帧，跳帧 4 帧，最大选用耗时 31.327 ms。
- 所有帧 chosen_cost_ms <= 预算；原始超预算帧 10。

## 账本与退出

- 正常退出报告 clean_exit=True，head_version=4，提交数 5。
- crash-demo：tmp-* 孤儿被回收，无退出报告时不伪造，恢复到最近已提交 head=4。

## 可复现哈希（两次运行逐字节相同）

报告哈希（`pcrecon hash-verify` 两次运行比对一致）：

- `reports/incremental_report.json`: `97fad3a2b4227adcf5dabb1769acc347da43d16650ac8ad0bdc2448129e87e9f`
- `reports/frame_report.json`: `350d1782f905e1beab69f90cf9d1006c005184cd17fb7c93e12b5de146eb2e11`

测试：`python -m unittest discover -s tests` → 17 个用例全部通过。
