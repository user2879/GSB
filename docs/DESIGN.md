# 设计与格式说明

## 1. 模块划分

| 模块 | 职责 |
| --- | --- |
| `peft_lora.base` | 基座二进制生成/加载、逐层和整包指纹 |
| `peft_lora.config` | YAML 子集解析、配置校验、语义哈希 |
| `peft_lora.tasks` / `samples` | 任务与 JSONL 加载、稳定排序键、确定性特征/标签 |
| `peft_lora.training` | 仅旁路低秩矩阵的固定顺序 SGD、产物校验/导出 |
| `peft_lora.compatibility` | 整包与目标层指纹兼容判定 |
| `peft_lora.merge` | 固定顺序叠加、定点量化、可逆检查、预算账本 |
| `peft_lora.snapshot` | 阶段版本目录、原子清单切换、恢复与产物发现 |
| `peft_lora.pipeline` | 加载→训练→合并→导出的完整编排 |
| `peft_lora.cli` | 命令行入口、信号安装、机器可读摘要 |

## 2. 基座格式与指纹

每个参数文件是 `<f8`、C order 的原始数组。`manifest.json` 的 `layers` 顺序不是语义顺序；指纹计算按层名排序。

单层指纹：

```text
sha256(algorithm, name, weight.shape, sha256(weight_bytes), bias.shape, sha256(bias_bytes))
```

整包指纹：

```text
sha256("base-fp-v1", sorted[(layer_name, layer_fingerprint)])
```

因此基座层改名、形状变化、权重/偏置任一字节变化或指纹算法版本变化都会改变指纹。v2 只改动 `layer_1`，所以 `layer_0`/`layer_2` 的层指纹保持不变。

## 3. 确定性训练

目标层的前向为：

```text
y = W x + b + scale · B A x
```

只训练 `A` 和 `B`，`W`、`b` 冻结。样本按 JSONL 文件顺序组成固定小批量；每个 epoch 按固定顺序遍历，使用固定学习率。损失轨迹记录每个 epoch 更新前的均方误差。

初始化种子只由 `task_name`、目标层和形状派生，不包含整包基座指纹。这样，当其他层升级而目标层权重未变时，复用适配器与在新基座空白工作区全量训练得到的权重和损失轨迹逐元素一致。

适配器 JSON 包含 `A`、`B`、任务名、目标层、秩、缩放、学习率/轮数/批量、样本 ID、损失轨迹、整包指纹、目标层指纹和内容哈希。

## 4. 兼容判定

判定顺序：

1. 没有旧产物：`NO_EXISTING_ADAPTER`，动作 `TRAIN`。
2. 产物结构/任务元数据非法：`ADAPTER_INVALID`、`E_ADAPTER_CONFIG`，动作 `TRAIN`。
3. 整包指纹相同：`COMPATIBLE_EXACT`，动作 `REUSE`。
4. 整包不同但目标层指纹相同：`COMPATIBLE_LAYER_UNCHANGED`，动作 `REUSE`。
5. 目标层指纹不同：`INCOMPATIBLE_BASE_VERSION`、`E_BASE_VERSION_MISMATCH`，动作 `TRAIN`。

兼容清单按任务名稳定排序，整份清单有独立 SHA-256。

## 5. 合并、预算与可逆性

对于目标层 `L` 和任务 `t`：

```text
merged[L] = quantize(W[L] + Σ_t scale_t · B_t A_t)
restored[L] = merged[L] - Σ_t scale_t · B_t A_t
```

求和严格按任务名字典序。量化使用 `round(x × scale) / scale`。允许误差为：

```text
tolerance_steps / quantization_scale
```

还原误差超限时抛出 `E_MERGE_NOT_REVERSIBLE`；该阶段不会发布新合并负载，历史版本目录和旧产物保持原样。

单任务预算成本：

```text
weight_elements = W.size + A.size + B.size
compute_units   = 2 · output_dim · input_dim · rank + output_dim · input_dim
```

每个请求记录序号、前后累计值、限制、允许/拒绝以及拒绝原因。遇到第一个无法满足的任务后停止叠加，当前任务及后续任务按稳定顺序进入待处理队列；它们没有任何权重被写入本次合并负载。

## 6. 阶段化快照

```text
workspace/
  report.json
  snapshot/
    CURRENT.json
    manifests/v00000005.json
    stages/
      load/v00000001/load.json
      training/v00000003/index.json + adapters/
      merge/v00000004/merged/ + merge.json
      export/v00000005/adapters/ + merged/ + report.json
```

提交协议：

1. 在 `snapshot/.tmp/` 创建唯一阶段临时目录。
2. 写完阶段负载、`index.json` 或 `merge.json`。
3. 为所有文件记录大小和 SHA-256，最后写 `_stage.json` 完成标记。
4. 将临时目录原子重命名到全新版本目录。
5. 先写历史清单，再原子替换 `CURRENT.json`。

恢复时验证 `CURRENT.json` 引用的阶段目录、完成标记和每个文件哈希。未出现在完整清单中的临时目录或孤儿阶段不会被信任。训练阶段可记录 `complete: false` 的部分结果；重启会跳过该阶段已完成且校验通过的任务。

## 7. 并发与稳定输出

线程池最多使用配置中的 `concurrency`。全部任务提交后，主线程仍按任务名字典序收集结果，因此快照检查点也是稳定前缀。每个任务有由任务名和目标层派生的 `stable_key`；最终兼容清单、复用/重训清单、失败清单、预算账本、导出任务列表均采用稳定顺序。

训练不共享 RNG，也不把线程完成顺序写入适配器。验收测试分别以并发度 1 和 4 运行并逐字段比较。

## 8. 退出语义

- 优雅信号只设置退出标记，不打断阶段写入。
- 当前阶段完成并提交后，不再启动下一阶段，返回退出码 `0`。
- 强制终止可能留下 `.tmp` 目录，但不改变最后一个完整 `CURRENT.json`。
- 重启从完整版本恢复并输出恢复点版本；不承诺强制终止进程自身生成退出报告。

