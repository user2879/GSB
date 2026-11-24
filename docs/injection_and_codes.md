# 注入项与业务码

独立注入配置：`samples/injected/injection_config.json`。每项包含稳定 `id`、阶段、动作、目标、延时（可选）和 `expected_code`。

## 当前映射

| 注入编号 | 阶段 | 动作 | 预期业务码 |
| --- | --- | --- | --- |
| `INJ-LOAD-001` | `load` | PNG 签名损坏 | `E_TEXTURE_HEADER` |
| `INJ-LOAD-002` | `load` | 纹理尺寸与分辨率不符 | `E_TEXTURE_SIZE` |
| `INJ-LOAD-003` | `load` | 不支持的 PNG 像素模式 | `E_TEXTURE_MODE` |
| `INJ-LAYER-001` | `layer_noise` | 层 0 延迟 90ms，触发低延迟 stale | `E_RESPONSE_STALE` |
| `INJ-MAT-001` | `material_mix` | `painted_plastic` 运行时失败 | `E_MATERIAL_RUNTIME` |
| `INJ-SNAP-001` | `snapshot` | 恢复版本/哈希不一致 | `E_SNAPSHOT_VERSION` |

`samples/injected/expected_codes.json` 和 `expected_diagnostic_structure.json` 提供机器可读映射与结构。

## 隔离规则

- 加载异常按纹理/材质/命令逐项记录，其他对象继续处理。
- 层运行时失败只阻断该层及其下游；不引用失败层的材质仍可产出。
- 材质运行时失败只阻断该材质，其他材质仍进入最终合成。
- 延迟注入不改变数值，只验证异步响应和待办排空。
- 快照不一致记录 `E_SNAPSHOT_VERSION` 后从空白缓存重建，不读取不兼容数组。

## 验证快照不一致注入

先建立兼容快照，再用错误期望版本触发不一致：

```bat
python -m matgen.cli run ^
  --input samples\injected ^
  --output artifacts\injected ^
  --injections samples\injected\injection_config.json ^
  --fresh

python -m matgen.cli run ^
  --input samples\injected ^
  --output artifacts\injected_version_mismatch ^
  --snapshot-dir artifacts\injected\snapshots ^
  --injections samples\injected\injection_config.json ^
  --expected-snapshot-version 999 ^
  --resume
```

第二份报告应同时出现 `E_SNAPSHOT_VERSION`，且 `INJ-SNAP-001` 为已触发并匹配。
