# 实际验证记录

验证日期：2026-11-24。执行环境为当前工作区提供的 64 位 Linux 容器，Python 3.11.0rc1、NumPy 1.26.4、Pillow 10.3.0。代码只使用 `pathlib`、`os.replace`、`threading`、NumPy 和 Pillow 等跨平台接口；未在真实 Windows 主机上重复执行，Windows 10+ 的运行步骤见 README。

## 自动化测试

```text
python -m unittest discover -s tests -v
Ran 28 tests in 11.664s
OK
```

完整日志：`artifacts/verified/unittest.log`。

覆盖点：

- PNG 头/尺寸/模式错误隔离，配置枚举与单调性错误码。
- SplitMix64 坐标噪声确定性，五种显式舍入模式。
- 量化档位和位宽切换确实改变中间整数和最终输出。
- 层频率、单通道权重、纹理替换的精确失效和复用。
- 等权与半值边界样本。
- 不同并发度下预算扣减序列完全一致，`E_POINT_BUDGET_PENDING` 和最终排空。
- 两档位逐级误差、首次越界环节、归因证据、空白全量重建逐像素一致。
- 交互延迟 stale、命令队列排空、快照恢复和版本/哈希不一致。
- 注入编号与业务码映射，材质失败隔离。

## 基础样例首次运行

```text
python -m matgen.cli run --input samples/basic --output artifacts/verified/basic_fresh --fresh
```

结果：

- `status = ok`，业务码为空。
- 6 条交互命令全部在 500ms 预算内返回：`within_budget_responses = 6`、`stale_responses = 0`。
- 实测响应耗时：`87.862, 9.605, 320.389, 146.779, 299.778, 117.459 ms`。
- 最终摘要：`ba8bacb4630ab59a192bc7f258c3ef4cae7cc8ea6e06a40d67eff3cfcd9b5819`。
- `full_rebuild.exact_match = true`，8 个材质通道逐像素与空白缓存全量重建一致。
- 调度 3 批，其中 1 批记录 `E_POINT_BUDGET_PENDING`，后续批全部完成、待办为 0。
- 误差追踪：配置位宽 20、对比位宽 24；首个越界环节为 `noise_composition`；最终 1024 个像素不一致；归因 `mix_order`，反序证据最大差为 `0.726654052734375`，量化证据最大差为 `0.0013427734375`，边界重合像素 269。
- 使用 32×32、4 层、2 材质、4 通道；共记录 274 个边界样本（等权、贡献舍入或精确半值）。

报告：`artifacts/verified/basic_fresh/diagnostic_report.json`。

## 快照恢复

```text
python -m matgen.cli run --input samples/basic --output artifacts/verified/basic_resume ^
  --snapshot-dir artifacts/verified/basic_fresh/snapshots --resume
```

结果：

- 恢复判定 `usable = true`，从命令游标 6 继续，`commands_processed = 0`。
- 最终摘要仍为 `ba8bac...b5819`，`full_rebuild.exact_match = true`。
- 恢复后快照全局版本从 4 增至 5，四个阶段均为 `committed`。
- 目录中只保留 `snapshots/current`，没有残留 `.staging-*` 或 `.previous-*`。

报告：`artifacts/verified/basic_resume/diagnostic_report.json`。

## 注入与延迟

```text
python -m matgen.cli run --input samples/injected --output artifacts/verified/injected ^
  --injections samples/injected/injection_config.json --fresh
```

结果：

- `E_TEXTURE_HEADER`、`E_TEXTURE_SIZE`、`E_TEXTURE_MODE` 各命中一张坏纹理，4 张有效纹理继续加载。
- 顶层业务码还包含 6 条响应产生的 `E_RESPONSE_STALE`。
- 低延迟预算 5ms 下 6 条命令均返回 `E_RESPONSE_STALE`，命令仍排队并最终排空到 revision 6。
- `INJ-LAYER-001`、`INJ-LOAD-001..003`、`INJ-MAT-001` 全部触发且预期码匹配。
- `painted_plastic` 因 `E_MATERIAL_RUNTIME` 隔离失败，`brushed_steel` 仍产出 4 通道。
- 首次运行未建立可比较旧版本，`INJ-SNAP-001` 正确标记为 untouched，未伪造观测码。

报告：`artifacts/verified/injected/diagnostic_report.json`。

## 快照版本不一致注入

使用已建立的相同快照，并故意提交错误期望版本：

```text
python -m matgen.cli run --input samples/injected ^
  --output artifacts/verified/injected_version_mismatch ^
  --snapshot-dir artifacts/verified/injected/snapshots ^
  --injections samples/injected/injection_config.json ^
  --expected-snapshot-version 999 --resume
```

结果：

- 报告包含 `E_SNAPSHOT_VERSION`，恢复 `usable = false`，随后从空白缓存重建并写出新版本快照。
- `INJ-SNAP-001` 触发 1 次且 `matched = true`。
- 其余加载、延迟、材质注入继续按原映射匹配，证明异常注入互不干扰。

报告：`artifacts/verified/injected_version_mismatch/diagnostic_report.json`。

## 结论

- 交互摘要来自真实完整构建；预算不足时明确 stale，不缩短计算。
- 增量、缓存复用结果与空白全量重建在所有完整材质通道上逐像素一致。
- 快照版本/哈希恢复、整目录替换和中断回滚逻辑通过测试。
- 注入编号与稳定业务码映射可机器校验。
