# 确定性纹理与材质生成组件

这是一个可直接集成的 Python 3.11+ 组件，用于从本地 PNG 基础纹理、材质描述、生成配置和交互命令流生成四通道材质，并满足以下约束：

- 逐层程序化噪声由像素坐标和层指纹派生，不使用运行时全局随机状态。
- 噪声量化、材质混合、通道合成全部使用显式定点整数运算。
- 缓存键由上游内容指纹、参数指纹和定点档位派生，支持精确失效与复用账本。
- 线程池并发执行层噪声与材质分片，单批点数预算和待办队列可复核。
- 交互命令在延迟预算内返回最近已完成摘要，无法及时完成时返回 `E_RESPONSE_STALE`，不缩短计算。
- 快照先写临时目录，再整体替换 `current`；支持版本/配置哈希恢复和中断回滚。
- 每次运行输出逐级误差追踪、逐像素重建一致性、注入码映射和诊断 JSON。

## 环境与安装

目标环境：64 位 Windows 10 或更高版本、Python 3.11+，`python` 在 `PATH` 中。依赖仅 `numpy` 和 `Pillow`，不需要 GPU、账号或外部服务。

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` 固定为：

```text
numpy==1.26.4
Pillow==10.3.0
```

组件不依赖 Docker、shell 脚本或平台私有 API。快照目录替换在 Windows 上使用“旧目录改名为备份 -> 暂存目录改名 -> 清理备份”的流程。

## 快速开始

仓库已经包含确定性样本；也可重新生成：

```bat
python -m matgen.cli generate-samples --root samples
python -m matgen.cli run --input samples\basic --output artifacts\basic --fresh
python -m matgen.cli run --input samples\basic --output artifacts\basic --resume
```

注入与异常隔离示例：

```bat
python -m matgen.cli run ^
  --input samples\injected ^
  --output artifacts\injected ^
  --injections samples\injected\injection_config.json ^
  --fresh
```

主要输出：

- `artifacts\basic\diagnostic_report.json`：完整诊断报告。
- `artifacts\basic\artifacts\layers.npz`：每层未量化值、量化整数和残差。
- `artifacts\basic\artifacts\material_channels.npz`：逐通道混合结果和逐层贡献。
- `artifacts\basic\artifacts\final_channels.npz`：最终四通道定点像素。
- `artifacts\basic\artifacts\error_trace_differences.npz`：四个环节逐像素差值。
- `artifacts\basic\artifacts\preview__*.png`：每个材质通道的 8 位预览。
- `artifacts\basic\snapshots\current\`：整体替换的版本化快照。

## 输入目录

```text
samples/basic/
  config.json
  materials.json
  commands.json
  textures/
    base_01.png ... base_04.png
```

加载器递归扫描输入目录下全部 `*.png`，校验 PNG 容器签名、尺寸和像素模式；单张纹理失败只进入失败清单，不终止其余纹理、材质或命令解析。

### 生成配置

`config.json` 的关键字段：

- `resolution_tiers`：严格递增的正整数分辨率档位。
- `resolution`：必须位于档位列表内，且与所有基础纹理尺寸一致。
- `noise_layers[]`：`texture`、正有限 `frequency`、正有限 `amplitude`；数组下标即稳定层序号。
- `material_mix_mode`：`weighted`、`multiply`、`screen` 或 `overlay`。
- `fixed_point`：位宽、小数位宽、量化档位、允许档位集合、四通道舍入模式。
- `error_trace.bit_width_ladder`：必须包含当前位宽和至少一个更高位宽。
- `concurrency`、`max_concurrency`、`point_budget`、`latency_budget_ms`、`snapshot_commit_interval`。

### 材质与命令

材质层引用噪声层序号，并给出 `diffuse`、`roughness`、`metallic`、`normal_perturbation` 四通道非负权重。命令支持：

```json
{"op":"set_layer","layer":1,"frequency":0.19,"amplitude":0.15}
{"op":"set_weight","material":"brushed_steel","layer":2,"channel":"roughness","value":0.34}
{"op":"set_mix_mode","value":"multiply"}
{"op":"set_quantization","level":4}
{"op":"set_bit_width","bit_width":20}
```

修改某一层使该层及下游层失效；修改某材质某通道权重只使该通道产物和最终合成失效；其他层、其他通道直接复用。

## 组件集成

```python
from matgen.cache import CacheStore
from matgen.engine import MaterialEngine
from matgen.injection import InjectionRegistry
from matgen.interactive import InteractiveRunner
from matgen.loader import input_fingerprint, load_input
from matgen.snapshot import SnapshotManager

loaded = load_input("samples/basic")
snapshots = SnapshotManager("artifacts/basic/snapshots")
cache = CacheStore()
engine = MaterialEngine(cache=cache, injections=InjectionRegistry())
runner = InteractiveRunner(
    engine=engine,
    textures=loaded.textures,
    input_config_hash=input_fingerprint(loaded),
    commands=loaded.commands,
    snapshot_manager=snapshots,
    resume=True,
)
output = runner.run(loaded.config, loaded.materials)
final = output["final_result"]
print(final.final_digest, output["stale_responses"])
```

## 兼容性与验证

- 全部随机性来自配置种子、层指纹、像素坐标和内容哈希。
- 定点整数、显式舍入和固定遍历顺序保证同输入多次运行逐像素一致。
- 报告中的 `full_rebuild.exact_match` 对比“增量/缓存结果”和“空白缓存全量重建”的最终通道数组。
- 已提供 28 项自动化测试，覆盖加载、定点、噪声、失效、预算、交互、快照、误差归因和注入隔离。

```bat
python -m unittest discover -s tests -v
```

设计说明见 `docs/architecture.md`，诊断字段见 `docs/diagnostic_report_schema.md`，注入码映射见 `docs/injection_and_codes.md`，实际验证记录见 `docs/verification.md`。
