# 技术说明：本地离屏多介质透射渲染引擎

## 1. 设计目标与约束

纯 Python 3.10+ 与 NumPy（1.x/2.x 均可）实现，无网络、无 GPU、无外部数据，
全部样本由固定种子（`SEED=20261119`，PCG64）确定性派生，重复运行逐字节一致。
平台路径处理使用 `os.path`，原子替换使用 `os.replace`（Windows/Linux 均为原子语义），
不使用任何 POSIX 专有调用，可在 Windows 直接运行。

## 2. 模块与职责

| 模块 | 职责 | 对应业务要求 |
| --- | --- | --- |
| `errors.py` | `E_xxx` 业务码、统一 `RenderError`、退出码 | 1、8 |
| `scene.py` | JSON 解析、有限性/范围/覆盖校验 | 1 |
| `geometry.py` | slab 法光线-AABB、有序区间链、中点归属 | 2 |
| `precision.py` | IEEE-754 尾数量化（23 位即 binary32） | 4 |
| `renderer.py` | 指数透射累乘、散射合成、双/单精度对照、截断 | 3、4 |
| `scheduler.py` | 有界线程池、索引序内存门、分块边界中断 | 6、8 |
| `persistence.py` | 版本摘要、原子写、归档、摘要校验恢复 | 7、8 |
| `assembly.py` | 行序拼装、超采样平均、PGM(P5) 原子写、报告 | 3、8 |
| `stability.py` | 步长/阈值抖动、偏差度量、Spearman 趋势判定 | 5 |
| `samplegen.py` | 确定性场景、介质栅格、解析期望透射率 | 准备 |
| `pipeline.py` / `cli.py` | 端到端编排、信号处理、退出码 | 全部 |

## 3. 多介质求交与界面归属（要求 2）

1. 每条光线 `P(t)=O+tD` 对每个 AABB 用三轴 slab 法得到参数区间 `[t_enter,t_exit]`。
2. 收集所有区间端点作为切分点，在相邻切分点之间取**段中点**，按盒声明顺序
   （`volume.index` 升序）选第一个覆盖中点的盒作为唯一归属。
3. 同盒相邻段合并，输出严格有序、首尾相接、互不重叠的区间链：
   交界点既是前段 `t_exit` 又是后段 `t_enter`，光学厚度只计一次，
   既不双重计入也无无主空隙（`validate_chain_integrity` 自检）。
4. 渲染步进以**段中点**查询消光系数，因此边界采样点不依赖端点归属判定，
   结果与分块调度顺序无关（有专门的并发一致性测试）。

## 4. 透射率、散射与终止（要求 3）

沿光线以固定步长 `ds` 切段，段长在区间链末端截断：

```
attenuation = exp(-sigma_t(mid) * seg_len)
T64        *= attenuation                          # float64 参考路径
T32         = quantize_mantissa(T32*attenuation, m) # 单精度路径，仅用于误差
scatter    += T64 * albedo * sigma_t * L_in * seg_len
```

* 光源项 `L_in` 对每盏点光源做阴影光线，阴影透射率沿区间链**解析积分**
  `exp(-∑sigma·Δl)`，并采用平方反比衰减（参考半径 `light_radius`）。
* 终止条件：`T64 < transmittance_cutoff`（`cutoff`）或步数达 `max_steps`
  （`max_steps`），逐光线记录原因并在报告统计占比。
* 灰度只由 float64 路径产生，经峰值归一化后 `round` 量化到 `0..255` 写 PGM。

## 5. 双/单精度误差对照（要求 4）

`quantize_mantissa(x, 23)` 与 `struct.pack('f',...)` 的 binary32 结果逐位一致
（有单元测试佐证）。每一步同时维护两条累乘路径，记录逐点绝对差
`|T64-T32|` 的逐光线最大值；首次超过 `transmittance_tolerance` 时记录
采样点序号、参数位置、介质盒名/序号、像素与子像素坐标及当时两条透射率。
最终图像不含任何单精度结果，单精度路径仅供误差评估。

误差传播机理：单精度在每一步引入约 `2^-24` 相对舍入，随累乘步数线性累积，
路径越长、介质层数越多，绝对差越大；容差收紧即可稳定复现首次超差定位。

## 6. 稳定性量化（要求 5）

每个抖动配置以相同相机/介质完整重渲，回到最终像素分辨率后对基准计算：

* `max_abs_pixel_deviation = max|a-b|`
* `mean_squared_pixel_deviation = mean((a-b)^2)`
* `mean_abs_pixel_deviation`

趋势完全由实测值驱动（无预设结论）：

* 步长：基准取最细步长（偏差 0），若偏差随步长增大单调不减且
  Spearman 秩相关 ≥ 0.9，判定“细化收敛”。
* 阈值：区分**放宽侧**（阈值 ≥ 基准）与收紧侧；放宽侧严格单调不减且
  Spearman ≥ 0.9 判定“随阈值放宽发散”。收紧侧常因截断下限出现平台，单独报告。

实测（默认样本）：步长序列 `[0.0625,0.125,0.25,0.5]` 的逐像素最大偏差
`[0,1.56,5.46,10.14]`，Spearman=1.000，呈收敛；阈值放宽侧
`[1e-3,5e-3,2e-2]` 偏差 `[0,0.042,0.096]`，Spearman=1.000，呈发散。

## 7. 并发、内存与中断（要求 6、8）

* 单派发线程严格按分块索引序把任务放入有界队列，队列容量由
  `memory_limit_bytes` 与每块缓冲记账（`tile_ray_memory`）共同决定；
  队列满即阻塞，实现“超内存按索引序等待”，绝不降步长/精度。
* 单块需求即超上限直接 `E_2003`；任一块异常置取消位，工作线程在分块边界
  停止领取新块，首个错误向上抛出（`E_2004`）。
* SIGINT/SIGTERM 或预置 `INTERRUPT` 文件只在分块边界生效：写中断标记、
  以退出码 `130` 与 `E_4001` 返回，未完成块不写正式文件，无法被误认为完成。
* 强制 kill 后重启：仅加载版本一致且 SHA-256 摘要有效的块，损坏块归档并报
  `E_3001`，不承诺生成退出报告。

## 8. 持久化与原子输出（要求 7、8）

* 版本标识 = 介质/光源/相机/步长/阈值规范化 JSON 的 SHA-256（`v1-<32hex>`）。
* 写块先写同目录 `*.tmp.<pid>.<nonce>`，`flush+fsync` 后 `os.replace` 整体替换。
* 已有块版本不一致或损坏时先移动到 `state/archive/` 再写新块，绝不直接覆盖。
* 最终 PGM 与 `report.json` 同样走临时文件 + 原子替换，成功后清除中断标记。

## 9. 错误码一览

| 码 | 含义 |
| --- | --- |
| E_1001 | JSON 解析失败 |
| E_1002 | 结构/必填字段缺失 |
| E_1003 | 介质盒非法（尺寸/消光/反照率/折射率） |
| E_1004 | 光源非法（越界/强度） |
| E_1005 | 相机非法（视场/宽高比/朝向） |
| E_1006 | 渲染/分块参数非法（含分块不覆盖网格） |
| E_1007 | 状态/输出目录不可写 |
| E_2001 | 求交链完整性失败 |
| E_2002 | 像素渲染异常 |
| E_2003 | 单块内存需求超上限 |
| E_2004 | 分块运行期异常/取消 |
| E_3001 | 状态文件损坏 |
| E_3002 | 原子写出失败 |
| E_4001 | 分块边界中断（退出码 130） |

校验在任何输出文件创建之前完成，非法输入保证零副作用（有测试与验证脚本佐证）。

## 10. 复现实验

```bash
pip install -r requirements.txt
python -m offscreen_render.cli generate --sample-dir samples --state-dir state --output-dir output
python -m offscreen_render.cli render   --scene samples/scene.json
python verify_local.py                  # 28 项端到端验收
python -m unittest discover -s offscreen_render/tests   # 38 个单元测试
```

更换参数无需改源码：可直接编辑 `samples/scene.json`，或向 `verify_local.py`
传入 `--width/--height/--supersample` 等参数重新生成样本。
