# 多通道采集与时钟同步诊断工作台 — 技术说明

## 1. 目标与范围

面向多机位、多设备采集现场的音视频时间轴对齐。输入为本地确定性合成的 4 路
独立晶振通道（48kHz/24bit 单声道 PCM + 320x180/30fps 灰度帧序列），每通道注入
ppm 级恒定漂移、缓慢温漂、非整数采样固定延迟、丢样本/丢帧空洞与周期同步脉冲。
工作台输出逐通道漂移估计、固定延迟、空洞填补、统一参考时间轴对齐、诊断快照，
并以依赖图驱动增量失效闭包，保证只重算失效节点、其余产物哈希不变，且在
强制终止后可从最近已提交状态一致恢复。

不使用 Docker、不依赖网络服务或真实采集设备；随机源全部来自固定种子的
numpy PCG64，禁用系统时间参与产物内容，因此两次相同序列运行逐字节一致。

## 2. 目录结构

```
syncbench/
  errors.py       8 个稳定业务错误码与异常链路
  config.py       4 通道真值、可调参数、声明容差与参数键
  rng.py          确定性随机源（PCG64 固定种子/独立流）
  hashing.py      规范化 JSON 与 SHA-256 内容哈希
  media.py        时钟物理模型 + 合成音视频素材生成器
  detection.py    脉冲/同步帧/空洞检测 + 素材格式校验
  estimation.py   分段 WLS 漂移拟合、置信区间、亚采样延迟
  gaps.py         三种空洞填补策略与 E_GAP_TOO_LONG
  timeline.py     双域统一时间轴映射与一致性约束
  graph.py        依赖图、参数键标注边、失效闭包、版本/复用
  snapshot.py     不可变快照、预算刷新与降级
  persistence.py  原子写、内容寻址块、孤儿回收、恢复
  engine.py       初始构建 + 增量闭包重算编排
  demo.py         固定序列演示（含强制终止/恢复）
  gui.py          PySide6 诊断视图（只渲染不可变快照）
  cli.py          命令行入口
tests/            39 个 pytest 用例，逐条覆盖 8 项要求
```

## 3. 时钟物理模型（media.py）

每通道相对参考时钟：

```
rate(u)   = 1 + (ppm + thermal_ppm*cos(2*pi*u/period)) / 1e6
dev(u)    = 积分 rate           # 参考时间 u -> 设备时间
content(u)= dev(u) + delay/fs   # 叠加非整数采样固定延迟
```

- 音频样本 n 的参考时刻由对 `content(u)=n/fs` 做 6 次牛顿迭代反演得到；
- 视频参考帧 k 经 `content()` 正向映射为带小数的本地帧位置，整数部分决定
  落入哪个本地帧槽，小数（亚帧相位）以 16 位像素列编码进同步标记，
  解决 30fps 整帧取整把亚帧漂移抹掉的问题；
- 同步脉冲在参考域周期注入，三角窗高电平（~0.9）远高于背景信号（<0.5）；
- 空洞通过置零音频样本/置黑视频帧注入，位置刻意避开脉冲窗与同步帧。

## 4. 估计、填补与双域对齐

- **脉冲检测**：绝对互相关 + 全序列非极大值抑制，按参考脉冲顺序单调配对，
  容忍高漂移通道后段数百样本的累积偏移；峰值经三点抛物线插值得到亚样本中心。
- **分段拟合**：参考位置->本地位置按时间分 3 段加权最小二乘，输出斜率/截距、
  标准误、95% t 置信区间与逐点残差；视频帧域量纲换算到样本域后做音视频联合。
- **固定延迟**：取最早两个脉冲的局部瞬时斜率外推回 ref=0，消除温漂对截距污染，
  线性通道延迟误差 < 0.5 样本。
- **漂移超限**：注入 ppm 超声明容差（默认 1000ppm）逐段返回 E_DRIFT_EXCEED，
  带通道号、段号、估计值与置信区间，而非仅点估计。
- **空洞填补**：线性插值 / 按漂移速率外推 / 标记不可填补；记录漂移估计版本戳
  与插值区间；长度超阈值返回 E_GAP_TOO_LONG 并给首个超限区间。
- **双域映射**：音频与视频片段由同一 `TimelineMap` 在同一连续时刻网格派生，
  一致性检查逐点比较，超差返回 E_SYNC_MISMATCH 并定位首个帧号与样本位置。

## 5. 依赖图与增量失效（graph.py / engine.py）

节点（每通道一套）：`drift -> gapfill / map -> aalign / valign -> tile`。
每条边标注触发失效的参数键（drift_model / delay_compensation /
gap_policy / reference_channel）。单参数修改：

1. 计算精确失效闭包（直接绑定 + 沿触发边向下游传播）；
2. 仅闭包内节点按拓扑序重算，闭包外节点保留内容哈希并写 reuse_source；
3. `assert_closure_matches` 强校验闭包集合 == 实际重算集合，不一致即
   E_DEPENDENCY_STALE；
4. 参考通道切换只使 map/aalign/valign/tile 失效，drift 与 gapfill
   不依赖参考通道，全部复用且哈希不变（测试与报告均有证据）。

时间轴版本戳由真实上游输入（漂移哈希、本通道延迟开关、参考通道）派生，
不含全局参数版本，避免无关通道修改污染版本。

## 6. 快照、预算与持久化

- **不可变快照**：创建即固化整体内容哈希与单调版本号；`SnapshotStore.replace`
  校验哈希与版本，半更新/旧版本一律 E_SNAPSHOT_STALE 拒绝；视图只经
  `render_guard` 读取当前快照。
- **预算刷新**：逐瓦片测量渲染耗时，超声明预算即跳帧降级并累计降级次数、
  返回 E_BUDGET_EXCEEDED；降级只减展示密度，不触碰测量/对齐状态。
- **原子持久化**：所有状态先写 `.tmp` + fsync，再 `os.replace` 原子改名；
  块内容寻址存于 `store/blobs`；未完成 replace 的文件重启即孤儿回收。
- **受控 vs 强制终止**：受控退出写 `shutdown.marker` 完整清单；强制终止
  （演示用 `os._exit(137)` 模拟 SIGKILL）无 marker，重启只信任最近已提交
  `state.json`，回收 `.tmp` 孤儿并记 `orphans.log`，绝不伪造退出报告。
- **恢复一致性**：状态内含完整参数快照，恢复时重建崩溃前配置，按拓扑序
  重算内存产物并逐节点比对内容哈希，任何漂移即 E_DEPENDENCY_STALE。

## 7. 业务错误码

| 码 | 触发场景 |
| --- | --- |
| E_CH_FORMAT | 通道数/采样率/位深/帧尺寸不合规 |
| E_PULSE_MISSING | 脉冲候选不足或同步帧数量不符 |
| E_DRIFT_EXCEED | 注入 ppm 超声明容差，带通道与段号 |
| E_GAP_TOO_LONG | 空洞超长度阈值，带首个超限区间 |
| E_SYNC_MISMATCH | 音视频双域映射超差，带帧号/样本定位 |
| E_DEPENDENCY_STALE | 读失效节点 / 闭包与重算集合不符 / 恢复哈希漂移 |
| E_SNAPSHOT_STALE | 快照哈希不符或视图渲染过期版本 |
| E_BUDGET_EXCEEDED | 瓦片渲染超交互预算，跳帧降级 |

## 8. 如何运行（Windows）

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1     # 建 venv 装依赖
.\.venv\Scripts\python -m syncbench.cli demo --workspace workspace --crash
$env:QT_QPA_PLATFORM="offscreen"; .\.venv\Scripts\python -m pytest tests/
.\.venv\Scripts\python -m syncbench.cli gui --workspace workspace
```

CLI 等价（Linux/macOS）：`python -m syncbench.cli demo --workspace workspace --crash`。
产物：`media/` 素材、`manifest.json` 真值清单、`store/` 持久化状态、
`reports/demo_report.json` 与 `reports/node_hashes.json`。

## 9. 实际验证结果

- 全套 39 个 pytest 用例通过（`pytest tests/`）。
- 线性通道分段 ppm 误差 < 5ppm；高漂移通道（1350ppm）正确触发 E_DRIFT_EXCEED。
- 线性通道亚采样延迟误差 < 0.5 样本；音视频双域映射最大时间差 0（同网格同源）。
- 单参数修改闭包与实际重算集合完全一致；参考切换时 8 个 drift/gapfill
  节点哈希全部保持不变。
- 两次相同序列演示的 `demo_report.json`、素材文件、`state.json` 逐字节一致。
- 强制终止（退出码 137）后孤儿 `state.json.tmp` 被回收，从 seq=4 已提交态
  恢复并继续增量，结果与未中断序列一致。
