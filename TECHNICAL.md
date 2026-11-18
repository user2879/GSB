# 视频事件边界标注与事件轨迹重建诊断工作台 — 技术说明

## 概览

工作台以逐帧 PNG 序列为输入，按「证据提取 → 边界判定 → 轨迹重建 → 结果汇总」
四个阶段处理多路合成视频序列，并对并行预算调度做全程审计。全部输出确定性：
任何并发度下事件段与轨迹结果与串行执行逐字段一致。

## 目录结构

```
run_workbench.py          # CLI 入口（gen-samples / run）
workbench/
  pngcodec.py             # 纯 Python PNG 编解码（8-bit RGB，含 5 种扫描线滤波）
  config.py               # 事件判定参数 / 调度配置加载与配置哈希（SHA-256）
  samplegen.py            # 确定性样本与配置生成（无外部下载）
  evidence.py             # 清单校验、分辨率校验、变化掩码/比例/颜色直方图
  candidates.py           # 连通域聚合、图元级候选、重叠抑制（直方图距离→面积→类别优先级）
  segments.py             # 时间轴事件段：最短时长过滤、短间隙合并（记录合并链）
  tracks.py               # 跨段轨迹关联：颜色签名 + 空间连续性，空档帧断开
  blockcache.py           # 帧块驻留 LRU（同访问序号按键字典序淘汰）
  scheduler.py            # 全局帧预算 + 单序列配额账本、线程池调度、线程峰值统计
  snapshot.py             # 阶段快照：临时目录整体替换提交，清单含版本号/配置哈希/游标
  report.py               # 诊断报告字段与业务码一致性校验、原子写盘
  pipeline.py             # 四阶段编排、预算扣减、信号处理、恢复判定
  errors.py               # 业务码：E_DECODE_FAIL / E_BUDGET_EXHAUSTED / E_CONFIG_MISMATCH / E_INTERNAL
tests/
  run_tests.py            # 验收测试（38 项断言，7 个场景）
  test_units.py           # 单元测试（抑制、合并链、轨迹断开、置信度裁剪）
samples/                  # 确定性生成的 4 路 × 40 帧 PNG 序列 + 帧清单
configs/                  # event_params.json / schedule.json
runs/full/                # 一次完整运行的产物（events/tracks/report/failures/snapshots）
```

## 确定性契约（并行 = 串行）

- 帧调度顺序在任何并发度下固定：按序列名字典序轮询、帧序号升序；
  预算账本只在主线程按该顺序扣减，工作线程仅计算证据。
- 所有排序键确定：候选按（帧号, 左上角坐标），事件段按（起始毫秒, 类别），
  轨迹按（置信度降序, 首次出现毫秒, 轨迹标识）。
- LRU 淘汰同访问序号按帧块键字典序决断；淘汰块再次被引用时从原始帧文件
  重新解码重算，并与首次结果逐字段比对（不一致记 E_INTERNAL/RecomputeMismatch）。

## 预算与恢复语义

- 全局帧预算与单序列配额为「单次运行额度」；预算耗尽时输出
  E_BUDGET_EXHAUSTED，保留已完成帧证据与待处理帧游标，续跑（同配置）从游标继续。
- 快照按 `snapshot_commit_interval_frames` 及各阶段完成时提交；清单记录版本号、
  配置哈希、各序列游标、已消耗全局预算、事件段摘要、驻留帧块键集合与阶段产物版本。
- 启动时校验清单版本号与配置哈希，不一致输出 E_CONFIG_MISMATCH 并以退出码 2 拒绝恢复。
- SIGINT/SIGTERM：完成当前帧产物落盘与进度提交后以退出码 0 退出；
  SIGKILL 后仅依据已完整替换的快照恢复，恢复点（阶段 + 各序列帧游标）打印并写入报告。

## 运行方式

```bash
python3 -m venv .venv && .venv/bin/pip install numpy pyyaml   # 环境
.venv/bin/python run_workbench.py gen-samples --samples samples --configs configs
.venv/bin/python run_workbench.py run --samples samples --configs configs --workdir runs/full
.venv/bin/python run_workbench.py run --samples samples --configs configs \
    --workdir runs/par --max-concurrency 4                     # 指定并发度
.venv/bin/python tests/test_units.py                           # 单元测试
.venv/bin/python tests/run_tests.py                            # 验收测试
```

注：开发验证在 Linux(WSL) 环境完成；快照/报告均使用「临时文件/目录 + os.replace
原子替换」，在 Windows 上语义一致，工作目录可写即可。

## 实际验证结果（2026-11-18，Python 3.11 + numpy 2.4.6 + PyYAML 6.0.3）

- 样本确定性：两次生成逐字节一致（md5 校验通过），不依赖外部下载。
- 并行/串行一致：并发 1 与并发 4 的 events.json / tracks.json 逐字段一致，
  配额扣减序列一致，并行线程峰值 4。
- 预算耗尽续跑：全局预算 100（总量 160）首轮恰好消耗 100 并输出
  E_BUDGET_EXHAUSTED，续跑完成全部帧，事件段/轨迹与串行参考逐字段一致。
- 强制终止恢复：SIGKILL 后快照清单完整，恢复运行打印恢复点序列与帧游标，
  最终结果与串行参考逐字段一致。
- 信号退出：SIGTERM 后退出码 0，进度已提交，续跑结果一致。
- 解码失败隔离：损坏单帧 PNG → E_DECODE_FAIL，该序列被排除出边界输出，
  其余序列结果与参考一致。
- 配置不匹配：修改事件参数后续跑输出 E_CONFIG_MISMATCH，退出码 2，拒绝恢复。
- 驻留压力：驻留预算 1 时发生 39 次淘汰、20 次重算，无 RecomputeMismatch，
  事件段/轨迹与串行参考逐字段一致。
- 验收测试：`tests/run_tests.py` 38 项全部通过；`tests/test_units.py` 7 项全部通过。
