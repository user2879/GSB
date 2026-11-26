# 本地验收结果

总体状态：**PASS**

## 环境与样本
- Python：`3.11.0rc1 (main, Jun 29 2026, 00:05:30) [GCC 11.4.0]`
- numpy：`1.26.4`
- tkinter：{'available': False, 'error': 'libtk8.6.so: cannot open shared object file: No such file or directory', 'required_for_core_render': False}
- 帧数：3，去噪步：9，分块行：2

## 双版本交换
- v1 重生成一致：`True`
- sidecar 重生成一致：`True`
- v2→v1→v2 字段无损：`True`

## 增量失效
- 编辑：`[{'index': 0, 'op': 'set_image', 'kind': 'images', 'id': 'img_shared', 'path': 'images/geometry_shared_replacement.png'}]`
- 未受影响帧 1 全部复用：`True`
- 帧 2 重算去噪步：`3`
- 帧 3 重算去噪步：`3`
- 本批重算点：`6`

## 分块行重划分
- 编辑前每帧分块数：`[2, 2, 2]`
- 编辑后每帧分块数：`[3, 3, 3]`
- 新行区间：`[[0, 10], [10, 21], [21, 32]]`
- 去噪重算点：`0`
- 像素不变：`True`
- 规范化完成顺序：`['chunk:1:0', 'chunk:1:1', 'chunk:1:2', 'chunk:2:0', 'chunk:2:1', 'chunk:2:2', 'chunk:3:0', 'chunk:3:1', 'chunk:3:2']`

## 预算与恢复
- 低预算代码：`E_BUDGET_PENDING`，消耗 `2`，待办 `7`
- 恢复代码：`OK`，本批消耗 `7`
- 恢复像素与全量一致：`True`
- 强杀返回码：`137`；重启前已提交代：`gen-000003`
- 强杀后状态：`{'exit_code': 4, 'code': 'E_SNAPSHOT_INCOMPLETE'}`
- 恢复像素与全量一致：`True`

## 并发一致性
- 观察并发：串行 `1` / 并行 `3`，上限 `4`
- 预算扣减序列一致：`True`
- 待办队列一致：`True`
- 已完成帧摘要一致：`True`
- 全量帧/分块摘要一致：`True`
- 全量账本事件一致：`True`
- 全量游标与预算序列一致：`True` / `True`

## 校验错误码
- `unknown_term` → `E_UNKNOWN_TERM`
- `bad_png_dimension` → `E_PNG_DIMENSION`
- `v1_width_overflow` → `E_V1_WIDTH`
- `unknown_edit_command` → `E_UNKNOWN_COMMAND`

- pytest：`29 passed in 9.69s`

机器可读明细见 `reports/verification.json`。
