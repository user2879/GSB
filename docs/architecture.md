# 架构与恢复协议说明

## 数据流

```text
JSON 解析/校验
   -> PCG64 固定种子生成 float32 张量
   -> 每个调用建立 CallState
   -> 为每个阶段/lane 计算内容指纹与缓存键
   -> ThreadPoolExecutor 只提交缓存未命中的 lane
   -> 按 (调用序号, lane, 阶段) 排序应用结果
   -> 原子快照 + 交互摘要 + 最终报告
```

命令执行由单线程 state machine 串行化。lane 计算使用独立线程池，因此参数和输入张量的
修改不会与正在进行的 lane 计算交叉。命令的完成事件在状态更新后触发；快照在事件之后
执行，避免交互读取被磁盘 I/O 阻塞。

## 两类 lane

二维张量中，阶段缓存单位不是固定意义上的“数据行”，而是归约轴上的 lane：

- `axis=1`：一个 lane 是一行，lane 号就是行号；
- `axis=0`：一个 lane 是一列，lane 号是列号。

LayerNorm 的均值/方差是每个 lane 一个标量，但输出以与输入同形的矩阵保存，所以逐行
统计量在 `axis=1` 时是独立行统计，在 `axis=0` 时同列所有行共享统计量。

## 缓存键

缓存键是 canonical JSON 的 SHA-256，输入包括：

- 格式版本；
- 调用 ID 和稳定调用序号；
- 阶段名和 lane；
- lane 内容的 SHA-256；
- 归约轴；
- Softmax 稳定化开关（仅 Softmax）；
- epsilon 的 `float.hex()`（仅 LayerNorm）；
- LayerNorm 缩放的 `float.hex()`（仅 LayerNorm）。

参数不相关阶段不做过度失效：epsilon 不会触发 Softmax 重算，稳定化开关不会触发
LayerNorm 重算。内容替换由 lane 内容指纹自然触发，无需单独的“脏标记”扫描。

## 快照注意事项

- 所有数据文件位于同一个临时快照目录，目录完成后才整体重命名。
- `current.json` 是唯一提交点；它自身也通过同目录临时文件 + `os.replace` 提交。
- 阶段校验同时检查版本号和 SHA-256；配置哈希还包含初始调用与张量描述。
- 未引用目录可由运维安全清理；恢复只读取 `current.json` 指向的完整快照。
- 续跑游标是 `next_command_index`，不是快照版本。快照版本只标识提交次数。

## 确定性边界

固定线程数、固定 NumPy 版本与平台时输出位级可复现。跨 CPU 架构或 NumPy 大版本时，
`np.exp`、`np.sqrt`、`np.max` 等底层 libm/SIMD 实现可能带来最后一位差异；项目通过
显式固定求和顺序消除了线程归约带来的非确定性，但不承诺跨数学库实现的逐位一致。
