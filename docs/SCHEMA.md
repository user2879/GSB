# JSON 格式

所有坐标字段都带单位后缀。场景和当前记录只接受米制；v1 明确声明厘米。数值必须是有限数。

## 场景 `level-workbench.scene`

```json
{
  "schema": "level-workbench.scene",
  "schemaVersion": 1,
  "name": "deterministic-multiversion-pick-scene",
  "unit": "meter",
  "coordinateToleranceMeters": 0.0005,
  "concurrencyLimit": 4,
  "memoryBudgetBytes": 1048576,
  "chunkSize": 2,
  "viewport": { "width": 1280, "height": 720 },
  "camera": {
    "positionMeters": [5, 4, 8],
    "targetMeters": [0, 0, 0],
    "up": [0, 1, 0],
    "verticalFovDegrees": 48,
    "nearMeters": 0.05,
    "farMeters": 100
  },
  "layers": [
    { "id": "layer-main", "name": "Main", "visible": true, "pickable": true, "pickFilter": "default" }
  ],
  "proxies": [
    {
      "id": "proxy-a",
      "proxyNumber": 102,
      "stableId": "stable-proxy-a",
      "layerId": "layer-main",
      "visible": true,
      "pickable": true,
      "pickFilter": "default",
      "transform": {
        "positionMeters": [-2.4, 0.2, -0.2],
        "rotationEulerDegrees": [0, 0, 0],
        "scale": [1, 1, 1]
      },
      "geometry": {
        "verticesMeters": [[-0.625, -0.85, -0.525]],
        "indices": [0, 1, 2]
      },
      "snapshotDigest": "sha256:..."
    }
  ]
}
```

校验规则：

- `id`、`stableId`、`proxyNumber` 分别全局唯一。
- 每个索引必须是整数且落在顶点范围内；索引数非零且能被 3 整除。
- 顶点三分量有限；缩放分量非零；相机参数有效。
- 图层 `visible`、`pickable`、`pickFilter` 全部必填。
- `snapshotDigest` 若声明，必须等于对局部顶点和三角形索引的规范化 SHA-256。
- 三角形变换顺序为缩放、绕 X/Y/Z 旋转（`Rz * Ry * Rx`）、平移。

## 记录集合

```json
{
  "schema": "level-workbench.pick-records",
  "formatVersion": 3,
  "records": [
    {
      "version": 1,
      "recordId": "legacy-a-001",
      "unit": "centimeter",
      "proxyNumber": 102,
      "faceIndex": 0,
      "hitPointCm": [-192.62195121951224, 44.32926829268303, 32.5],
      "snapshotDigest": "sha256:...",
      "legacyNote": "unknown fields are preserved"
    },
    {
      "version": 2,
      "recordId": "modern-center-001",
      "unit": "meter",
      "stableId": "stable-center-far",
      "faceIndex": 0,
      "hitPointMeters": [1.6588997082058672, 1.3271197665646932, 2.6542395331293864],
      "faceNormal": [0, 0, 1],
      "screenDistanceMeters": 6.847218038829546,
      "screenPoint": { "x": 640, "y": 360 },
      "layerId": "layer-main",
      "layerVisible": true,
      "pickable": true,
      "pickFilter": "default"
    },
    {
      "version": 3,
      "recordId": "current-b-001",
      "unit": "meter",
      "stableId": "stable-proxy-b",
      "faceIndex": 1,
      "hitPointMeters": [2.3861842105263156, 0.21447368421052593, 1.1500000000000004],
      "faceNormal": [0, 0, 1],
      "screenDistanceMeters": 8.251347918300171,
      "screenPoint": { "x": 681.8, "y": 342.6 },
      "layerId": "layer-main",
      "layerVisible": true,
      "pickable": true,
      "pickFilter": "default",
      "viewport": { "width": 1280, "height": 720, "zoom": 1, "panXPixels": 0, "panYPixels": 0 }
    }
  ]
}
```

已知版本为 `1, 2, 3`，文件级版本和记录级版本都必须属于该集合。混载允许单文件中同时出现多个记录版本。

## 拾取脚本 `level-workbench.pick-script`

```json
{
  "schema": "level-workbench.pick-script",
  "version": 1,
  "name": "deterministic-interaction-script",
  "viewport": { "zoom": 1, "panXPixels": 0, "panYPixels": 0 },
  "steps": [
    { "kind": "pick", "screen": { "x": 640, "y": 360 }, "filter": "default" },
    { "kind": "select", "screen": { "x": 504, "y": 307 }, "additive": false, "filter": "default" },
    { "kind": "drag", "start": { "x": 504, "y": 307 }, "end": { "x": 559, "y": 332 }, "filter": "default" },
    { "kind": "box", "min": { "x": 300, "y": 150 }, "max": { "x": 900, "y": 600 }, "additive": false, "filter": "default" }
  ]
}
```

脚本在状态化编辑器会话中执行。每个块完成后持久化该块步骤结果、产出记录、编辑器位姿/选中集/视口快照和块摘要。
