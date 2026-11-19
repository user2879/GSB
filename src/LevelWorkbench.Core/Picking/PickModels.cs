using System.Text.Json.Nodes;

namespace LevelWorkbench.Core;

public sealed class PickQuery
{
    public string? Filter { get; init; }

    public ViewportTransform ViewportTransform { get; init; } = ViewportTransform.Identity;

    public ViewportDefinition? ViewportOverride { get; init; }

    public bool RespectVisibility { get; init; } = true;

    public int? ConcurrencyLimit { get; init; }

    public long? MemoryBudgetBytes { get; init; }
}

public sealed class PickHit
{
    public required string StableId { get; init; }
    public required int ProxyNumber { get; init; }
    public required string ProxyId { get; init; }
    public required string LayerId { get; init; }
    public required int FaceIndex { get; init; }
    public required Vec3 HitPointMeters { get; init; }
    public required Vec3 FaceNormal { get; init; }
    public required double ScreenDistanceMeters { get; init; }
    public required Vec2 ScreenPoint { get; init; }
    public required Vec2 BaseViewportPoint { get; init; }
    public required Ray3 Ray { get; init; }
    public required string? PickFilter { get; init; }
    public required bool LayerVisible { get; init; }
    public required bool LayerPickable { get; init; }
    public required bool ProxyVisible { get; init; }
    public required bool ProxyPickable { get; init; }
    public required ViewportTransform ViewportTransform { get; init; }

    public JsonObject ToCurrentRecord(string recordId, ViewportDefinition viewport) => new()
    {
        ["version"] = RecordVersions.Current,
        ["recordId"] = recordId,
        ["unit"] = "meter",
        ["stableId"] = StableId,
        ["proxyNumber"] = ProxyNumber,
        ["faceIndex"] = FaceIndex,
        ["hitPointMeters"] = SceneDigest.VecNode(HitPointMeters),
        ["faceNormal"] = SceneDigest.VecNode(FaceNormal),
        ["screenDistanceMeters"] = ScreenDistanceMeters,
        ["screenPoint"] = new JsonObject
        {
            ["x"] = ScreenPoint.X,
            ["y"] = ScreenPoint.Y
        },
        ["layerId"] = LayerId,
        ["layerVisible"] = LayerVisible,
        ["pickable"] = LayerPickable && ProxyPickable,
        ["pickFilter"] = PickFilter,
        ["viewport"] = new JsonObject
        {
            ["width"] = viewport.Width,
            ["height"] = viewport.Height,
            ["zoom"] = ViewportTransform.Zoom,
            ["panXPixels"] = ViewportTransform.PanXPixels,
            ["panYPixels"] = ViewportTransform.PanYPixels
        }
    };
}
