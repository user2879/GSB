using System.Text.Json.Nodes;

namespace LevelWorkbench.Core;

public static class RecordMigrator
{
    public static JsonObject MigrateToCurrent(SceneDocument scene, NormalizedPickRecord record)
    {
        var output = (JsonObject)record.Raw.Node.DeepClone();
        if (!output.ContainsKey("sourceVersion"))
        {
            output["sourceVersion"] = record.Raw.Version;
        }

        if (!output.ContainsKey("recordId"))
        {
            output["recordId"] = record.RecordId;
        }

        if (record.Raw.Version == RecordVersions.Legacy)
        {
            // Version 1 intentionally keeps hitPointCm and snapshotDigest so the legacy
            // coordinates and identity evidence remain auditable after the version switch.
            output["version"] = RecordVersions.StableIdentity;
            output["unit"] = "meter";
            output["stableId"] = record.Target.StableId;
            output["proxyNumber"] = record.Target.ProxyNumber;
            output["faceIndex"] = record.FaceIndex;
            output["hitPointMeters"] = SceneDigest.VecNode(record.HitPointMeters);
            output["faceNormal"] = SceneDigest.VecNode(record.FaceNormal);
            output["screenDistanceMeters"] = record.ScreenDistanceMeters;
            output["screenPoint"] = ScreenPointNode(record.ScreenPoint);
            output["layerId"] = record.LayerId;
            output["layerVisible"] = record.LayerVisible;
            output["pickable"] = record.Pickable;
            output["pickFilter"] = record.PickFilter;
        }

        if (record.Raw.Version <= RecordVersions.StableIdentity)
        {
            output["version"] = RecordVersions.Current;
            output["unit"] = "meter";
            output["stableId"] = record.Target.StableId;
            output["proxyNumber"] = record.Target.ProxyNumber;
            output["faceIndex"] = record.FaceIndex;
            output["hitPointMeters"] = SceneDigest.VecNode(record.HitPointMeters);
            output["faceNormal"] = SceneDigest.VecNode(record.FaceNormal);
            output["screenDistanceMeters"] = record.ScreenDistanceMeters;
            output["screenPoint"] = ScreenPointNode(record.ScreenPoint);
            output["layerId"] = record.LayerId;
            output["layerVisible"] = record.LayerVisible;
            output["pickable"] = record.Pickable;
            output["pickFilter"] = record.PickFilter;
            output["viewport"] = ViewportNode(record.Viewport, record.ViewportTransform);
        }

        if (record.Raw.Version == RecordVersions.Current)
        {
            // Re-validate and normalize units/normals even for current records, while leaving
            // every unknown property untouched.
            output["version"] = RecordVersions.Current;
            output["unit"] = "meter";
            output["stableId"] = record.Target.StableId;
            output["proxyNumber"] = record.Target.ProxyNumber;
            output["faceIndex"] = record.FaceIndex;
            output["hitPointMeters"] = SceneDigest.VecNode(record.HitPointMeters);
            output["faceNormal"] = SceneDigest.VecNode(record.FaceNormal);
            output["screenDistanceMeters"] = record.ScreenDistanceMeters;
            output["screenPoint"] = ScreenPointNode(record.ScreenPoint);
            output["layerId"] = record.LayerId;
            output["layerVisible"] = record.LayerVisible;
            output["pickable"] = record.Pickable;
            output["pickFilter"] = record.PickFilter;
            output["viewport"] = ViewportNode(record.Viewport, record.ViewportTransform);
        }

        return output;
    }

    private static JsonObject ScreenPointNode(Vec2? point)
    {
        if (point is null)
        {
            return new JsonObject { ["x"] = 0.0, ["y"] = 0.0 };
        }

        return new JsonObject
        {
            ["x"] = point.Value.X,
            ["y"] = point.Value.Y
        };
    }

    private static JsonObject ViewportNode(ViewportDefinition viewport, ViewportTransform transform) => new()
    {
        ["width"] = viewport.Width,
        ["height"] = viewport.Height,
        ["zoom"] = transform.Zoom,
        ["panXPixels"] = transform.PanXPixels,
        ["panYPixels"] = transform.PanYPixels
    };
}
