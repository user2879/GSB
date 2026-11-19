using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using LevelWorkbench.Core;

namespace LevelWorkbench.SampleGen;

public static class Program
{
    private static readonly JsonSerializerOptions Pretty = new() { WriteIndented = true };

    public static int Main(string[] args)
    {
        var output = GetOption(args, "--output") ?? Path.Combine("sample-data");
        try
        {
            Generate(output);
            Console.WriteLine(JsonSerializer.Serialize(new
            {
                ok = true,
                output = Path.GetFullPath(output),
                files = Directory.EnumerateFiles(output, "*", SearchOption.AllDirectories)
                    .Select(Path.GetFullPath)
                    .OrderBy(p => p, StringComparer.Ordinal)
                    .ToArray()
            }, Pretty));
            return 0;
        }
        catch (Exception ex)
        {
            var error = ex is WorkbenchException wb
                ? wb.ToErrorNode()
                : new JsonObject { ["ok"] = false, ["error"] = new JsonObject { ["code"] = ErrorCodes.Internal, ["message"] = ex.Message } };
            Console.WriteLine(error.ToJsonString(Pretty));
            return 2;
        }
    }

    public static void Generate(string outputDirectory)
    {
        Directory.CreateDirectory(outputDirectory);
        var recordsDirectory = Path.Combine(outputDirectory, "records");
        var invalidDirectory = Path.Combine(outputDirectory, "invalid");
        Directory.CreateDirectory(recordsDirectory);
        Directory.CreateDirectory(invalidDirectory);

        var cameraPosition = new Vec3(5, 4, 8);
        var cameraTarget = Vec3.Zero;
        var forward = (cameraTarget - cameraPosition).Normalized();
        var sceneNode = BuildScene(
            cameraPosition,
            cameraTarget,
            forward);
        var sceneJson = sceneNode.ToJsonString(Pretty) + "\n";
        Write(Path.Combine(outputDirectory, "scene.json"), sceneJson);
        var scene = SceneLoader.Parse(sceneJson, "generated-scene");

        var pickEngine = new PickEngine();
        var identity = ViewportTransform.Identity;
        var center = new Vec2(scene.Viewport.Width / 2.0, scene.Viewport.Height / 2.0);
        var centerHit = RequireHit(scene, pickEngine, center, "default", "center far proxy");
        var proxyA = scene.ProxyByStableId("stable-proxy-a");
        var proxyB = scene.ProxyByStableId("stable-proxy-b");
        var gizmo = scene.ProxyByStableId("stable-gizmo");
        var pointA = FindHitScreen(scene, pickEngine, proxyA.Id, "default");
        var pointB = FindHitScreen(scene, pickEngine, proxyB.Id, "default");
        var pointGizmo = FindHitScreen(scene, pickEngine, gizmo.Id, "gizmo");
        var hitA = RequireHit(scene, pickEngine, pointA, "default", "proxy A");
        var hitB = RequireHit(scene, pickEngine, pointB, "default", "proxy B");
        var hitGizmo = RequireHit(scene, pickEngine, pointGizmo, "gizmo", "gizmo proxy");

        if (hitA.StableId != proxyA.StableId || hitB.StableId != proxyB.StableId || hitGizmo.StableId != gizmo.StableId)
        {
            throw new InvalidOperationException("Deterministic sample screen points do not resolve to their intended proxies.");
        }

        var legacy = LegacyRecord("legacy-a-001", hitA, proxyA);
        var modern = ModernRecord("modern-center-001", 2, centerHit, scene, "legacy-studio-v2");
        var current = ModernRecord("current-b-001", 3, hitB, scene, "current-studio-v3");
        var gizmoRecord = ModernRecord("current-gizmo-001", 3, hitGizmo, scene, "current-gizmo-v3");

        WriteRecordFile(
            Path.Combine(recordsDirectory, "legacy-v1.json"),
            1,
            new JsonObject { ["studio"] = "legacy-v1-sample" },
            legacy);
        WriteRecordFile(
            Path.Combine(recordsDirectory, "modern-v2.json"),
            2,
            new JsonObject { ["studio"] = "modern-v2-sample" },
            modern);
        WriteRecordFile(
            Path.Combine(recordsDirectory, "current-v3.json"),
            3,
            new JsonObject { ["studio"] = "current-v3-sample" },
            current, gizmoRecord);
        WriteRecordFile(
            Path.Combine(recordsDirectory, "mixed.json"),
            3,
            new JsonObject { ["studio"] = "mixed-version-sample", ["preserveMe"] = new JsonObject { ["nested"] = true } },
            legacy, modern, current, gizmoRecord);

        var script = BuildPickScript(center, pointA, pointB, pointGizmo);
        Write(Path.Combine(outputDirectory, "pick-script.json"), script.ToJsonString(Pretty) + "\n");

        GenerateInvalidSamples(invalidDirectory, sceneNode, scene, hitA, hitB, centerHit);
    }

    private static JsonObject BuildScene(Vec3 cameraPosition, Vec3 cameraTarget, Vec3 forward)
    {
        var centerFarPosition = cameraPosition + forward * 8.0;
        var hiddenPosition = cameraPosition + forward * 5.0;
        var noPickPosition = cameraPosition + forward * 3.5;
        var scene = new JsonObject
        {
            ["schema"] = "level-workbench.scene",
            ["schemaVersion"] = 1,
            ["name"] = "deterministic-multiversion-pick-scene",
            ["unit"] = "meter",
            ["coordinateToleranceMeters"] = 0.0005,
            ["concurrencyLimit"] = 4,
            ["memoryBudgetBytes"] = 1048576,
            ["chunkSize"] = 2,
            ["viewport"] = new JsonObject { ["width"] = 1280, ["height"] = 720 },
            ["camera"] = new JsonObject
            {
                ["positionMeters"] = Vec(cameraPosition),
                ["targetMeters"] = Vec(cameraTarget),
                ["up"] = Vec(Vec3.UnitY),
                ["verticalFovDegrees"] = 48.0,
                ["nearMeters"] = 0.05,
                ["farMeters"] = 100.0
            },
            ["layers"] = new JsonArray
            {
                Layer("layer-main", "Main", true, true, "default"),
                Layer("layer-hidden", "Hidden but geometrically nearer", false, true, "default"),
                Layer("layer-no-pick", "Visible but not pickable", true, false, "default"),
                Layer("layer-gizmo", "Gizmo filter", true, true, "gizmo"),
                Layer("layer-offscreen", "Viewport clipping", true, true, "default")
            },
            ["proxies"] = new JsonArray
            {
                Proxy("proxy-center-far", 101, "stable-center-far", "layer-main", true, true, "default", centerFarPosition, Box(1.8, 1.8, 1.8)),
                Proxy("proxy-hidden", 201, "stable-hidden-center", "layer-hidden", true, true, "default", hiddenPosition, Box(1.2, 1.2, 1.2)),
                Proxy("proxy-nopick", 202, "stable-nopick-center", "layer-no-pick", true, true, "default", noPickPosition, Box(1.0, 1.0, 1.0)),
                Proxy("proxy-a", 102, "stable-proxy-a", "layer-main", true, true, "default", new Vec3(-2.4, 0.2, -0.2), Box(1.25, 1.7, 1.05)),
                Proxy("proxy-b", 103, "stable-proxy-b", "layer-main", true, true, "default", new Vec3(2.1, -0.2, 0.4), Box(1.35, 1.1, 1.5)),
                Proxy("proxy-gizmo", 301, "stable-gizmo", "layer-gizmo", true, true, "gizmo", new Vec3(0.9, 2.1, -2.0), Box(0.9, 0.9, 0.9)),
                Proxy("proxy-offscreen", 401, "stable-offscreen", "layer-offscreen", true, true, "default", new Vec3(12, -2, 3), Box(0.7, 0.7, 0.7)),
                Proxy("proxy-twin-left", 501, "stable-twin-left", "layer-no-pick", true, false, "default", new Vec3(-5, -2, 0), Box(0.55, 0.55, 0.55)),
                Proxy("proxy-twin-right", 502, "stable-twin-right", "layer-no-pick", true, false, "default", new Vec3(5, -2, 0), Box(0.55, 0.55, 0.55))
            }
        };
        return scene;
    }

    private static JsonObject Layer(string id, string name, bool visible, bool pickable, string filter) => new()
    {
        ["id"] = id,
        ["name"] = name,
        ["visible"] = visible,
        ["pickable"] = pickable,
        ["pickFilter"] = filter
    };

    private static JsonObject Proxy(
        string id,
        int number,
        string stableId,
        string layerId,
        bool visible,
        bool pickable,
        string filter,
        Vec3 position,
        (Vec3[] Vertices, int[] Indices) geometry)
    {
        return new JsonObject
        {
            ["id"] = id,
            ["proxyNumber"] = number,
            ["stableId"] = stableId,
            ["layerId"] = layerId,
            ["visible"] = visible,
            ["pickable"] = pickable,
            ["pickFilter"] = filter,
            ["transform"] = new JsonObject
            {
                ["positionMeters"] = Vec(position),
                ["rotationEulerDegrees"] = Vec(Vec3.Zero),
                ["scale"] = Vec(new Vec3(1, 1, 1))
            },
            ["geometry"] = new JsonObject
            {
                ["verticesMeters"] = new JsonArray(geometry.Vertices.Select(v => (JsonNode?)Vec(v)).ToArray()),
                ["indices"] = new JsonArray(geometry.Indices.Select(i => (JsonNode?)JsonValue.Create(i)).ToArray())
            },
            ["snapshotDigest"] = GeometryDigest.Compute(geometry.Vertices, geometry.Indices)
        };
    }

    private static (Vec3[] Vertices, int[] Indices) Box(double x, double y, double z)
    {
        var hx = x / 2.0;
        var hy = y / 2.0;
        var hz = z / 2.0;
        var vertices = new[]
        {
            new Vec3(-hx, -hy, -hz), new Vec3(hx, -hy, -hz), new Vec3(hx, hy, -hz), new Vec3(-hx, hy, -hz),
            new Vec3(-hx, -hy, hz), new Vec3(hx, -hy, hz), new Vec3(hx, hy, hz), new Vec3(-hx, hy, hz)
        };
        var indices = new[]
        {
            4, 5, 6, 4, 6, 7,
            1, 0, 3, 1, 3, 2,
            5, 1, 2, 5, 2, 6,
            0, 4, 7, 0, 7, 3,
            3, 7, 6, 3, 6, 2,
            0, 1, 5, 0, 5, 4
        };
        return (vertices, indices);
    }

    private static JsonObject LegacyRecord(string recordId, PickHit hit, ProxyDefinition target) => new()
    {
        ["version"] = 1,
        ["recordId"] = recordId,
        ["unit"] = "centimeter",
        ["proxyNumber"] = target.ProxyNumber,
        ["faceIndex"] = hit.FaceIndex,
        ["hitPointCm"] = new JsonArray(
            JsonValue.Create(hit.HitPointMeters.X * 100.0),
            JsonValue.Create(hit.HitPointMeters.Y * 100.0),
            JsonValue.Create(hit.HitPointMeters.Z * 100.0)),
        ["snapshotDigest"] = target.Geometry.SnapshotDigest,
        ["legacyNote"] = "preserve-this-unknown-field"
    };

    private static JsonObject ModernRecord(string recordId, int version, PickHit hit, SceneDocument scene, string note)
    {
        var record = new JsonObject
        {
            ["version"] = version,
            ["recordId"] = recordId,
            ["unit"] = "meter",
            ["stableId"] = hit.StableId,
            ["proxyNumber"] = hit.ProxyNumber,
            ["faceIndex"] = hit.FaceIndex,
            ["hitPointMeters"] = Vec(hit.HitPointMeters),
            ["faceNormal"] = Vec(hit.FaceNormal),
            ["screenDistanceMeters"] = hit.ScreenDistanceMeters,
            ["screenPoint"] = new JsonObject { ["x"] = hit.ScreenPoint.X, ["y"] = hit.ScreenPoint.Y },
            ["layerId"] = hit.LayerId,
            ["layerVisible"] = hit.LayerVisible,
            ["pickable"] = hit.LayerPickable && hit.ProxyPickable,
            ["pickFilter"] = hit.PickFilter,
            ["studioNote"] = note,
            ["unknownModern"] = new JsonObject { ["kept"] = true }
        };
        if (version == 3)
        {
            record["viewport"] = new JsonObject
            {
                ["width"] = scene.Viewport.Width,
                ["height"] = scene.Viewport.Height,
                ["zoom"] = hit.ViewportTransform.Zoom,
                ["panXPixels"] = hit.ViewportTransform.PanXPixels,
                ["panYPixels"] = hit.ViewportTransform.PanYPixels
            };
        }

        return record;
    }

    private static JsonObject BuildPickScript(Vec2 center, Vec2 pointA, Vec2 pointB, Vec2 pointGizmo) => new()
    {
        ["schema"] = "level-workbench.pick-script",
        ["version"] = 1,
        ["name"] = "deterministic-interaction-script",
        ["viewport"] = new JsonObject { ["zoom"] = 1.0, ["panXPixels"] = 0.0, ["panYPixels"] = 0.0 },
        ["steps"] = new JsonArray
        {
            new JsonObject { ["kind"] = "pick", ["screen"] = Screen(center), ["filter"] = "default" },
            new JsonObject { ["kind"] = "pick", ["screen"] = Screen(pointA), ["filter"] = "default" },
            new JsonObject { ["kind"] = "select", ["screen"] = Screen(pointA), ["additive"] = false, ["filter"] = "default" },
            new JsonObject
            {
                ["kind"] = "drag",
                ["start"] = Screen(pointA),
                ["end"] = Screen(new Vec2(pointA.X + 55, pointA.Y + 25)),
                ["filter"] = "default"
            },
            new JsonObject
            {
                ["kind"] = "box",
                ["min"] = Screen(new Vec2(System.Math.Max(0, pointA.X - 150), System.Math.Max(0, pointA.Y - 150))),
                ["max"] = Screen(new Vec2(System.Math.Min(1279, pointB.X + 150), System.Math.Min(719, pointB.Y + 150))),
                ["additive"] = false,
                ["filter"] = "default"
            },
            new JsonObject { ["kind"] = "pick", ["screen"] = Screen(pointGizmo), ["filter"] = "gizmo" }
        }
    };

    private static JsonObject Screen(Vec2 point) => new()
    {
        ["x"] = point.X,
        ["y"] = point.Y
    };

    private static PickHit RequireHit(SceneDocument scene, PickEngine engine, Vec2 screen, string filter, string label)
    {
        var hit = engine.PickSerial(SceneSnapshot.FromDocument(scene), screen, new PickQuery
        {
            Filter = filter,
            ViewportTransform = ViewportTransform.Identity
        });
        return hit ?? throw new InvalidOperationException($"No deterministic hit for {label} at {screen}.");
    }

    private static Vec2 FindHitScreen(SceneDocument scene, PickEngine engine, string proxyId, string filter)
    {
        var proxy = scene.Proxies.First(p => p.Id == proxyId);
        var basis = CameraBasis.FromCamera(scene.Camera);
        var projection = basis.ProjectWorld(
            proxy.Transform.TransformPoint(Vec3.Zero),
            scene.Viewport,
            ViewportTransform.Identity,
            scene.Camera.NearMeters)
            ?? throw new InvalidOperationException($"Proxy {proxyId} is not projectable.");
        var candidates = new List<Vec2> { projection.Screen };
        for (var gx = -3; gx <= 3; gx++)
        {
            for (var gy = -3; gy <= 3; gy++)
            {
                candidates.Add(new Vec2(projection.Screen.X + gx * 12, projection.Screen.Y + gy * 12));
            }
        }

        foreach (var candidate in candidates)
        {
            if (candidate.X < 0 || candidate.X >= scene.Viewport.Width || candidate.Y < 0 || candidate.Y >= scene.Viewport.Height)
            {
                continue;
            }

            var hit = engine.PickSerial(SceneSnapshot.FromDocument(scene), candidate, new PickQuery
            {
                Filter = filter,
                ViewportTransform = ViewportTransform.Identity
            });
            if (hit?.ProxyId == proxyId)
            {
                return candidate;
            }
        }

        throw new InvalidOperationException($"Could not find a deterministic screen point for {proxyId}.");
    }

    private static void GenerateInvalidSamples(
        string directory,
        JsonObject sceneNode,
        SceneDocument scene,
        PickHit hitA,
        PickHit hitB,
        PickHit centerHit)
    {
        var duplicate = (JsonObject)sceneNode.DeepClone();
        ((JsonObject)((JsonArray)duplicate["proxies"]!)[1]!)["stableId"] = JsonValue.Create("stable-center-far");
        Write(Path.Combine(directory, "duplicate-stable-id.scene.json"), duplicate.ToJsonString(Pretty) + "\n");

        var badIndex = (JsonObject)sceneNode.DeepClone();
        var badGeometry = (JsonObject)((JsonObject)((JsonArray)badIndex["proxies"]!)[0]!)["geometry"]!;
        ((JsonArray)badGeometry["indices"]!)[0] = JsonValue.Create(9999);
        Write(Path.Combine(directory, "index-out-of-bounds.scene.json"), badIndex.ToJsonString(Pretty) + "\n");

        var badLayer = (JsonObject)sceneNode.DeepClone();
        ((JsonObject)((JsonArray)badLayer["layers"]!)[0]!).Remove("visible");
        Write(Path.Combine(directory, "missing-layer-visibility.scene.json"), badLayer.ToJsonString(Pretty) + "\n");

        var unknownVersion = new JsonObject
        {
            ["version"] = 99,
            ["recordId"] = "bad-version",
            ["unit"] = "meter"
        };
        WriteRecordFile(Path.Combine(directory, "unknown-record-version.json"), 99, new JsonObject(), unknownVersion);

        var badUnit = (JsonObject)ModernRecord("bad-unit", 2, centerHit, scene, "bad-unit");
        badUnit["unit"] = "centimeter";
        badUnit.Remove("hitPointMeters");
        badUnit["hitPointCm"] = SceneDigest.VecNode(centerHit.HitPointMeters * 100.0);
        WriteRecordFile(Path.Combine(directory, "unit-mismatch.records.json"), 2, new JsonObject(), badUnit);

        var badDigest = (JsonObject)LegacyRecord("bad-digest", hitA, scene.ProxyById(hitA.ProxyId));
        badDigest["snapshotDigest"] = "sha256:" + new string('0', 64);
        WriteRecordFile(Path.Combine(directory, "digest-mismatch.records.json"), 1, new JsonObject(), badDigest);

        var twin = scene.ProxyById("proxy-twin-left");
        var ambiguous = (JsonObject)LegacyRecord("ambiguous-digest", hitB, scene.ProxyById(hitB.ProxyId));
        ambiguous["proxyNumber"] = JsonValue.Create(twin.ProxyNumber);
        ambiguous["snapshotDigest"] = twin.Geometry.SnapshotDigest;
        ambiguous["faceIndex"] = JsonValue.Create(0);
        ambiguous["hitPointCm"] = SceneDigest.VecNode(twin.Transform.TransformPoint(twin.Geometry.Vertices[0]) * 100.0);
        WriteRecordFile(Path.Combine(directory, "ambiguous-digest.records.json"), 1, new JsonObject(), ambiguous);

        var tampered = (JsonObject)ModernRecord("tampered-replay", 3, hitB, scene, "tampered");
        var point = SceneDigest.VecNode(hitB.HitPointMeters + new Vec3(0.25, 0, 0));
        tampered["hitPointMeters"] = point;
        WriteRecordFile(Path.Combine(directory, "unreproducible.records.json"), 3, new JsonObject(), tampered);
    }

    private static void WriteRecordFile(string path, int formatVersion, JsonObject extras, params JsonObject[] records)
    {
        var root = new JsonObject();
        foreach (var (key, value) in extras)
        {
            root[key] = value?.DeepClone();
        }

        root["schema"] = "level-workbench.pick-records";
        root["formatVersion"] = formatVersion;
        root["recordCount"] = records.Length;
        root["records"] = new JsonArray(records.Select(record => (JsonNode?)record.DeepClone()).ToArray());
        Write(path, root.ToJsonString(Pretty) + "\n");
    }

    private static JsonArray Vec(Vec3 value) => new(
        JsonValue.Create(value.X),
        JsonValue.Create(value.Y),
        JsonValue.Create(value.Z));

    private static void Write(string path, string content)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        File.WriteAllText(path, content, new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));
    }

    private static string? GetOption(string[] args, string name)
    {
        for (var i = 0; i < args.Length; i++)
        {
            if (args[i] == name && i + 1 < args.Length)
            {
                return args[i + 1];
            }
        }

        return null;
    }
}
