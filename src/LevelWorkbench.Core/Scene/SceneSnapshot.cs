namespace LevelWorkbench.Core;

public sealed class ProxyInstanceState
{
    public required ProxyDefinition Definition { get; init; }
    public required Transform3 Transform { get; init; }
}

public sealed class SceneSnapshot
{
    private readonly Dictionary<string, ProxyInstanceState> _byId;
    private readonly Dictionary<string, ProxyInstanceState> _byStableId;
    private readonly Dictionary<string, LayerDefinition> _layers;

    private SceneSnapshot(SceneDocument scene, IReadOnlyList<ProxyInstanceState> proxies)
    {
        Scene = scene;
        Proxies = proxies;
        _byId = proxies.ToDictionary(p => p.Definition.Id, StringComparer.Ordinal);
        _byStableId = proxies.ToDictionary(p => p.Definition.StableId, StringComparer.Ordinal);
        _layers = scene.Layers.ToDictionary(l => l.Id, StringComparer.Ordinal);
    }

    public SceneDocument Scene { get; }

    public IReadOnlyList<ProxyInstanceState> Proxies { get; }

    public ProxyInstanceState ById(string id) => _byId[id];

    public bool TryGetById(string id, out ProxyInstanceState instance) => _byId.TryGetValue(id, out instance!);

    public bool TryGetByStableId(string stableId, out ProxyInstanceState instance) => _byStableId.TryGetValue(stableId, out instance!);

    public LayerDefinition Layer(string id) => _layers[id];

    public static SceneSnapshot FromDocument(SceneDocument scene) => new(
        scene,
        scene.Proxies.Select(proxy => new ProxyInstanceState
        {
            Definition = proxy,
            Transform = proxy.Transform
        }).ToArray());

    public static SceneSnapshot FromPoses(SceneDocument scene, IReadOnlyDictionary<string, Transform3> poses) => new(
        scene,
        scene.Proxies.Select(proxy => new ProxyInstanceState
        {
            Definition = proxy,
            Transform = poses.TryGetValue(proxy.Id, out var pose) ? pose : proxy.Transform
        }).ToArray());

    public string ComputeDigest()
    {
        var root = new System.Text.Json.Nodes.JsonObject();
        SceneDigest.AddSceneFields(root, Scene, Proxies);
        return Sha256Util.HashWithPrefix(CanonicalJson.Serialize(root));
    }
}

public static class SceneDigest
{
    public static string Compute(SceneDocument scene) => Sha256Util.HashWithPrefix(CanonicalJson.Serialize(Build(scene, null)));

    public static System.Text.Json.Nodes.JsonObject Build(SceneDocument scene, IReadOnlyList<ProxyInstanceState>? instances)
    {
        var root = new System.Text.Json.Nodes.JsonObject();
        AddSceneFields(root, scene, instances);
        return root;
    }

    internal static void AddSceneFields(
        System.Text.Json.Nodes.JsonObject root,
        SceneDocument scene,
        IReadOnlyList<ProxyInstanceState>? instances)
    {
        root["schema"] = scene.Schema;
        root["schemaVersion"] = scene.SchemaVersion;
        root["unit"] = scene.Unit;
        root["coordinateToleranceMeters"] = scene.CoordinateToleranceMeters;
        root["memoryBudgetBytes"] = scene.MemoryBudgetBytes;
        root["concurrencyLimit"] = scene.ConcurrencyLimit;
        root["viewport"] = new System.Text.Json.Nodes.JsonObject
        {
            ["width"] = scene.Viewport.Width,
            ["height"] = scene.Viewport.Height
        };
        root["camera"] = new System.Text.Json.Nodes.JsonObject
        {
            ["positionMeters"] = VecNode(scene.Camera.PositionMeters),
            ["targetMeters"] = VecNode(scene.Camera.TargetMeters),
            ["up"] = VecNode(scene.Camera.Up),
            ["verticalFovDegrees"] = scene.Camera.VerticalFovDegrees,
            ["nearMeters"] = scene.Camera.NearMeters,
            ["farMeters"] = scene.Camera.FarMeters
        };
        root["layers"] = new System.Text.Json.Nodes.JsonArray(scene.Layers.Select(layer => (System.Text.Json.Nodes.JsonNode?)new System.Text.Json.Nodes.JsonObject
        {
            ["id"] = layer.Id,
            ["name"] = layer.Name,
            ["visible"] = layer.Visible,
            ["pickable"] = layer.Pickable,
            ["pickFilter"] = layer.PickFilter
        }).ToArray());

        var proxyStates = instances is null
            ? scene.Proxies.Select(p => new ProxyInstanceState { Definition = p, Transform = p.Transform }).ToArray()
            : instances.ToArray();
        root["proxies"] = new System.Text.Json.Nodes.JsonArray(proxyStates
            .OrderBy(p => p.Definition.Id, StringComparer.Ordinal)
            .Select(p => (System.Text.Json.Nodes.JsonNode?)new System.Text.Json.Nodes.JsonObject
            {
                ["id"] = p.Definition.Id,
                ["proxyNumber"] = p.Definition.ProxyNumber,
                ["stableId"] = p.Definition.StableId,
                ["layerId"] = p.Definition.LayerId,
                ["visible"] = p.Definition.Visible,
                ["pickable"] = p.Definition.Pickable,
                ["pickFilter"] = p.Definition.PickFilter,
                ["snapshotDigest"] = p.Definition.Geometry.SnapshotDigest,
                ["transform"] = new System.Text.Json.Nodes.JsonObject
                {
                    ["positionMeters"] = VecNode(p.Transform.PositionMeters),
                    ["rotationEulerDegrees"] = VecNode(p.Transform.RotationEulerDegrees),
                    ["scale"] = VecNode(p.Transform.Scale)
                }
            }).ToArray());
    }

    public static System.Text.Json.Nodes.JsonArray VecNode(Vec3 v) => new(
        System.Text.Json.Nodes.JsonValue.Create(v.X),
        System.Text.Json.Nodes.JsonValue.Create(v.Y),
        System.Text.Json.Nodes.JsonValue.Create(v.Z));
}
