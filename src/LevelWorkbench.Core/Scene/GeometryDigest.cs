using System.Text.Json.Nodes;

namespace LevelWorkbench.Core;

public static class GeometryDigest
{
    public static string Compute(IReadOnlyList<Vec3> vertices, IReadOnlyList<int> indices)
    {
        var root = new JsonObject
        {
            ["indices"] = new JsonArray(indices.Select(i => (JsonNode?)JsonValue.Create(i)).ToArray()),
            ["verticesMeters"] = new JsonArray(vertices
                .Select(v => (JsonNode?)new JsonArray(JsonValue.Create(v.X), JsonValue.Create(v.Y), JsonValue.Create(v.Z)))
                .ToArray())
        };

        return Sha256Util.HashWithPrefix(CanonicalJson.Serialize(root));
    }
}
