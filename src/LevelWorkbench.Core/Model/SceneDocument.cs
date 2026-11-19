namespace LevelWorkbench.Core;

public sealed class ViewportDefinition
{
    public required int Width { get; init; }
    public required int Height { get; init; }
}

public sealed class CameraDefinition
{
    public required Vec3 PositionMeters { get; init; }
    public required Vec3 TargetMeters { get; init; }
    public required Vec3 Up { get; init; }
    public required double VerticalFovDegrees { get; init; }
    public required double NearMeters { get; init; }
    public required double FarMeters { get; init; }
}

public sealed class LayerDefinition
{
    public required string Id { get; init; }
    public required string Name { get; init; }
    public required bool Visible { get; init; }
    public required bool Pickable { get; init; }
    public required string PickFilter { get; init; }
}

public sealed class ProxyGeometryDefinition
{
    private readonly Vec3[] _vertices;
    private readonly int[] _indices;

    public ProxyGeometryDefinition(Vec3[] vertices, int[] indices)
    {
        _vertices = vertices;
        _indices = indices;
        LocalBounds = Bounds3.FromPoints(vertices);
        SnapshotDigest = GeometryDigest.Compute(vertices, indices);
    }

    public IReadOnlyList<Vec3> Vertices => _vertices;

    public IReadOnlyList<int> Indices => _indices;

    public int TriangleCount => _indices.Length / 3;

    public Bounds3 LocalBounds { get; }

    public string SnapshotDigest { get; }

    public Triangle3 TriangleLocal(int faceIndex)
    {
        var offset = faceIndex * 3;
        return new Triangle3(_vertices[_indices[offset]], _vertices[_indices[offset + 1]], _vertices[_indices[offset + 2]]);
    }

    public Triangle3 TriangleWorld(int faceIndex, Transform3 transform)
    {
        var local = TriangleLocal(faceIndex);
        return new Triangle3(
            transform.TransformPoint(local.A),
            transform.TransformPoint(local.B),
            transform.TransformPoint(local.C));
    }

    public long EstimatedTriangleBufferBytes => (long)TriangleCount * 3L * 24L;
}

public sealed class ProxyDefinition
{
    public required string Id { get; init; }
    public required int ProxyNumber { get; init; }
    public required string StableId { get; init; }
    public required string LayerId { get; init; }
    public required bool Visible { get; init; }
    public required bool Pickable { get; init; }
    public required string PickFilter { get; init; }
    public required Transform3 Transform { get; init; }
    public required ProxyGeometryDefinition Geometry { get; init; }
}

public sealed class SceneDocument
{
    public required string Schema { get; init; }
    public required int SchemaVersion { get; init; }
    public required string Name { get; init; }
    public required string Unit { get; init; }
    public required double CoordinateToleranceMeters { get; init; }
    public required int ConcurrencyLimit { get; init; }
    public required long MemoryBudgetBytes { get; init; }
    public required int ChunkSize { get; init; }
    public required ViewportDefinition Viewport { get; init; }
    public required CameraDefinition Camera { get; init; }
    public required IReadOnlyList<LayerDefinition> Layers { get; init; }
    public required IReadOnlyList<ProxyDefinition> Proxies { get; init; }

    public LayerDefinition LayerById(string id) => Layers.First(layer => layer.Id == id);

    public ProxyDefinition ProxyById(string id) => Proxies.First(proxy => proxy.Id == id);

    public ProxyDefinition ProxyByStableId(string stableId) => Proxies.First(proxy => proxy.StableId == stableId);

    public ProxyDefinition ProxyByNumber(int number) => Proxies.First(proxy => proxy.ProxyNumber == number);
}
