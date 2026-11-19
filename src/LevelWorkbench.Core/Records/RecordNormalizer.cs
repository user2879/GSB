using System.Text.Json.Nodes;

namespace LevelWorkbench.Core;

public sealed class NormalizedPickRecord
{
    public required RawPickRecord Raw { get; init; }
    public required int Version { get; init; }
    public required int RecordIndex { get; init; }
    public required string RecordId { get; init; }
    public required ProxyDefinition Target { get; init; }
    public required int FaceIndex { get; init; }
    public required Vec3 HitPointMeters { get; init; }
    public required Vec3 FaceNormal { get; init; }
    public required double ScreenDistanceMeters { get; init; }
    public required Vec2? ScreenPoint { get; init; }
    public required ViewportTransform ViewportTransform { get; init; }
    public required ViewportDefinition Viewport { get; init; }
    public required string LayerId { get; init; }
    public required bool LayerVisible { get; init; }
    public required bool Pickable { get; init; }
    public required string PickFilter { get; init; }
    public required string SourceUnit { get; init; }
    public string? SnapshotDigest { get; init; }
}

public sealed class RecordNormalizer
{
    public IReadOnlyList<NormalizedPickRecord> NormalizeAll(SceneDocument scene, LoadedRecordSet records)
    {
        var result = new List<NormalizedPickRecord>(records.Records.Count);
        var ids = new HashSet<string>(StringComparer.Ordinal);
        foreach (var raw in records.Records)
        {
            var normalized = Normalize(scene, raw);
            if (!ids.Add(normalized.RecordId))
            {
                throw new WorkbenchException(ErrorCodes.Ident, "Record id must be unique across the loaded collection.", new JsonObject
                {
                    ["recordIndex"] = normalized.RecordIndex,
                    ["recordId"] = normalized.RecordId,
                    ["field"] = "recordId"
                });
            }

            result.Add(normalized);
        }

        return result;
    }

    public NormalizedPickRecord Normalize(SceneDocument scene, RawPickRecord raw)
    {
        return raw.Version switch
        {
            RecordVersions.Legacy => NormalizeLegacy(scene, raw),
            RecordVersions.StableIdentity => NormalizeModern(scene, raw, hasViewport: false),
            RecordVersions.Current => NormalizeModern(scene, raw, hasViewport: true),
            _ => throw new WorkbenchException(ErrorCodes.Version, "Unknown record version.", new JsonObject
            {
                ["recordIndex"] = raw.Index,
                ["version"] = raw.Version
            })
        };
    }

    private static NormalizedPickRecord NormalizeLegacy(SceneDocument scene, RawPickRecord raw)
    {
        var node = raw.Node;
        var recordId = RecordId(node, raw.Index);
        var unit = JsonRead.RequireString(node, "unit", "$");
        if (!string.Equals(unit, "centimeter", StringComparison.Ordinal))
        {
            throw UnitError(raw, unit, "centimeter", "hitPointCm");
        }

        if (!node.ContainsKey("hitPointCm"))
        {
            throw UnitError(raw, unit, "centimeter/hitPointCm", node.ContainsKey("hitPointMeters") ? "hitPointMeters" : "hitPointCm");
        }

        var proxyNumber = JsonRead.RequireInt32(node, "proxyNumber", "$");
        var faceIndex = JsonRead.RequireInt32(node, "faceIndex", "$");
        var pointCm = JsonRead.RequireVec3(node, "hitPointCm", "$", ErrorCodes.Unit);
        var digest = JsonRead.RequireString(node, "snapshotDigest", "$");
        var byDigest = scene.Proxies.Where(p => string.Equals(p.Geometry.SnapshotDigest, digest, StringComparison.Ordinal)).ToArray();
        if (byDigest.Length == 0)
        {
            var numberCandidates = scene.Proxies.Where(p => p.ProxyNumber == proxyNumber).Select(ProxyCandidate).ToArray();
            throw new WorkbenchException(ErrorCodes.Ident, "Legacy snapshot digest does not match any current proxy geometry.", new JsonObject
            {
                ["recordIndex"] = raw.Index,
                ["version"] = raw.Version,
                ["recordId"] = recordId,
                ["field"] = "snapshotDigest",
                ["snapshotDigest"] = digest,
                ["declaredProxyNumber"] = proxyNumber,
                ["candidates"] = new JsonArray(numberCandidates.Select(c => (JsonNode?)c).ToArray())
            });
        }

        if (byDigest.Length > 1)
        {
            throw new WorkbenchException(ErrorCodes.Ident, "Legacy snapshot digest is ambiguous; binding by proxy number is forbidden.", new JsonObject
            {
                ["recordIndex"] = raw.Index,
                ["version"] = raw.Version,
                ["recordId"] = recordId,
                ["field"] = "snapshotDigest",
                ["snapshotDigest"] = digest,
                ["candidates"] = new JsonArray(byDigest.Select(p => (JsonNode?)ProxyCandidate(p)).ToArray())
            });
        }

        var target = byDigest[0];
        if (target.ProxyNumber != proxyNumber)
        {
            throw new WorkbenchException(ErrorCodes.Ident, "Legacy snapshot digest resolves to a proxy with a different proxy number.", new JsonObject
            {
                ["recordIndex"] = raw.Index,
                ["version"] = raw.Version,
                ["recordId"] = recordId,
                ["field"] = "proxyNumber",
                ["declaredProxyNumber"] = proxyNumber,
                ["boundProxyNumber"] = target.ProxyNumber,
                ["candidates"] = new JsonArray(ProxyCandidate(target))
            });
        }

        if (faceIndex < 0 || faceIndex >= target.Geometry.TriangleCount)
        {
            throw new WorkbenchException(ErrorCodes.Geometry, "Record face index is outside the target proxy geometry.", new JsonObject
            {
                ["recordIndex"] = raw.Index,
                ["recordId"] = recordId,
                ["field"] = "faceIndex",
                ["faceIndex"] = faceIndex,
                ["triangleCount"] = target.Geometry.TriangleCount
            });
        }

        var hitPointMeters = pointCm / 100.0;
        var normal = FaceNormalTowardCamera(scene, target, faceIndex, hitPointMeters);
        var viewport = scene.Viewport;
        var screenPoint = ProjectPoint(scene, hitPointMeters, viewport, ViewportTransform.Identity);
        var layer = scene.LayerById(target.LayerId);
        return new NormalizedPickRecord
        {
            Raw = raw,
            Version = raw.Version,
            RecordIndex = raw.Index,
            RecordId = recordId,
            Target = target,
            FaceIndex = faceIndex,
            HitPointMeters = hitPointMeters,
            FaceNormal = normal,
            ScreenDistanceMeters = Vec3.Distance(scene.Camera.PositionMeters, hitPointMeters),
            ScreenPoint = screenPoint,
            ViewportTransform = ViewportTransform.Identity,
            Viewport = viewport,
            LayerId = target.LayerId,
            LayerVisible = layer.Visible,
            Pickable = layer.Pickable && target.Pickable,
            PickFilter = target.PickFilter,
            SourceUnit = unit,
            SnapshotDigest = digest
        };
    }

    private static NormalizedPickRecord NormalizeModern(SceneDocument scene, RawPickRecord raw, bool hasViewport)
    {
        var node = raw.Node;
        var recordId = RecordId(node, raw.Index);
        var unit = JsonRead.RequireString(node, "unit", "$");
        if (!string.Equals(unit, "meter", StringComparison.Ordinal))
        {
            throw UnitError(raw, unit, "meter", "hitPointMeters");
        }

        if (!node.ContainsKey("hitPointMeters"))
        {
            throw UnitError(raw, unit, "meter/hitPointMeters", node.ContainsKey("hitPointCm") ? "hitPointCm" : "hitPointMeters");
        }

        var stableId = JsonRead.RequireString(node, "stableId", "$");
        var target = scene.Proxies.FirstOrDefault(p => string.Equals(p.StableId, stableId, StringComparison.Ordinal));
        if (target is null)
        {
            throw new WorkbenchException(ErrorCodes.Ident, "Stable object id does not resolve to a current proxy.", new JsonObject
            {
                ["recordIndex"] = raw.Index,
                ["version"] = raw.Version,
                ["recordId"] = recordId,
                ["field"] = "stableId",
                ["stableId"] = stableId,
                ["candidates"] = new JsonArray()
            });
        }

        if (node.ContainsKey("proxyNumber"))
        {
            var declaredNumber = JsonRead.RequireInt32(node, "proxyNumber", "$");
            if (declaredNumber != target.ProxyNumber)
            {
                throw new WorkbenchException(ErrorCodes.Ident, "Record proxyNumber conflicts with its stableId.", new JsonObject
                {
                    ["recordIndex"] = raw.Index,
                    ["recordId"] = recordId,
                    ["field"] = "proxyNumber",
                    ["stableId"] = stableId,
                    ["declaredProxyNumber"] = declaredNumber,
                    ["boundProxyNumber"] = target.ProxyNumber,
                    ["candidates"] = new JsonArray(ProxyCandidate(target))
                });
            }
        }

        var hitPoint = JsonRead.RequireVec3(node, "hitPointMeters", "$");
        var storedNormal = JsonRead.RequireVec3(node, "faceNormal", "$");
        if (storedNormal.LengthSquared <= 1e-20)
        {
            throw new WorkbenchException(ErrorCodes.Geometry, "Record faceNormal must be non-zero.", new JsonObject
            {
                ["recordIndex"] = raw.Index,
                ["recordId"] = recordId,
                ["field"] = "faceNormal"
            });
        }

        var faceNormal = storedNormal.Normalized();
        var faceIndex = node.ContainsKey("faceIndex")
            ? JsonRead.RequireInt32(node, "faceIndex", "$")
            : DeriveFaceIndex(scene, target, hitPoint, faceNormal, raw, recordId);
        if (faceIndex < 0 || faceIndex >= target.Geometry.TriangleCount)
        {
            throw new WorkbenchException(ErrorCodes.Geometry, "Record face index is outside the target proxy geometry.", new JsonObject
            {
                ["recordIndex"] = raw.Index,
                ["recordId"] = recordId,
                ["field"] = "faceIndex",
                ["faceIndex"] = faceIndex,
                ["triangleCount"] = target.Geometry.TriangleCount
            });
        }

        var screenDistance = JsonRead.RequireFiniteDouble(node, "screenDistanceMeters", "$");
        if (screenDistance <= 0)
        {
            throw new WorkbenchException(ErrorCodes.Schema, "screenDistanceMeters must be positive.", new JsonObject
            {
                ["recordIndex"] = raw.Index,
                ["recordId"] = recordId,
                ["field"] = "screenDistanceMeters"
            });
        }

        var layerVisible = JsonRead.RequireBool(node, "layerVisible", "$");
        var pickable = JsonRead.RequireBool(node, "pickable", "$");
        var pickFilter = JsonRead.RequireString(node, "pickFilter", "$");
        var layerId = JsonRead.OptionalString(node, "layerId") ?? target.LayerId;
        if (!string.Equals(layerId, target.LayerId, StringComparison.Ordinal))
        {
            throw new WorkbenchException(ErrorCodes.Ident, "Record layerId conflicts with the resolved proxy layer.", new JsonObject
            {
                ["recordIndex"] = raw.Index,
                ["recordId"] = recordId,
                ["field"] = "layerId",
                ["recordLayerId"] = layerId,
                ["proxyLayerId"] = target.LayerId
            });
        }

        var viewport = hasViewport ? ReadViewport(node, raw, recordId) : scene.Viewport;
        var viewportTransform = hasViewport ? ReadViewportTransform(node, raw, recordId) : ViewportTransform.Identity;
        Vec2? screenPoint = null;
        if (node["screenPoint"] is JsonObject pointObj)
        {
            var x = JsonRead.RequireFiniteDouble(pointObj, "x", "$.screenPoint");
            var y = JsonRead.RequireFiniteDouble(pointObj, "y", "$.screenPoint");
            screenPoint = new Vec2(x, y);
        }
        else
        {
            screenPoint = ProjectPoint(scene, hitPoint, viewport, viewportTransform);
        }

        return new NormalizedPickRecord
        {
            Raw = raw,
            Version = raw.Version,
            RecordIndex = raw.Index,
            RecordId = recordId,
            Target = target,
            FaceIndex = faceIndex,
            HitPointMeters = hitPoint,
            FaceNormal = faceNormal,
            ScreenDistanceMeters = screenDistance,
            ScreenPoint = screenPoint,
            ViewportTransform = viewportTransform,
            Viewport = viewport,
            LayerId = target.LayerId,
            LayerVisible = layerVisible,
            Pickable = pickable,
            PickFilter = pickFilter,
            SourceUnit = unit
        };
    }

    private static int DeriveFaceIndex(
        SceneDocument scene,
        ProxyDefinition target,
        Vec3 point,
        Vec3 normal,
        RawPickRecord raw,
        string recordId)
    {
        var candidates = new List<int>();
        var tolerance = System.Math.Max(scene.CoordinateToleranceMeters, 1e-7);
        for (var face = 0; face < target.Geometry.TriangleCount; face++)
        {
            var triangle = target.Geometry.TriangleWorld(face, target.Transform);
            var unitNormal = triangle.UnitNormal();
            if (unitNormal.LengthSquared <= 1e-20)
            {
                continue;
            }

            var planeDistance = System.Math.Abs(Vec3.Dot(normal, point - triangle.A));
            var normalAgreement = System.Math.Abs(Vec3.Dot(unitNormal, normal));
            if (planeDistance <= tolerance && normalAgreement >= 1 - 1e-6 && PointInTriangle(point, triangle, tolerance))
            {
                candidates.Add(face);
            }
        }

        if (candidates.Count == 1)
        {
            return candidates[0];
        }

        throw new WorkbenchException(candidates.Count == 0 ? ErrorCodes.Replay : ErrorCodes.Ident,
            "Record omits faceIndex and its hit point/normal do not identify exactly one face.", new JsonObject
            {
                ["recordIndex"] = raw.Index,
                ["recordId"] = recordId,
                ["field"] = "faceIndex",
                ["candidates"] = new JsonArray(candidates.Select(i => (JsonNode?)JsonValue.Create(i)).ToArray())
            });
    }

    private static ViewportDefinition ReadViewport(JsonObject node, RawPickRecord raw, string recordId)
    {
        var viewportObj = node["viewport"] as JsonObject
                          ?? throw new WorkbenchException(ErrorCodes.Schema, "Version 3 record requires viewport.", new JsonObject
                          {
                              ["recordIndex"] = raw.Index,
                              ["recordId"] = recordId,
                              ["field"] = "viewport"
                          });
        var width = JsonRead.RequireInt32(viewportObj, "width", "$.viewport");
        var height = JsonRead.RequireInt32(viewportObj, "height", "$.viewport");
        if (width <= 0 || height <= 0)
        {
            throw new WorkbenchException(ErrorCodes.Schema, "Record viewport dimensions must be positive.", new JsonObject
            {
                ["recordIndex"] = raw.Index,
                ["recordId"] = recordId,
                ["field"] = "viewport"
            });
        }

        return new ViewportDefinition { Width = width, Height = height };
    }

    private static ViewportTransform ReadViewportTransform(JsonObject node, RawPickRecord raw, string recordId)
    {
        var viewportObj = (JsonObject)node["viewport"]!;
        var zoom = JsonRead.RequireFiniteDouble(viewportObj, "zoom", "$.viewport");
        var panX = JsonRead.RequireFiniteDouble(viewportObj, "panXPixels", "$.viewport");
        var panY = JsonRead.RequireFiniteDouble(viewportObj, "panYPixels", "$.viewport");
        if (zoom <= 0)
        {
            throw new WorkbenchException(ErrorCodes.Schema, "Record viewport zoom must be positive.", new JsonObject
            {
                ["recordIndex"] = raw.Index,
                ["recordId"] = recordId,
                ["field"] = "viewport.zoom"
            });
        }

        return new ViewportTransform(zoom, panX, panY);
    }

    private static Vec3 FaceNormalTowardCamera(SceneDocument scene, ProxyDefinition target, int faceIndex, Vec3 hitPoint)
    {
        var triangle = target.Geometry.TriangleWorld(faceIndex, target.Transform);
        var normal = triangle.UnitNormal();
        if (normal.LengthSquared <= 1e-20)
        {
            throw new WorkbenchException(ErrorCodes.Geometry, "Record targets a degenerate triangle.", new JsonObject
            {
                ["proxyId"] = target.Id,
                ["faceIndex"] = faceIndex
            });
        }

        return Vec3.Dot(normal, hitPoint - scene.Camera.PositionMeters) > 0 ? -normal : normal;
    }

    private static Vec2? ProjectPoint(SceneDocument scene, Vec3 point, ViewportDefinition viewport, ViewportTransform transform)
    {
        var basis = CameraBasis.FromCamera(scene.Camera);
        return basis.ProjectWorld(point, viewport, transform, scene.Camera.NearMeters)?.Screen;
    }

    private static bool PointInTriangle(Vec3 point, Triangle3 triangle, double tolerance)
    {
        var v0 = triangle.B - triangle.A;
        var v1 = triangle.C - triangle.A;
        var v2 = point - triangle.A;
        var dot00 = Vec3.Dot(v0, v0);
        var dot01 = Vec3.Dot(v0, v1);
        var dot02 = Vec3.Dot(v0, v2);
        var dot11 = Vec3.Dot(v1, v1);
        var dot12 = Vec3.Dot(v1, v2);
        var inv = dot00 * dot11 - dot01 * dot01;
        if (System.Math.Abs(inv) <= 1e-20)
        {
            return false;
        }

        var u = (dot11 * dot02 - dot01 * dot12) / inv;
        var v = (dot00 * dot12 - dot01 * dot02) / inv;
        var epsilon = tolerance <= 0 ? 1e-7 : System.Math.Min(1e-3, tolerance);
        return u >= -epsilon && v >= -epsilon && u + v <= 1 + epsilon;
    }

    private static string RecordId(JsonObject node, int index) =>
        JsonRead.OptionalString(node, "recordId") ?? FormattableString.Invariant($"record-{index:D4}");

    private static WorkbenchException UnitError(RawPickRecord raw, string actualUnit, string expectedUnit, string field) => new(
        ErrorCodes.Unit,
        "Record unit declaration does not match its coordinate fields or the expected version format.",
        new JsonObject
        {
            ["recordIndex"] = raw.Index,
            ["version"] = raw.Version,
            ["field"] = field,
            ["expectedUnit"] = expectedUnit,
            ["actualUnit"] = actualUnit
        });

    private static JsonObject ProxyCandidate(ProxyDefinition proxy) => new()
    {
        ["id"] = proxy.Id,
        ["stableId"] = proxy.StableId,
        ["proxyNumber"] = proxy.ProxyNumber,
        ["snapshotDigest"] = proxy.Geometry.SnapshotDigest
    };
}
