using System.Text.Json.Nodes;

namespace LevelWorkbench.Core;

public static class SceneLoader
{
    public const string ExpectedSchema = "level-workbench.scene";
    public const string ExpectedUnit = "meter";
    public const int ExpectedSchemaVersion = 1;

    public static SceneDocument Load(string path)
    {
        if (!File.Exists(path))
        {
            throw new WorkbenchException(ErrorCodes.Schema, "Scene file does not exist.", new JsonObject { ["path"] = path });
        }

        try
        {
            return Parse(File.ReadAllText(path), path);
        }
        catch (WorkbenchException)
        {
            throw;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            throw new WorkbenchException(ErrorCodes.Schema, "Scene file could not be read.", new JsonObject
            {
                ["path"] = path,
                ["reason"] = ex.Message
            }, ex);
        }
    }

    public static SceneDocument Parse(string json, string source = "<memory>")
    {
        var root = JsonRead.ParseObject(json, "scene");
        var schema = JsonRead.RequireString(root, "schema", "$");
        if (schema != ExpectedSchema)
        {
            throw new WorkbenchException(ErrorCodes.Schema, "Unsupported scene schema.", new JsonObject
            {
                ["source"] = source,
                ["expected"] = ExpectedSchema,
                ["actual"] = schema
            });
        }

        var schemaVersion = JsonRead.RequireInt32(root, "schemaVersion", "$");
        if (schemaVersion != ExpectedSchemaVersion)
        {
            throw new WorkbenchException(ErrorCodes.Version, "Unsupported scene schema version.", new JsonObject
            {
                ["source"] = source,
                ["known"] = new JsonArray(ExpectedSchemaVersion),
                ["actual"] = schemaVersion
            });
        }

        var unit = JsonRead.RequireString(root, "unit", "$");
        if (!string.Equals(unit, ExpectedUnit, StringComparison.Ordinal))
        {
            throw new WorkbenchException(ErrorCodes.Unit, "Scene unit must be meter for verticesMeters and positionMeters fields.", new JsonObject
            {
                ["source"] = source,
                ["expected"] = ExpectedUnit,
                ["actual"] = unit
            });
        }

        var name = JsonRead.RequireString(root, "name", "$");
        var tolerance = JsonRead.RequireFiniteDouble(root, "coordinateToleranceMeters", "$");
        if (tolerance <= 0)
        {
            throw JsonRead.Schema("coordinateToleranceMeters must be greater than zero.", "$", "coordinateToleranceMeters");
        }

        var concurrency = JsonRead.RequireInt32(root, "concurrencyLimit", "$");
        if (concurrency is < 1 or > 1024)
        {
            throw JsonRead.Schema("concurrencyLimit must be between 1 and 1024.", "$", "concurrencyLimit");
        }

        var memoryBudget = JsonRead.RequireInt64(root, "memoryBudgetBytes", "$");
        if (memoryBudget <= 0)
        {
            throw JsonRead.Schema("memoryBudgetBytes must be greater than zero.", "$", "memoryBudgetBytes");
        }

        var chunkSize = JsonRead.RequireInt32(root, "chunkSize", "$");
        if (chunkSize <= 0)
        {
            throw JsonRead.Schema("chunkSize must be greater than zero.", "$", "chunkSize");
        }

        var viewportObj = JsonRead.RequireObject(root, "viewport", "$");
        var viewport = new ViewportDefinition
        {
            Width = RequirePositiveInt(viewportObj, "width", "$.viewport"),
            Height = RequirePositiveInt(viewportObj, "height", "$.viewport")
        };

        var cameraObj = JsonRead.RequireObject(root, "camera", "$");
        var camera = new CameraDefinition
        {
            PositionMeters = JsonRead.RequireVec3(cameraObj, "positionMeters", "$.camera"),
            TargetMeters = JsonRead.RequireVec3(cameraObj, "targetMeters", "$.camera"),
            Up = JsonRead.RequireVec3(cameraObj, "up", "$.camera"),
            VerticalFovDegrees = JsonRead.RequireFiniteDouble(cameraObj, "verticalFovDegrees", "$.camera"),
            NearMeters = JsonRead.RequireFiniteDouble(cameraObj, "nearMeters", "$.camera"),
            FarMeters = JsonRead.RequireFiniteDouble(cameraObj, "farMeters", "$.camera")
        };
        ValidateCamera(camera);

        var layers = ParseLayers(root);
        var layerIds = layers.Select(l => l.Id).ToHashSet(StringComparer.Ordinal);
        var proxies = ParseProxies(root, layerIds);

        return new SceneDocument
        {
            Schema = schema,
            SchemaVersion = schemaVersion,
            Name = name,
            Unit = unit,
            CoordinateToleranceMeters = tolerance,
            ConcurrencyLimit = concurrency,
            MemoryBudgetBytes = memoryBudget,
            ChunkSize = chunkSize,
            Viewport = viewport,
            Camera = camera,
            Layers = layers,
            Proxies = proxies
        };
    }

    private static int RequirePositiveInt(JsonObject parent, string property, string path)
    {
        var value = JsonRead.RequireInt32(parent, property, path);
        if (value <= 0)
        {
            throw JsonRead.Schema($"{path}.{property} must be positive.", path, property);
        }

        return value;
    }

    private static void ValidateCamera(CameraDefinition camera)
    {
        if (!camera.PositionMeters.IsFinite || !camera.TargetMeters.IsFinite || !camera.Up.IsFinite)
        {
            throw JsonRead.Schema("Camera vectors must be finite.", "$.camera", "positionMeters");
        }

        if ((camera.TargetMeters - camera.PositionMeters).LengthSquared <= 1e-20)
        {
            throw JsonRead.Schema("Camera position and target must differ.", "$.camera", "targetMeters");
        }

        if (camera.Up.LengthSquared <= 1e-20 || Vec3.Cross(camera.TargetMeters - camera.PositionMeters, camera.Up).LengthSquared <= 1e-20)
        {
            throw JsonRead.Schema("Camera up must be non-zero and not parallel to the view direction.", "$.camera", "up");
        }

        if (!(camera.VerticalFovDegrees > 0 && camera.VerticalFovDegrees < 179))
        {
            throw JsonRead.Schema("verticalFovDegrees must be in (0, 179).", "$.camera", "verticalFovDegrees");
        }

        if (!(camera.NearMeters > 0 && camera.FarMeters > camera.NearMeters))
        {
            throw JsonRead.Schema("Camera clipping requires 0 < nearMeters < farMeters.", "$.camera", "nearMeters");
        }
    }

    private static IReadOnlyList<LayerDefinition> ParseLayers(JsonObject root)
    {
        var array = JsonRead.RequireArray(root, "layers", "$");
        var result = new List<LayerDefinition>(array.Count);
        var ids = new HashSet<string>(StringComparer.Ordinal);
        for (var i = 0; i < array.Count; i++)
        {
            var path = $"$.layers[{i}]";
            if (array[i] is not JsonObject obj)
            {
                throw JsonRead.Schema("Layer entry must be an object.", path, "layer");
            }

            var id = JsonRead.RequireString(obj, "id", path);
            if (!ids.Add(id))
            {
                throw new WorkbenchException(ErrorCodes.Layer, "Layer id must be unique.", new JsonObject
                {
                    ["layerId"] = id,
                    ["index"] = i
                });
            }

            if (!obj.ContainsKey("visible"))
            {
                throw new WorkbenchException(ErrorCodes.Layer, "Layer visibility field is required.", new JsonObject
                {
                    ["layerId"] = id,
                    ["field"] = "visible"
                });
            }

            if (!obj.ContainsKey("pickable"))
            {
                throw new WorkbenchException(ErrorCodes.Layer, "Layer pick filter field is required.", new JsonObject
                {
                    ["layerId"] = id,
                    ["field"] = "pickable"
                });
            }

            var name = JsonRead.RequireString(obj, "name", path);
            var visible = JsonRead.RequireBool(obj, "visible", path);
            var pickable = JsonRead.RequireBool(obj, "pickable", path);
            var pickFilter = JsonRead.RequireString(obj, "pickFilter", path);
            result.Add(new LayerDefinition
            {
                Id = id,
                Name = name,
                Visible = visible,
                Pickable = pickable,
                PickFilter = pickFilter
            });
        }

        return result;
    }

    private static IReadOnlyList<ProxyDefinition> ParseProxies(JsonObject root, HashSet<string> layerIds)
    {
        var array = JsonRead.RequireArray(root, "proxies", "$");
        var result = new List<ProxyDefinition>(array.Count);
        var ids = new HashSet<string>(StringComparer.Ordinal);
        var stableIds = new HashSet<string>(StringComparer.Ordinal);
        var numbers = new HashSet<int>();

        for (var i = 0; i < array.Count; i++)
        {
            var path = $"$.proxies[{i}]";
            if (array[i] is not JsonObject obj)
            {
                throw JsonRead.Schema("Proxy entry must be an object.", path, "proxy");
            }

            var id = JsonRead.RequireString(obj, "id", path);
            var stableId = JsonRead.RequireString(obj, "stableId", path);
            var proxyNumber = JsonRead.RequireInt32(obj, "proxyNumber", path);
            if (proxyNumber <= 0)
            {
                throw JsonRead.Schema("proxyNumber must be positive.", path, "proxyNumber");
            }

            if (!ids.Add(id))
            {
                throw new WorkbenchException(ErrorCodes.Ident, "Proxy id must be unique.", new JsonObject
                {
                    ["field"] = "id",
                    ["value"] = id,
                    ["index"] = i
                });
            }

            if (!stableIds.Add(stableId))
            {
                throw new WorkbenchException(ErrorCodes.Ident, "Proxy stableId must be unique.", new JsonObject
                {
                    ["field"] = "stableId",
                    ["value"] = stableId,
                    ["index"] = i
                });
            }

            if (!numbers.Add(proxyNumber))
            {
                throw new WorkbenchException(ErrorCodes.Ident, "Proxy proxyNumber must be unique.", new JsonObject
                {
                    ["field"] = "proxyNumber",
                    ["value"] = proxyNumber,
                    ["index"] = i
                });
            }

            var layerId = JsonRead.RequireString(obj, "layerId", path);
            if (!layerIds.Contains(layerId))
            {
                throw new WorkbenchException(ErrorCodes.Layer, "Proxy references an unknown layer.", new JsonObject
                {
                    ["proxyId"] = id,
                    ["layerId"] = layerId
                });
            }

            var visible = JsonRead.RequireBool(obj, "visible", path);
            var pickable = JsonRead.RequireBool(obj, "pickable", path);
            var pickFilter = JsonRead.RequireString(obj, "pickFilter", path);
            var transformObj = JsonRead.RequireObject(obj, "transform", path);
            var transform = new Transform3(
                JsonRead.RequireVec3(transformObj, "positionMeters", path + ".transform"),
                JsonRead.RequireVec3(transformObj, "rotationEulerDegrees", path + ".transform"),
                JsonRead.RequireVec3(transformObj, "scale", path + ".transform"));
            if (transform.Scale.X == 0 || transform.Scale.Y == 0 || transform.Scale.Z == 0)
            {
                throw new WorkbenchException(ErrorCodes.Geometry, "Proxy scale components must be non-zero.", new JsonObject
                {
                    ["proxyId"] = id,
                    ["field"] = "scale"
                });
            }

            var geometryObj = JsonRead.RequireObject(obj, "geometry", path);
            var geometry = ParseGeometry(geometryObj, path + ".geometry", id);
            var declaredDigest = JsonRead.OptionalString(obj, "snapshotDigest");
            if (declaredDigest is not null && declaredDigest != geometry.SnapshotDigest)
            {
                throw new WorkbenchException(ErrorCodes.Geometry, "Declared snapshotDigest does not match the current geometry snapshot.", new JsonObject
                {
                    ["proxyId"] = id,
                    ["field"] = "snapshotDigest",
                    ["declared"] = declaredDigest,
                    ["computed"] = geometry.SnapshotDigest
                });
            }

            result.Add(new ProxyDefinition
            {
                Id = id,
                ProxyNumber = proxyNumber,
                StableId = stableId,
                LayerId = layerId,
                Visible = visible,
                Pickable = pickable,
                PickFilter = pickFilter,
                Transform = transform,
                Geometry = geometry
            });
        }

        return result;
    }

    private static ProxyGeometryDefinition ParseGeometry(JsonObject obj, string path, string proxyId)
    {
        var vertexArray = JsonRead.RequireArray(obj, "verticesMeters", path);
        var vertices = new Vec3[vertexArray.Count];
        for (var i = 0; i < vertexArray.Count; i++)
        {
            if (vertexArray[i] is not JsonArray vertex || vertex.Count != 3)
            {
                throw new WorkbenchException(ErrorCodes.Geometry, "Each vertex must be a three-element array.", new JsonObject
                {
                    ["proxyId"] = proxyId,
                    ["vertexIndex"] = i
                });
            }

            var values = new double[3];
            for (var c = 0; c < 3; c++)
            {
                if (vertex[c] is not JsonValue value || !value.TryGetValue<double>(out values[c]) || !double.IsFinite(values[c]))
                {
                    throw new WorkbenchException(ErrorCodes.Geometry, "Vertex coordinate must be finite.", new JsonObject
                    {
                        ["proxyId"] = proxyId,
                        ["vertexIndex"] = i,
                        ["component"] = c
                    });
                }
            }

            vertices[i] = new Vec3(values[0], values[1], values[2]);
        }

        var indexArray = JsonRead.RequireArray(obj, "indices", path);
        if (indexArray.Count == 0 || indexArray.Count % 3 != 0)
        {
            throw new WorkbenchException(ErrorCodes.Geometry, "Triangle indices must be non-empty and divisible by three.", new JsonObject
            {
                ["proxyId"] = proxyId,
                ["indexCount"] = indexArray.Count
            });
        }

        var indices = new int[indexArray.Count];
        for (var i = 0; i < indexArray.Count; i++)
        {
            if (indexArray[i] is not JsonValue value || !value.TryGetValue<int>(out indices[i]))
            {
                if (indexArray[i] is JsonValue doubleValue && doubleValue.TryGetValue<double>(out var d)
                    && d == System.Math.Truncate(d) && d is >= int.MinValue and <= int.MaxValue)
                {
                    indices[i] = (int)d;
                }
                else
                {
                    throw new WorkbenchException(ErrorCodes.Geometry, "Triangle index must be an integer.", new JsonObject
                    {
                        ["proxyId"] = proxyId,
                        ["indexPosition"] = i
                    });
                }
            }

            if (indices[i] < 0 || indices[i] >= vertices.Length)
            {
                throw new WorkbenchException(ErrorCodes.Geometry, "Triangle index is out of bounds.", new JsonObject
                {
                    ["proxyId"] = proxyId,
                    ["indexPosition"] = i,
                    ["index"] = indices[i],
                    ["vertexCount"] = vertices.Length
                });
            }
        }

        return new ProxyGeometryDefinition(vertices, indices);
    }
}
