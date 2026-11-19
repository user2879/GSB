using System.Text.Json.Nodes;

namespace LevelWorkbench.Core;

public enum PickScriptStepKind
{
    Pick,
    Select,
    Box,
    Drag
}

public sealed class PickScriptStep
{
    public required int Index { get; init; }
    public required PickScriptStepKind Kind { get; init; }
    public Vec2? Screen { get; init; }
    public ScreenRect? Rect { get; init; }
    public Vec2? DragStart { get; init; }
    public Vec2? DragEnd { get; init; }
    public bool Additive { get; init; }
    public string? Filter { get; init; }
}

public sealed class PickScriptDocument
{
    public required string Name { get; init; }
    public required ViewportTransform ViewportTransform { get; init; }
    public required IReadOnlyList<PickScriptStep> Steps { get; init; }
    public required string CanonicalDigest { get; init; }
    public required JsonObject RawRoot { get; init; }
}

public static class PickScriptLoader
{
    public const string ExpectedSchema = "level-workbench.pick-script";
    public const int ExpectedVersion = 1;

    public static PickScriptDocument Load(string path)
    {
        if (!File.Exists(path))
        {
            throw new WorkbenchException(ErrorCodes.Schema, "Pick script file does not exist.", new JsonObject { ["path"] = path });
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
            throw new WorkbenchException(ErrorCodes.Schema, "Pick script file could not be read.", new JsonObject
            {
                ["path"] = path,
                ["reason"] = ex.Message
            }, ex);
        }
    }

    public static PickScriptDocument Parse(string json, string source)
    {
        var root = JsonRead.ParseObject(json, "pick-script");
        var schema = JsonRead.RequireString(root, "schema", "$");
        if (schema != ExpectedSchema)
        {
            throw new WorkbenchException(ErrorCodes.Schema, "Unsupported pick script schema.", new JsonObject
            {
                ["source"] = source,
                ["expected"] = ExpectedSchema,
                ["actual"] = schema
            });
        }

        var version = JsonRead.RequireInt32(root, "version", "$");
        if (version != ExpectedVersion)
        {
            throw new WorkbenchException(ErrorCodes.Version, "Unsupported pick script version.", new JsonObject
            {
                ["source"] = source,
                ["expected"] = ExpectedVersion,
                ["actual"] = version
            });
        }

        var name = JsonRead.RequireString(root, "name", "$");
        var viewportObj = JsonRead.RequireObject(root, "viewport", "$");
        var viewportTransform = new ViewportTransform(
            JsonRead.RequireFiniteDouble(viewportObj, "zoom", "$.viewport"),
            JsonRead.RequireFiniteDouble(viewportObj, "panXPixels", "$.viewport"),
            JsonRead.RequireFiniteDouble(viewportObj, "panYPixels", "$.viewport"));
        if (!viewportTransform.IsValid)
        {
            throw new WorkbenchException(ErrorCodes.Schema, "Pick script viewport transform is invalid.", new JsonObject { ["source"] = source });
        }

        var stepsArray = JsonRead.RequireArray(root, "steps", "$");
        var steps = new List<PickScriptStep>(stepsArray.Count);
        for (var i = 0; i < stepsArray.Count; i++)
        {
            var path = $"$.steps[{i}]";
            var obj = stepsArray[i] as JsonObject
                      ?? throw JsonRead.Schema("Pick script step must be an object.", path, "step");
            var kindText = JsonRead.RequireString(obj, "kind", path);
            var additive = obj.ContainsKey("additive") && JsonRead.RequireBool(obj, "additive", path);
            var filter = JsonRead.OptionalString(obj, "filter");
            switch (kindText)
            {
                case "pick":
                    steps.Add(new PickScriptStep
                    {
                        Index = i,
                        Kind = PickScriptStepKind.Pick,
                        Screen = ReadScreen(obj, "screen", path),
                        Filter = filter
                    });
                    break;
                case "select":
                    steps.Add(new PickScriptStep
                    {
                        Index = i,
                        Kind = PickScriptStepKind.Select,
                        Screen = ReadScreen(obj, "screen", path),
                        Additive = additive,
                        Filter = filter
                    });
                    break;
                case "box":
                    var min = ReadScreen(obj, "min", path);
                    var max = ReadScreen(obj, "max", path);
                    steps.Add(new PickScriptStep
                    {
                        Index = i,
                        Kind = PickScriptStepKind.Box,
                        Rect = ScreenRect.FromPoints(min, max),
                        Additive = additive,
                        Filter = filter
                    });
                    break;
                case "drag":
                    steps.Add(new PickScriptStep
                    {
                        Index = i,
                        Kind = PickScriptStepKind.Drag,
                        DragStart = ReadScreen(obj, "start", path),
                        DragEnd = ReadScreen(obj, "end", path),
                        Filter = filter
                    });
                    break;
                default:
                    throw JsonRead.Schema("Unknown pick script step kind.", path, "kind");
            }
        }

        return new PickScriptDocument
        {
            Name = name,
            ViewportTransform = viewportTransform,
            Steps = steps,
            CanonicalDigest = Sha256Util.HashWithPrefix(CanonicalJson.Serialize(root)),
            RawRoot = root
        };
    }

    private static Vec2 ReadScreen(JsonObject obj, string property, string path)
    {
        var point = JsonRead.RequireObject(obj, property, path);
        var x = JsonRead.RequireFiniteDouble(point, "x", path + "." + property);
        var y = JsonRead.RequireFiniteDouble(point, "y", path + "." + property);
        return new Vec2(x, y);
    }
}
