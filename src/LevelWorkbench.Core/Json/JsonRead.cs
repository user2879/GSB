using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace LevelWorkbench.Core;

public static class JsonRead
{
    public static JsonObject ParseObject(string text, string kind)
    {
        try
        {
            var node = JsonNode.Parse(text, documentOptions: new JsonDocumentOptions
            {
                AllowTrailingCommas = false,
                CommentHandling = JsonCommentHandling.Disallow
            });
            return node as JsonObject
                   ?? throw new WorkbenchException(ErrorCodes.Schema, $"{kind} root must be a JSON object.");
        }
        catch (WorkbenchException)
        {
            throw;
        }
        catch (Exception ex) when (ex is JsonException or FormatException or InvalidOperationException)
        {
            throw new WorkbenchException(ErrorCodes.Json, $"Malformed JSON for {kind}.", new JsonObject
            {
                ["kind"] = kind,
                ["reason"] = ex.Message
            }, ex);
        }
    }

    public static JsonObject RequireObject(JsonObject parent, string property, string path)
    {
        var node = parent[property];
        if (node is JsonObject obj)
        {
            return obj;
        }

        throw Schema($"{path}.{property} must be an object.", path, property);
    }

    public static JsonArray RequireArray(JsonObject parent, string property, string path)
    {
        var node = parent[property];
        if (node is JsonArray array)
        {
            return array;
        }

        throw Schema($"{path}.{property} must be an array.", path, property);
    }

    public static string RequireString(JsonObject parent, string property, string path)
    {
        var node = parent[property];
        if (node is JsonValue value && value.TryGetValue<string>(out var text) && !string.IsNullOrWhiteSpace(text))
        {
            return text;
        }

        throw Schema($"{path}.{property} must be a non-empty string.", path, property);
    }

    public static string? OptionalString(JsonObject parent, string property)
    {
        var node = parent[property];
        if (node is JsonValue value && value.TryGetValue<string>(out var text))
        {
            return text;
        }

        return null;
    }

    public static bool RequireBool(JsonObject parent, string property, string path)
    {
        var node = parent[property];
        if (node is JsonValue value && value.TryGetValue<bool>(out var result))
        {
            return result;
        }

        throw Schema($"{path}.{property} must be a boolean.", path, property);
    }

    public static double RequireFiniteDouble(JsonObject parent, string property, string path, string code = ErrorCodes.Schema)
    {
        var node = parent[property];
        if (node is JsonValue value && value.TryGetValue<double>(out var result) && double.IsFinite(result))
        {
            return result;
        }

        throw new WorkbenchException(code, $"{path}.{property} must be a finite number.", new JsonObject
        {
            ["path"] = path,
            ["property"] = property
        });
    }

    public static int RequireInt32(JsonObject parent, string property, string path)
    {
        var node = parent[property];
        if (node is JsonValue value && value.TryGetValue<int>(out var result))
        {
            return result;
        }

        if (node is JsonValue dvalue && dvalue.TryGetValue<double>(out var d) && d == System.Math.Truncate(d) && d is >= int.MinValue and <= int.MaxValue)
        {
            return (int)d;
        }

        throw Schema($"{path}.{property} must be a 32-bit integer.", path, property);
    }

    public static long RequireInt64(JsonObject parent, string property, string path)
    {
        var node = parent[property];
        if (node is JsonValue value && value.TryGetValue<long>(out var result))
        {
            return result;
        }

        throw Schema($"{path}.{property} must be a 64-bit integer.", path, property);
    }

    public static Vec3 RequireVec3(JsonObject parent, string property, string path, string code = ErrorCodes.Schema)
    {
        var array = RequireArray(parent, property, path);
        if (array.Count != 3)
        {
            throw new WorkbenchException(code, $"{path}.{property} must contain exactly three numbers.", new JsonObject
            {
                ["path"] = path,
                ["property"] = property,
                ["length"] = array.Count
            });
        }

        var values = new double[3];
        for (var i = 0; i < 3; i++)
        {
            if (array[i] is not JsonValue value || !value.TryGetValue<double>(out values[i]) || !double.IsFinite(values[i]))
            {
                throw new WorkbenchException(code, $"{path}.{property}[{i}] must be a finite number.", new JsonObject
                {
                    ["path"] = path,
                    ["property"] = property,
                    ["index"] = i
                });
            }
        }

        return new Vec3(values[0], values[1], values[2]);
    }

    public static WorkbenchException Schema(string message, string path, string property, JsonObject? details = null)
    {
        details ??= new JsonObject();
        details["path"] = path;
        details["property"] = property;
        return new WorkbenchException(ErrorCodes.Schema, message, details);
    }

    public static string Invariant(double value) => value.ToString("R", CultureInfo.InvariantCulture);
}
