using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace LevelWorkbench.Core;

public static class CanonicalJson
{
    public static string Serialize(JsonNode? node)
    {
        var builder = new StringBuilder();
        WriteNode(builder, node);
        return builder.ToString();
    }

    public static string ArrayDigest(IEnumerable<JsonNode?> nodes)
    {
        var array = new JsonArray();
        foreach (var node in nodes)
        {
            array.Add(node?.DeepClone());
        }

        return Sha256Util.HashHex(Serialize(array));
    }

    private static void WriteNode(StringBuilder builder, JsonNode? node)
    {
        switch (node)
        {
            case null:
                builder.Append("null");
                break;
            case JsonObject obj:
                builder.Append('{');
                var firstProperty = true;
                foreach (var property in obj.OrderBy(p => p.Key, StringComparer.Ordinal))
                {
                    if (!firstProperty)
                    {
                        builder.Append(',');
                    }

                    firstProperty = false;
                    builder.Append(JsonSerializer.Serialize(property.Key));
                    builder.Append(':');
                    WriteNode(builder, property.Value);
                }

                builder.Append('}');
                break;
            case JsonArray array:
                builder.Append('[');
                for (var i = 0; i < array.Count; i++)
                {
                    if (i > 0)
                    {
                        builder.Append(',');
                    }

                    WriteNode(builder, array[i]);
                }

                builder.Append(']');
                break;
            case JsonValue value:
                WriteValue(builder, value);
                break;
            default:
                throw new WorkbenchException(ErrorCodes.Json, "Unsupported JSON node during canonicalization.");
        }
    }

    private static void WriteValue(StringBuilder builder, JsonValue value)
    {
        if (value.TryGetValue<bool>(out var boolean))
        {
            builder.Append(boolean ? "true" : "false");
            return;
        }

        if (value.TryGetValue<string>(out var text))
        {
            builder.Append(JsonSerializer.Serialize(text));
            return;
        }

        var raw = value.ToJsonString();
        if (double.TryParse(raw, NumberStyles.Float, CultureInfo.InvariantCulture, out var number))
        {
            if (!double.IsFinite(number))
            {
                throw new WorkbenchException(ErrorCodes.Json, "Non-finite number cannot be canonicalized.");
            }

            builder.Append(number == 0 ? "0" : number.ToString("R", CultureInfo.InvariantCulture));
            return;
        }

        throw new WorkbenchException(ErrorCodes.Json, "Unsupported JSON scalar during canonicalization.", new System.Text.Json.Nodes.JsonObject
        {
            ["raw"] = raw
        });
    }
}
