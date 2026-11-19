using System.Text.Json.Nodes;

namespace LevelWorkbench.Core;

public static class ErrorCodes
{
    public const string Json = "E_JSON";
    public const string Schema = "E_SCHEMA";
    public const string Geometry = "E_GEOMETRY";
    public const string Layer = "E_LAYER";
    public const string Version = "E_VERSION";
    public const string Unit = "E_UNIT";
    public const string Ident = "E_IDENT";
    public const string Replay = "E_REPLAY";
    public const string Write = "E_WRITE";
    public const string State = "E_STATE";
    public const string Resource = "E_RESOURCE";
    public const string Cancelled = "E_CANCELLED";
    public const string Internal = "E_INTERNAL";
}

public sealed class WorkbenchException : Exception
{
    public WorkbenchException(string code, string message, JsonObject? details = null, Exception? inner = null)
        : base(message, inner)
    {
        Code = code;
        Details = details ?? new JsonObject();
    }

    public string Code { get; }

    public JsonObject Details { get; }

    public JsonObject ToErrorNode()
    {
        var error = new JsonObject
        {
            ["code"] = Code,
            ["message"] = Message
        };
        if (Details.Count > 0)
        {
            error["details"] = Details.DeepClone();
        }

        return new JsonObject
        {
            ["ok"] = false,
            ["error"] = error
        };
    }
}
