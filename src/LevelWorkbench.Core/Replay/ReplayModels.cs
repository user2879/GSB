using System.Text.Json.Nodes;

namespace LevelWorkbench.Core;

public sealed class ReplayConflict
{
    public required string Field { get; init; }
    public required string Message { get; init; }
    public JsonNode? Expected { get; init; }
    public JsonNode? Actual { get; init; }

    public JsonObject ToJson() => new()
    {
        ["field"] = Field,
        ["message"] = Message,
        ["expected"] = Expected?.DeepClone(),
        ["actual"] = Actual?.DeepClone()
    };
}

public sealed class ReplayRecordResult
{
    public required int RecordIndex { get; init; }
    public required string RecordId { get; init; }
    public required int Version { get; init; }
    public required bool Ok { get; init; }
    public required string ExpectedStableId { get; init; }
    public required int ExpectedFaceIndex { get; init; }
    public required Vec3 ExpectedHitPointMeters { get; init; }
    public string? ActualStableId { get; init; }
    public int? ActualFaceIndex { get; init; }
    public Vec3? ActualHitPointMeters { get; init; }
    public Vec3? ActualFaceNormal { get; init; }
    public double? ActualScreenDistanceMeters { get; init; }
    public required IReadOnlyList<ReplayConflict> Conflicts { get; init; }

    public JsonObject ToJson()
    {
        var node = new JsonObject
        {
            ["recordIndex"] = RecordIndex,
            ["recordId"] = RecordId,
            ["version"] = Version,
            ["ok"] = Ok,
            ["expected"] = new JsonObject
            {
                ["stableId"] = ExpectedStableId,
                ["faceIndex"] = ExpectedFaceIndex,
                ["hitPointMeters"] = SceneDigest.VecNode(ExpectedHitPointMeters)
            },
            ["actual"] = ActualStableId is null ? null : new JsonObject
            {
                ["stableId"] = ActualStableId,
                ["faceIndex"] = ActualFaceIndex,
                ["hitPointMeters"] = ActualHitPointMeters is { } point ? SceneDigest.VecNode(point) : null,
                ["faceNormal"] = ActualFaceNormal is { } normal ? SceneDigest.VecNode(normal) : null,
                ["screenDistanceMeters"] = ActualScreenDistanceMeters
            },
            ["conflicts"] = new JsonArray(Conflicts.Select(conflict => (JsonNode?)conflict.ToJson()).ToArray())
        };
        return node;
    }

    public static ReplayRecordResult FromJson(JsonObject node)
    {
        var expected = JsonRead.RequireObject(node, "expected", "$");
        var conflicts = new List<ReplayConflict>();
        if (node["conflicts"] is JsonArray conflictArray)
        {
            foreach (var conflictNode in conflictArray.OfType<JsonObject>())
            {
                conflicts.Add(new ReplayConflict
                {
                    Field = JsonRead.RequireString(conflictNode, "field", "$.conflicts[]"),
                    Message = JsonRead.RequireString(conflictNode, "message", "$.conflicts[]"),
                    Expected = conflictNode["expected"]?.DeepClone(),
                    Actual = conflictNode["actual"]?.DeepClone()
                });
            }
        }

        var actualNode = node["actual"] as JsonObject;
        return new ReplayRecordResult
        {
            RecordIndex = JsonRead.RequireInt32(node, "recordIndex", "$"),
            RecordId = JsonRead.RequireString(node, "recordId", "$"),
            Version = JsonRead.RequireInt32(node, "version", "$"),
            Ok = JsonRead.RequireBool(node, "ok", "$"),
            ExpectedStableId = JsonRead.RequireString(expected, "stableId", "$.expected"),
            ExpectedFaceIndex = JsonRead.RequireInt32(expected, "faceIndex", "$.expected"),
            ExpectedHitPointMeters = JsonRead.RequireVec3(expected, "hitPointMeters", "$.expected"),
            ActualStableId = actualNode is null ? null : JsonRead.RequireString(actualNode, "stableId", "$.actual"),
            ActualFaceIndex = actualNode?["faceIndex"] is null ? null : JsonRead.RequireInt32(actualNode, "faceIndex", "$.actual"),
            ActualHitPointMeters = actualNode?["hitPointMeters"] is JsonArray ? JsonRead.RequireVec3(actualNode, "hitPointMeters", "$.actual") : null,
            ActualFaceNormal = actualNode?["faceNormal"] is JsonArray ? JsonRead.RequireVec3(actualNode, "faceNormal", "$.actual") : null,
            ActualScreenDistanceMeters = actualNode?["screenDistanceMeters"] is null ? null : JsonRead.RequireFiniteDouble(actualNode, "screenDistanceMeters", "$.actual"),
            Conflicts = conflicts
        };
    }
}

public sealed class ReplayReport
{
    public required string VersionId { get; init; }
    public required string SceneDigest { get; init; }
    public required string RecordsDigest { get; init; }
    public required int ChunkSize { get; init; }
    public required bool Interrupted { get; init; }
    public required int InterruptExitCode { get; init; }
    public required int ResumedChunkCount { get; init; }
    public required int ComputedChunkCount { get; init; }
    public required IReadOnlyList<ReplayRecordResult> Records { get; init; }

    public bool Ok => Records.All(result => result.Ok);

    public int FailedCount => Records.Count(result => !result.Ok);

    public JsonObject ToJson()
    {
        var node = new JsonObject
        {
            ["ok"] = Ok && !Interrupted,
            ["operation"] = "replay",
            ["versionId"] = VersionId,
            ["sceneDigest"] = SceneDigest,
            ["recordsDigest"] = RecordsDigest,
            ["chunkSize"] = ChunkSize,
            ["recordCount"] = Records.Count,
            ["successCount"] = Records.Count(result => result.Ok),
            ["failedCount"] = FailedCount,
            ["resumedChunkCount"] = ResumedChunkCount,
            ["computedChunkCount"] = ComputedChunkCount,
            ["interrupted"] = Interrupted,
            ["exitCode"] = Interrupted ? InterruptExitCode : (Ok ? 0 : 2),
            ["records"] = new JsonArray(Records.Select(result => (JsonNode?)result.ToJson()).ToArray())
        };
        if (!Ok)
        {
            node["error"] = new JsonObject
            {
                ["code"] = ErrorCodes.Replay,
                ["message"] = "One or more historical pick records cannot be reproduced.",
                ["details"] = new JsonObject
                {
                    ["failedCount"] = FailedCount,
                    ["failedRecords"] = new JsonArray(Records
                        .Where(result => !result.Ok)
                        .Select(result => (JsonNode?)new JsonObject
                        {
                            ["recordIndex"] = result.RecordIndex,
                            ["recordId"] = result.RecordId,
                            ["version"] = result.Version,
                            ["conflictFields"] = new JsonArray(result.Conflicts.Select(c => (JsonNode?)JsonValue.Create(c.Field)).ToArray())
                        }).ToArray())
                }
            };
        }

        return node;
    }
}
