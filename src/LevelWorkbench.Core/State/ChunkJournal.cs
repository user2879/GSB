using System.Text.Json.Nodes;

namespace LevelWorkbench.Core;

public sealed class ChunkState
{
    public required int ChunkIndex { get; init; }
    public required int FirstRecordIndex { get; init; }
    public required int RecordCount { get; init; }
    public required JsonArray Results { get; init; }
    public JsonObject? Metadata { get; init; }
    public required string ChunkDigest { get; init; }
}

public sealed class ChunkJournal
{
    public const string Schema = "level-workbench.chunk";
    public const int FormatVersion = 1;

    public ChunkJournal(string stateDirectory, string versionId, string operation, int chunkSize)
    {
        RunDirectory = Path.Combine(
            Path.GetFullPath(stateDirectory),
            "runs",
            versionId.Replace(':', '-'),
            FormattableString.Invariant($"{operation}-c{chunkSize}"));
        VersionId = versionId;
        Operation = operation;
        ChunkSize = chunkSize;
    }

    public string RunDirectory { get; }

    public string VersionId { get; }

    public string Operation { get; }

    public int ChunkSize { get; }

    public string ChunkPath(int chunkIndex) => Path.Combine(RunDirectory, FormattableString.Invariant($"chunk-{chunkIndex:D6}.json"));

    public ChunkState? TryReadValidChunk(int chunkIndex, int expectedFirstRecordIndex, int expectedRecordCount)
    {
        var path = ChunkPath(chunkIndex);
        if (!File.Exists(path))
        {
            return null;
        }

        try
        {
            var root = JsonRead.ParseObject(File.ReadAllText(path), "chunk-state");
            var schemaOk = JsonRead.RequireString(root, "schema", "$") == Schema;
            var formatOk = JsonRead.RequireInt32(root, "formatVersion", "$") == FormatVersion;
            var operationOk = JsonRead.RequireString(root, "operation", "$") == Operation;
            var versionOk = JsonRead.RequireString(root, "versionId", "$") == VersionId;
            var indexOk = JsonRead.RequireInt32(root, "chunkIndex", "$") == chunkIndex;
            var firstOk = JsonRead.RequireInt32(root, "firstRecordIndex", "$") == expectedFirstRecordIndex;
            var countOk = JsonRead.RequireInt32(root, "recordCount", "$") == expectedRecordCount;
            var completedOk = JsonRead.RequireBool(root, "completed", "$");
            if (!(schemaOk && formatOk && operationOk && versionOk && indexOk && firstOk && countOk && completedOk))
            {
                return null;
            }

            var results = root["results"] as JsonArray;
            if (results is null || results.Count != expectedRecordCount)
            {
                return null;
            }

            var digest = JsonRead.RequireString(root, "chunkDigest", "$");
            var metadata = root["metadata"] as JsonObject;
            var computed = ComputeChunkDigest(results, metadata);
            if (!string.Equals(digest, computed, StringComparison.Ordinal))
            {
                return null;
            }

            return new ChunkState
            {
                ChunkIndex = chunkIndex,
                FirstRecordIndex = expectedFirstRecordIndex,
                RecordCount = expectedRecordCount,
                Results = (JsonArray)results.DeepClone(),
                Metadata = metadata is null ? null : (JsonObject)metadata.DeepClone(),
                ChunkDigest = digest
            };
        }
        catch (Exception ex) when (ex is WorkbenchException or IOException or UnauthorizedAccessException)
        {
            return null;
        }
    }

    public void WriteChunk(int chunkIndex, int firstRecordIndex, JsonArray results, JsonObject? metadata = null)
    {
        var digest = ComputeChunkDigest(results, metadata);
        var root = new JsonObject
        {
            ["schema"] = Schema,
            ["formatVersion"] = FormatVersion,
            ["operation"] = Operation,
            ["versionId"] = VersionId,
            ["chunkIndex"] = chunkIndex,
            ["firstRecordIndex"] = firstRecordIndex,
            ["recordCount"] = results.Count,
            ["results"] = results.DeepClone(),
            ["metadata"] = metadata?.DeepClone(),
            ["chunkDigest"] = digest,
            ["completed"] = true
        };
        var json = root.ToJsonString(new System.Text.Json.JsonSerializerOptions { WriteIndented = true }) + "\n";
        AtomicFile.WriteAllText(ChunkPath(chunkIndex), json, ErrorCodes.State);
    }

    public static string ComputeChunkDigest(JsonArray results, JsonObject? metadata)
    {
        var digestInput = new JsonObject
        {
            ["results"] = results.DeepClone(),
            ["metadata"] = metadata?.DeepClone()
        };
        return Sha256Util.HashWithPrefix(CanonicalJson.Serialize(digestInput));
    }

    public void WriteInterrupt(int nextChunkIndex, int exitCode)
    {
        var root = new JsonObject
        {
            ["schema"] = "level-workbench.interrupt",
            ["formatVersion"] = 1,
            ["operation"] = Operation,
            ["versionId"] = VersionId,
            ["nextChunkIndex"] = nextChunkIndex,
            ["exitCode"] = exitCode,
            ["interruptedAtUtc"] = DateTime.UtcNow.ToString("O", System.Globalization.CultureInfo.InvariantCulture),
            ["boundary"] = "chunk"
        };
        AtomicFile.WriteAllText(
            Path.Combine(RunDirectory, "interrupt.json"),
            root.ToJsonString(new System.Text.Json.JsonSerializerOptions { WriteIndented = true }) + "\n",
            ErrorCodes.State);
    }

    public void WriteCompletionReport(JsonObject report)
    {
        AtomicFile.WriteAllText(
            Path.Combine(RunDirectory, "report.json"),
            report.ToJsonString(new System.Text.Json.JsonSerializerOptions { WriteIndented = true }) + "\n",
            ErrorCodes.State);
    }

    public int CountValidChunks(int totalChunks, int totalRecords)
    {
        var count = 0;
        for (var chunk = 0; chunk < totalChunks; chunk++)
        {
            var first = chunk * ChunkSize;
            var size = System.Math.Min(ChunkSize, totalRecords - first);
            if (TryReadValidChunk(chunk, first, size) is not null)
            {
                count++;
            }
        }

        return count;
    }
}
