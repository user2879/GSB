using System.Text.Json.Nodes;

namespace LevelWorkbench.Core;

public sealed class LoadedRecordFile
{
    public required string Path { get; init; }
    public required int FormatVersion { get; init; }
    public required JsonObject Root { get; init; }
    public required IReadOnlyList<RawPickRecord> Records { get; init; }
}

public sealed class RawPickRecord
{
    public required int Index { get; init; }
    public required int FileLocalIndex { get; init; }
    public required string SourceFile { get; init; }
    public required int FileFormatVersion { get; init; }
    public required int Version { get; init; }
    public required JsonObject Node { get; init; }
}

public sealed class LoadedRecordSet
{
    public required IReadOnlyList<LoadedRecordFile> Files { get; init; }
    public required IReadOnlyList<RawPickRecord> Records { get; init; }
    public required string CanonicalDigest { get; init; }

    public JsonObject PrimaryEnvelopeUnknownFields()
    {
        var first = Files.FirstOrDefault();
        if (first is null)
        {
            return new JsonObject();
        }

        var result = (JsonObject)first.Root.DeepClone();
        result.Remove("schema");
        result.Remove("formatVersion");
        result.Remove("records");
        return result;
    }
}

public static class RecordLoader
{
    public const string ExpectedSchema = "level-workbench.pick-records";

    public static LoadedRecordSet Load(string path)
    {
        if (File.Exists(path))
        {
            var file = LoadFile(path);
            return Assemble(new[] { file });
        }

        if (!Directory.Exists(path))
        {
            throw new WorkbenchException(ErrorCodes.Schema, "Record path does not exist.", new JsonObject { ["path"] = path });
        }

        var files = Directory.EnumerateFiles(path, "*.json", SearchOption.TopDirectoryOnly)
            .OrderBy(p => Path.GetFileName(p), StringComparer.Ordinal)
            .Select(LoadFile)
            .ToArray();
        if (files.Length == 0)
        {
            throw new WorkbenchException(ErrorCodes.Schema, "Record directory contains no JSON files.", new JsonObject { ["path"] = path });
        }

        return Assemble(files);
    }

    public static LoadedRecordFile LoadFile(string path)
    {
        if (!File.Exists(path))
        {
            throw new WorkbenchException(ErrorCodes.Schema, "Record file does not exist.", new JsonObject { ["path"] = path });
        }

        JsonObject root;
        try
        {
            root = JsonRead.ParseObject(File.ReadAllText(path), "pick-record-file");
        }
        catch (WorkbenchException)
        {
            throw;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            throw new WorkbenchException(ErrorCodes.Schema, "Pick-record file could not be read.", new JsonObject
            {
                ["path"] = path,
                ["reason"] = ex.Message
            }, ex);
        }
        var schema = JsonRead.RequireString(root, "schema", "$");
        if (schema != ExpectedSchema)
        {
            throw new WorkbenchException(ErrorCodes.Schema, "Unsupported pick-record schema.", new JsonObject
            {
                ["path"] = path,
                ["expected"] = ExpectedSchema,
                ["actual"] = schema
            });
        }

        var formatVersion = JsonRead.RequireInt32(root, "formatVersion", "$");
        if (!RecordVersions.IsKnown(formatVersion))
        {
            throw new WorkbenchException(ErrorCodes.Version, "Unknown pick-record file format version.", new JsonObject
            {
                ["path"] = path,
                ["knownVersions"] = new JsonArray(RecordVersions.Known.OrderBy(v => v).Select(v => (JsonNode?)JsonValue.Create(v)).ToArray()),
                ["actual"] = formatVersion
            });
        }

        var recordsArray = JsonRead.RequireArray(root, "records", "$");
        var records = new List<RawPickRecord>(recordsArray.Count);
        for (var i = 0; i < recordsArray.Count; i++)
        {
            var recordPath = $"$.records[{i}]";
            if (recordsArray[i] is not JsonObject node)
            {
                throw JsonRead.Schema("Each pick record must be an object.", recordPath, "record");
            }

            var version = JsonRead.RequireInt32(node, "version", recordPath);
            if (!RecordVersions.IsKnown(version))
            {
                throw new WorkbenchException(ErrorCodes.Version, "Unknown pick-record version.", new JsonObject
                {
                    ["path"] = path,
                    ["recordIndex"] = i,
                    ["knownVersions"] = new JsonArray(RecordVersions.Known.OrderBy(v => v).Select(v => (JsonNode?)JsonValue.Create(v)).ToArray()),
                    ["actual"] = version
                });
            }

            records.Add(new RawPickRecord
            {
                Index = -1,
                FileLocalIndex = i,
                SourceFile = Path.GetFullPath(path),
                FileFormatVersion = formatVersion,
                Version = version,
                Node = node
            });
        }

        return new LoadedRecordFile
        {
            Path = Path.GetFullPath(path),
            FormatVersion = formatVersion,
            Root = root,
            Records = records
        };
    }

    private static LoadedRecordSet Assemble(IReadOnlyList<LoadedRecordFile> files)
    {
        var records = new List<RawPickRecord>();
        foreach (var file in files)
        {
            for (var i = 0; i < file.Records.Count; i++)
            {
                var record = file.Records[i];
                records.Add(new RawPickRecord
                {
                    Index = records.Count,
                    FileLocalIndex = i,
                    SourceFile = record.SourceFile,
                    FileFormatVersion = record.FileFormatVersion,
                    Version = record.Version,
                    Node = record.Node
                });
            }
        }

        var canonical = new JsonArray(files.Select(file => (JsonNode?)new JsonObject
        {
            ["formatVersion"] = file.FormatVersion,
            ["records"] = new JsonArray(file.Records.Select(record => (JsonNode?)record.Node.DeepClone()).ToArray())
        }).ToArray());
        var digest = Sha256Util.HashWithPrefix(CanonicalJson.Serialize(canonical));
        return new LoadedRecordSet
        {
            Files = files,
            Records = records,
            CanonicalDigest = digest
        };
    }
}
