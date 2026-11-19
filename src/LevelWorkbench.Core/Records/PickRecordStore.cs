using System.Text.Json.Nodes;

namespace LevelWorkbench.Core;

public sealed class RecordWriteResult
{
    public required string Path { get; init; }
    public required int RecordCount { get; init; }
    public required string ContentDigest { get; init; }
    public string? ArchivedPath { get; init; }
    public int? PreviousFormatVersion { get; init; }
    public string? PreviousContentDigest { get; init; }
}

public sealed class PickRecordStore
{
    private readonly RecordNormalizer _normalizer = new();

    public RecordWriteResult MigrateAndWrite(string targetPath, SceneDocument scene, LoadedRecordSet source)
    {
        // Normalize every record before touching the target or its archive directory.
        var normalized = _normalizer.NormalizeAll(scene, source);
        var records = normalized.Select(record => (JsonNode?)RecordMigrator.MigrateToCurrent(scene, record)).ToArray();
        return WriteCurrentNodes(targetPath, scene, records, source.PrimaryEnvelopeUnknownFields(), "migrated");
    }

    public RecordWriteResult AppendHit(string targetPath, SceneDocument scene, PickHit hit, string recordId)
    {
        LoadedRecordSet? existing = null;
        IReadOnlyList<NormalizedPickRecord> normalizedExisting = Array.Empty<NormalizedPickRecord>();
        if (File.Exists(targetPath))
        {
            existing = RecordLoader.Load(targetPath);
        }

        var nodes = new List<JsonNode?>();
        JsonObject envelopeExtras;
        if (existing is not null)
        {
            normalizedExisting = _normalizer.NormalizeAll(scene, existing);
            nodes.AddRange(normalizedExisting.Select(record => (JsonNode?)RecordMigrator.MigrateToCurrent(scene, record)));
            envelopeExtras = existing.PrimaryEnvelopeUnknownFields();
        }
        else
        {
            envelopeExtras = new JsonObject();
        }

        var hitNode = hit.ToCurrentRecord(recordId, scene.Viewport);
        if (normalizedExisting.Any(record => string.Equals(record.RecordId, recordId, StringComparison.Ordinal)))
        {
            throw new WorkbenchException(ErrorCodes.Ident, "New pick record id duplicates an existing record id.", new JsonObject
            {
                ["recordId"] = recordId,
                ["field"] = "recordId"
            });
        }

        var synthetic = new RawPickRecord
        {
            Index = nodes.Count,
            FileLocalIndex = nodes.Count,
            SourceFile = targetPath,
            FileFormatVersion = RecordVersions.Current,
            Version = RecordVersions.Current,
            Node = hitNode
        };
        _normalizer.Normalize(scene, synthetic);
        nodes.Add(hitNode);
        return WriteCurrentNodes(targetPath, scene, nodes, envelopeExtras, "desktop-pick");
    }

    public RecordWriteResult WriteCurrent(string targetPath, SceneDocument scene, LoadedRecordSet source)
    {
        var normalized = _normalizer.NormalizeAll(scene, source);
        var nodes = normalized.Select(record => (JsonNode?)RecordMigrator.MigrateToCurrent(scene, record)).ToArray();
        return WriteCurrentNodes(targetPath, scene, nodes, source.PrimaryEnvelopeUnknownFields(), "migration");
    }

    public RecordWriteResult WriteGeneratedCurrent(string targetPath, SceneDocument scene, IReadOnlyList<JsonObject> currentRecords)
    {
        var ids = new HashSet<string>(StringComparer.Ordinal);
        for (var i = 0; i < currentRecords.Count; i++)
        {
            var normalized = _normalizer.Normalize(scene, new RawPickRecord
            {
                Index = i,
                FileLocalIndex = i,
                SourceFile = targetPath,
                FileFormatVersion = RecordVersions.Current,
                Version = RecordVersions.Current,
                Node = currentRecords[i]
            });
            if (!ids.Add(normalized.RecordId))
            {
                throw new WorkbenchException(ErrorCodes.Ident, "Generated record ids must be unique.", new JsonObject
                {
                    ["recordIndex"] = i,
                    ["recordId"] = normalized.RecordId,
                    ["field"] = "recordId"
                });
            }
        }

        return WriteCurrentNodes(
            targetPath,
            scene,
            currentRecords.Select(node => (JsonNode?)node.DeepClone()).ToArray(),
            new JsonObject(),
            "generated-pick-script");
    }

    private static RecordWriteResult WriteCurrentNodes(
        string targetPath,
        SceneDocument scene,
        IReadOnlyList<JsonNode?> records,
        JsonObject envelopeUnknownFields,
        string writer)
    {
        int? previousVersion = null;
        string? previousDigest = null;
        var archivePath = (string?)null;
        var fullTarget = Path.GetFullPath(targetPath);
        if (File.Exists(fullTarget))
        {
            var previousBytes = File.ReadAllBytes(fullTarget);
            previousDigest = Sha256Util.HashBytesWithPrefix(previousBytes);
            var previousRoot = JsonRead.ParseObject(File.ReadAllText(fullTarget), "existing-pick-record-file");
            previousVersion = JsonRead.RequireInt32(previousRoot, "formatVersion", "$");
            if (!RecordVersions.IsKnown(previousVersion.Value))
            {
                throw new WorkbenchException(ErrorCodes.Version, "Existing target file has an unknown format version.", new JsonObject
                {
                    ["path"] = fullTarget,
                    ["formatVersion"] = previousVersion.Value
                });
            }

            if (previousVersion.Value != RecordVersions.Current)
            {
                var archiveDir = Path.Combine(Path.GetDirectoryName(fullTarget)!, "archive");
                var stamp = DateTime.UtcNow.ToString("yyyyMMddTHHmmssfffZ", System.Globalization.CultureInfo.InvariantCulture);
                archivePath = Path.Combine(archiveDir, $"{Path.GetFileNameWithoutExtension(fullTarget)}.v{previousVersion.Value}.{stamp}{Path.GetExtension(fullTarget)}");
                AtomicFile.WriteAllBytes(archivePath, previousBytes);
            }
        }

        var root = new JsonObject();
        foreach (var property in envelopeUnknownFields)
        {
            if (property.Key is not ("schema" or "formatVersion" or "records"))
            {
                root[property.Key] = property.Value?.DeepClone();
            }
        }

        root["schema"] = RecordLoader.ExpectedSchema;
        root["formatVersion"] = RecordVersions.Current;
        root["unit"] = "meter";
        root["recordCount"] = records.Count;
        root["writtenBy"] = writer;
        root["records"] = new JsonArray(records.Select(node => node?.DeepClone()).ToArray());

        var json = root.ToJsonString(new System.Text.Json.JsonSerializerOptions { WriteIndented = true });
        AtomicFile.WriteAllText(fullTarget, json + "\n", ErrorCodes.Write);
        var digest = Sha256Util.HashWithPrefix(json + "\n");
        return new RecordWriteResult
        {
            Path = fullTarget,
            RecordCount = records.Count,
            ContentDigest = digest,
            ArchivedPath = archivePath,
            PreviousFormatVersion = previousVersion,
            PreviousContentDigest = previousDigest
        };
    }
}
