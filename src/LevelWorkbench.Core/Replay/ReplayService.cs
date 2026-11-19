using System.Text.Json.Nodes;

namespace LevelWorkbench.Core;

public sealed class ReplayService
{
    public const int InterruptExitCode = 130;

    private readonly PickEngine _pickEngine = new();

    public async Task<ReplayReport> RunAsync(
        SceneDocument scene,
        LoadedRecordSet recordSet,
        IReadOnlyList<NormalizedPickRecord> normalized,
        string stateDirectory,
        int? chunkSizeOverride = null,
        CancellationToken cancellationToken = default,
        IReadOnlyDictionary<string, Transform3>? poses = null)
    {
        var sceneSnapshot = poses is null
            ? SceneSnapshot.FromDocument(scene)
            : SceneSnapshot.FromPoses(scene, poses);
        var sceneDigest = sceneSnapshot.ComputeDigest();
        var versionId = ComputeVersionId(sceneDigest, recordSet.CanonicalDigest, "replay");
        var chunkSize = chunkSizeOverride ?? scene.ChunkSize;
        if (chunkSize <= 0)
        {
            throw new WorkbenchException(ErrorCodes.Schema, "Chunk size must be positive.");
        }

        var journal = new ChunkJournal(stateDirectory, versionId, "replay", chunkSize);
        var totalChunks = (normalized.Count + chunkSize - 1) / chunkSize;
        var allResults = new ReplayRecordResult[normalized.Count];
        var resumed = 0;
        var computed = 0;
        var interrupted = false;

        for (var chunk = 0; chunk < totalChunks; chunk++)
        {
            if (cancellationToken.IsCancellationRequested)
            {
                journal.WriteInterrupt(chunk, InterruptExitCode);
                interrupted = true;
                break;
            }

            var first = chunk * chunkSize;
            var count = System.Math.Min(chunkSize, normalized.Count - first);
            var cached = journal.TryReadValidChunk(chunk, first, count);
            JsonArray resultNodes;
            if (cached is not null)
            {
                resultNodes = cached.Results;
                resumed++;
            }
            else
            {
                var chunkRecords = normalized.Skip(first).Take(count).ToArray();
                // A termination request is observed only after this full chunk is computed and
                // persisted. Do not pass the user cancellation token into chunk work.
                var chunkResults = await ComputeChunkAsync(sceneSnapshot, chunkRecords, CancellationToken.None).ConfigureAwait(false);
                resultNodes = new JsonArray(chunkResults.Select(result => (JsonNode?)result.ToJson()).ToArray());
                journal.WriteChunk(chunk, first, resultNodes);
                computed++;
            }

            for (var i = 0; i < resultNodes.Count; i++)
            {
                if (resultNodes[i] is not JsonObject resultNode)
                {
                    throw new WorkbenchException(ErrorCodes.State, "Persisted chunk result is not an object.", new JsonObject
                    {
                        ["chunkIndex"] = chunk,
                        ["resultIndex"] = i
                    });
                }

                allResults[first + i] = ReplayRecordResult.FromJson(resultNode);
            }

            // Cancellation is observed only after the complete chunk has been persisted.
            if (cancellationToken.IsCancellationRequested)
            {
                journal.WriteInterrupt(chunk + 1, InterruptExitCode);
                interrupted = true;
                break;
            }
        }

        var completedResults = allResults.Where(result => result is not null).ToArray();
        var report = new ReplayReport
        {
            VersionId = versionId,
            SceneDigest = sceneDigest,
            RecordsDigest = recordSet.CanonicalDigest,
            ChunkSize = chunkSize,
            Interrupted = interrupted,
            InterruptExitCode = InterruptExitCode,
            ResumedChunkCount = resumed,
            ComputedChunkCount = computed,
            Records = completedResults
        };

        if (!interrupted)
        {
            journal.WriteCompletionReport(report.ToJson());
        }

        return report;
    }

    public static string ComputeVersionId(string sceneDigest, string recordsDigest, string operation) =>
        Sha256Util.HashWithPrefix($"level-workbench|operation={operation}|scene={sceneDigest}|records={recordsDigest}");

    private async Task<ReplayRecordResult[]> ComputeChunkAsync(
        SceneSnapshot scene,
        IReadOnlyList<NormalizedPickRecord> records,
        CancellationToken cancellationToken)
    {
        return (await BoundedParallel.RunAsync(
            records,
            record => FormattableString.Invariant($"{record.Target.StableId}:{record.RecordIndex:D8}"),
            record => EstimateRecordBytes(scene, record),
            scene.Scene.ConcurrencyLimit,
            scene.Scene.MemoryBudgetBytes,
            (record, token) => Task.FromResult(ReplayOne(scene, record, token)),
            cancellationToken).ConfigureAwait(false)).ToArray();
    }

    private ReplayRecordResult ReplayOne(SceneSnapshot scene, NormalizedPickRecord record, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var conflicts = new List<ReplayConflict>();
        var tolerance = scene.Scene.CoordinateToleranceMeters;

        var currentLayer = scene.Scene.LayerById(record.Target.LayerId);
        if (record.LayerVisible != currentLayer.Visible)
        {
            conflicts.Add(Conflict("layerVisible", "Record layer visibility differs from the current scene.", record.LayerVisible, currentLayer.Visible));
        }

        var currentPickable = currentLayer.Pickable && record.Target.Pickable;
        if (record.Pickable != currentPickable)
        {
            conflicts.Add(Conflict("pickable", "Record effective pickability differs from the current scene.", record.Pickable, currentPickable));
        }

        if (record.FaceIndex < 0 || record.FaceIndex >= record.Target.Geometry.TriangleCount)
        {
            conflicts.Add(Conflict("faceIndex", "Record face index is outside the current proxy geometry.", record.FaceIndex, record.Target.Geometry.TriangleCount));
            return Failure(record, conflicts);
        }

        var targetInstance = scene.ById(record.Target.Id);
        var face = record.Target.Geometry.TriangleWorld(record.FaceIndex, targetInstance.Transform);
        if (!PointOnFace(record.HitPointMeters, face, tolerance))
        {
            conflicts.Add(Conflict("hitPointMeters", "Recorded hit point no longer lies on the recorded face.", SceneDigest.VecNode(record.HitPointMeters), "not_on_face"));
        }

        var recordNormal = record.FaceNormal.NormalizedOrZero();
        var faceNormal = face.UnitNormal();
        if (faceNormal.LengthSquared > 1e-20 && Vec3.Dot(faceNormal, record.HitPointMeters - scene.Scene.Camera.PositionMeters) > 0)
        {
            faceNormal = -faceNormal;
        }

        if (recordNormal.LengthSquared <= 1e-20 || Vec3.Distance(recordNormal, faceNormal) > 1e-3)
        {
            conflicts.Add(Conflict("faceNormal", "Recorded face normal is not the normalized camera-facing face normal.", SceneDigest.VecNode(faceNormal), SceneDigest.VecNode(recordNormal)));
        }

        var screenPoint = record.ScreenPoint;
        if (screenPoint is null)
        {
            conflicts.Add(Conflict("screenPoint", "Record hit point cannot be projected into the current camera viewport.", null, null));
            return Failure(record, conflicts);
        }

        var query = new PickQuery
        {
            Filter = record.PickFilter,
            ViewportTransform = record.ViewportTransform,
            ViewportOverride = record.Viewport
        };
        var hit = _pickEngine.PickSerial(scene, screenPoint.Value, query, cancellationToken);
        if (hit is null)
        {
            conflicts.Add(Conflict("hit", "Replay ray produced no pick under the current filters.", record.Target.StableId, null));
            return Failure(record, conflicts);
        }

        if (!string.Equals(hit.StableId, record.Target.StableId, StringComparison.Ordinal))
        {
            conflicts.Add(Conflict("stableId", "Replay resolved a different proxy.", record.Target.StableId, hit.StableId));
        }

        if (hit.FaceIndex != record.FaceIndex)
        {
            conflicts.Add(Conflict("faceIndex", "Replay resolved a different face.", record.FaceIndex, hit.FaceIndex));
        }

        var pointError = Vec3.Distance(hit.HitPointMeters, record.HitPointMeters);
        if (pointError > tolerance)
        {
            conflicts.Add(Conflict("hitPointMeters", "Replay hit point exceeds the coordinate tolerance.", SceneDigest.VecNode(record.HitPointMeters), SceneDigest.VecNode(hit.HitPointMeters)));
        }

        if (record.Version >= RecordVersions.StableIdentity)
        {
            var distanceError = System.Math.Abs(hit.ScreenDistanceMeters - record.ScreenDistanceMeters);
            if (distanceError > tolerance)
            {
                conflicts.Add(Conflict("screenDistanceMeters", "Replay screen distance exceeds the coordinate tolerance.", record.ScreenDistanceMeters, hit.ScreenDistanceMeters));
            }

            if (Vec3.Distance(hit.FaceNormal, recordNormal) > 1e-3)
            {
                conflicts.Add(Conflict("faceNormal", "Replay normal differs from the recorded camera-facing normal.", SceneDigest.VecNode(recordNormal), SceneDigest.VecNode(hit.FaceNormal)));
            }
        }

        return new ReplayRecordResult
        {
            RecordIndex = record.RecordIndex,
            RecordId = record.RecordId,
            Version = record.Version,
            Ok = conflicts.Count == 0,
            ExpectedStableId = record.Target.StableId,
            ExpectedFaceIndex = record.FaceIndex,
            ExpectedHitPointMeters = record.HitPointMeters,
            ActualStableId = hit.StableId,
            ActualFaceIndex = hit.FaceIndex,
            ActualHitPointMeters = hit.HitPointMeters,
            ActualFaceNormal = hit.FaceNormal,
            ActualScreenDistanceMeters = hit.ScreenDistanceMeters,
            Conflicts = conflicts
        };
    }

    private static ReplayRecordResult Failure(NormalizedPickRecord record, IReadOnlyList<ReplayConflict> conflicts) => new()
    {
        RecordIndex = record.RecordIndex,
        RecordId = record.RecordId,
        Version = record.Version,
        Ok = false,
        ExpectedStableId = record.Target.StableId,
        ExpectedFaceIndex = record.FaceIndex,
        ExpectedHitPointMeters = record.HitPointMeters,
        Conflicts = conflicts
    };

    private static ReplayConflict Conflict(string field, string message, JsonNode? expected, JsonNode? actual) => new()
    {
        Field = field,
        Message = message,
        Expected = expected,
        Actual = actual
    };

    private static ReplayConflict Conflict(string field, string message, double expected, double actual) => Conflict(
        field, message, JsonValue.Create(expected), JsonValue.Create(actual));

    private static ReplayConflict Conflict(string field, string message, bool expected, bool actual) => Conflict(
        field, message, JsonValue.Create(expected), JsonValue.Create(actual));

    private static bool PointOnFace(Vec3 point, Triangle3 triangle, double tolerance)
    {
        var normal = triangle.UnitNormal();
        if (normal.LengthSquared <= 1e-20)
        {
            return false;
        }

        if (System.Math.Abs(Vec3.Dot(normal, point - triangle.A)) > System.Math.Max(tolerance, 1e-7))
        {
            return false;
        }

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
        var epsilon = System.Math.Min(1e-3, System.Math.Max(tolerance, 1e-7));
        return u >= -epsilon && v >= -epsilon && u + v <= 1 + epsilon;
    }

    private static long EstimateRecordBytes(SceneSnapshot scene, NormalizedPickRecord record) =>
        record.Target.Geometry.EstimatedTriangleBufferBytes + CanonicalJson.Serialize(record.Raw.Node).Length * 2L + 512L;
}
