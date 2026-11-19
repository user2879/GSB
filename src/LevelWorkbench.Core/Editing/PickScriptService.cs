using System.Text.Json.Nodes;

namespace LevelWorkbench.Core;

public sealed class PickScriptExecutionResult
{
    public required string VersionId { get; init; }
    public required bool Interrupted { get; init; }
    public required int InterruptExitCode { get; init; }
    public required int ResumedChunkCount { get; init; }
    public required int ComputedChunkCount { get; init; }
    public required IReadOnlyList<JsonObject> Records { get; init; }
    public required JsonArray StepResults { get; init; }
    public required EditorSnapshot FinalState { get; init; }
}

public sealed class PickScriptService
{
    public const int InterruptExitCode = 130;

    public Task<PickScriptExecutionResult> RunAsync(
        SceneDocument scene,
        PickScriptDocument script,
        string stateDirectory,
        int? chunkSizeOverride = null,
        CancellationToken cancellationToken = default)
        => Task.FromResult(Run(scene, script, stateDirectory, chunkSizeOverride, cancellationToken));

    private static PickScriptExecutionResult Run(
        SceneDocument scene,
        PickScriptDocument script,
        string stateDirectory,
        int? chunkSizeOverride,
        CancellationToken cancellationToken)
    {
        var sceneDigest = SceneSnapshot.FromDocument(scene).ComputeDigest();
        var versionId = ReplayService.ComputeVersionId(sceneDigest, script.CanonicalDigest, "pick");
        var chunkSize = chunkSizeOverride ?? scene.ChunkSize;
        if (chunkSize <= 0)
        {
            throw new WorkbenchException(ErrorCodes.Schema, "Chunk size must be positive.");
        }

        var journal = new ChunkJournal(stateDirectory, versionId, "pick", chunkSize);
        var session = new EditorSession(scene);
        session.SetViewportTransform(script.ViewportTransform);
        var totalChunks = (script.Steps.Count + chunkSize - 1) / chunkSize;
        var records = new List<JsonObject>();
        var stepResults = new JsonArray();
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
            var count = System.Math.Min(chunkSize, script.Steps.Count - first);
            var cached = journal.TryReadValidChunk(chunk, first, count);
            JsonArray chunkResults;
            if (cached is not null && cached.Metadata is not null)
            {
                chunkResults = cached.Results;
                RestoreEditor(session, cached.Metadata);
                resumed++;
            }
            else
            {
                chunkResults = new JsonArray();
                for (var i = 0; i < count; i++)
                {
                    var step = script.Steps[first + i];
                    var (stepNode, _) = ExecuteStep(session, step);
                    chunkResults.Add(stepNode);
                }

                journal.WriteChunk(chunk, first, chunkResults, CaptureEditor(session));
                computed++;
            }

            foreach (var node in chunkResults)
            {
                stepResults.Add(node?.DeepClone());
                if (node is JsonObject obj && obj["records"] is JsonArray recordArray)
                {
                    foreach (var record in recordArray.OfType<JsonObject>())
                    {
                        records.Add((JsonObject)record.DeepClone());
                    }
                }
            }

            if (cancellationToken.IsCancellationRequested)
            {
                journal.WriteInterrupt(chunk + 1, InterruptExitCode);
                interrupted = true;
                break;
            }
        }

        return new PickScriptExecutionResult
        {
            VersionId = versionId,
            Interrupted = interrupted,
            InterruptExitCode = InterruptExitCode,
            ResumedChunkCount = resumed,
            ComputedChunkCount = computed,
            Records = records,
            StepResults = stepResults,
            FinalState = CaptureSnapshot(session)
        };
    }

    private static (JsonObject StepResult, IReadOnlyList<JsonObject> Records) ExecuteStep(EditorSession session, PickScriptStep step)
    {
        var records = new List<JsonObject>();
        var result = new JsonObject
        {
            ["stepIndex"] = step.Index,
            ["kind"] = step.Kind.ToString().ToLowerInvariant()
        };

        switch (step.Kind)
        {
            case PickScriptStepKind.Pick:
            {
                var screen = RequireScreen(step.Screen, step.Index);
                var hit = session.Pick(screen, step.Filter);
                result["screen"] = ScreenNode(screen);
                result["hit"] = HitNode(hit);
                AddHitRecord(records, hit, step.Index, "pick", session.Scene.Viewport);
                break;
            }
            case PickScriptStepKind.Select:
            {
                var screen = RequireScreen(step.Screen, step.Index);
                var hit = session.SelectAt(screen, step.Filter, step.Additive);
                result["screen"] = ScreenNode(screen);
                result["hit"] = HitNode(hit);
                result["selection"] = SelectionNode(session);
                AddHitRecord(records, hit, step.Index, "select", session.Scene.Viewport);
                break;
            }
            case PickScriptStepKind.Box:
            {
                var rect = step.Rect ?? throw new WorkbenchException(ErrorCodes.Schema, "Box step is missing its rectangle.");
                var selection = session.BoxSelect(rect, step.Filter, step.Additive);
                result["selection"] = new JsonArray(selection.OrderBy(id => id, StringComparer.Ordinal).Select(id => (JsonNode?)JsonValue.Create(id)).ToArray());
                break;
            }
            case PickScriptStepKind.Drag:
            {
                var start = RequireScreen(step.DragStart, step.Index);
                var end = RequireScreen(step.DragEnd, step.Index);
                var hit = session.Pick(start, step.Filter);
                PickHit? finalHit = null;
                if (hit is not null)
                {
                    session.BeginDrag(hit, start, step.Filter, additiveSelection: false);
                    finalHit = session.UpdateDrag(end);
                    session.EndDrag();
                    AddHitRecord(records, finalHit, step.Index, "drag", session.Scene.Viewport);
                }

                result["start"] = ScreenNode(start);
                result["end"] = ScreenNode(end);
                result["initialHit"] = HitNode(hit);
                result["finalHit"] = HitNode(finalHit);
                result["selection"] = SelectionNode(session);
                break;
            }
            default:
                throw new WorkbenchException(ErrorCodes.Schema, "Unsupported script step kind.");
        }

        result["records"] = new JsonArray(records.Select(record => (JsonNode?)record.DeepClone()).ToArray());
        return (result, records);
    }

    private static void AddHitRecord(
        List<JsonObject> records,
        PickHit? hit,
        int stepIndex,
        string prefix,
        ViewportDefinition sessionViewport)
    {
        if (hit is null)
        {
            return;
        }

        records.Add(hit.ToCurrentRecord(
            FormattableString.Invariant($"{prefix}-{stepIndex:D4}-001"),
            sessionViewport));
    }

    private static JsonNode? HitNode(PickHit? hit)
    {
        if (hit is null)
        {
            return null;
        }

        return new JsonObject
        {
            ["stableId"] = hit.StableId,
            ["proxyNumber"] = hit.ProxyNumber,
            ["proxyId"] = hit.ProxyId,
            ["faceIndex"] = hit.FaceIndex,
            ["hitPointMeters"] = SceneDigest.VecNode(hit.HitPointMeters),
            ["faceNormal"] = SceneDigest.VecNode(hit.FaceNormal),
            ["screenDistanceMeters"] = hit.ScreenDistanceMeters,
            ["screenPoint"] = ScreenNode(hit.ScreenPoint)
        };
    }

    private static JsonObject ScreenNode(Vec2 point) => new() { ["x"] = point.X, ["y"] = point.Y };

    private static Vec2 RequireScreen(Vec2? point, int stepIndex) => point
        ?? throw new WorkbenchException(ErrorCodes.Schema, "Script step is missing screen coordinates.", new JsonObject { ["stepIndex"] = stepIndex });

    private static JsonArray SelectionNode(EditorSession session) => new(
        session.Selection.OrderBy(id => id, StringComparer.Ordinal).Select(id => (JsonNode?)JsonValue.Create(id)).ToArray());

    private static JsonObject CaptureEditor(EditorSession session)
    {
        var poses = new JsonObject();
        foreach (var (proxyId, pose) in session.Poses.OrderBy(pair => pair.Key, StringComparer.Ordinal))
        {
            poses[proxyId] = new JsonObject
            {
                ["positionMeters"] = SceneDigest.VecNode(pose.PositionMeters),
                ["rotationEulerDegrees"] = SceneDigest.VecNode(pose.RotationEulerDegrees),
                ["scale"] = SceneDigest.VecNode(pose.Scale)
            };
        }

        return new JsonObject
        {
            ["poses"] = poses,
            ["selection"] = SelectionNode(session),
            ["viewport"] = new JsonObject
            {
                ["zoom"] = session.ViewportTransform.Zoom,
                ["panXPixels"] = session.ViewportTransform.PanXPixels,
                ["panYPixels"] = session.ViewportTransform.PanYPixels
            }
        };
    }

    private static EditorSnapshot CaptureSnapshot(EditorSession session)
    {
        var snapshot = session.Snapshot();
        return new EditorSnapshot
        {
            Poses = snapshot.Proxies.ToDictionary(p => p.Definition.Id, p => p.Transform, StringComparer.Ordinal),
            Selection = new HashSet<string>(session.Selection, StringComparer.Ordinal),
            ViewportTransform = session.ViewportTransform
        };
    }

    private static void RestoreEditor(EditorSession session, JsonObject metadata)
    {
        var posesObject = JsonRead.RequireObject(metadata, "poses", "$.metadata");
        foreach (var (proxyId, node) in posesObject)
        {
            if (node is not JsonObject poseNode)
            {
                throw new WorkbenchException(ErrorCodes.State, "Persisted editor pose is invalid.");
            }

            var pose = new Transform3(
                JsonRead.RequireVec3(poseNode, "positionMeters", "$.metadata.poses." + proxyId),
                JsonRead.RequireVec3(poseNode, "rotationEulerDegrees", "$.metadata.poses." + proxyId),
                JsonRead.RequireVec3(poseNode, "scale", "$.metadata.poses." + proxyId));
            session.RestoreProxyPoseForRecovery(proxyId, pose);
        }

        if (metadata["selection"] is JsonArray selectionArray)
        {
            session.RestoreSelectionForRecovery(selectionArray.OfType<JsonValue>()
                .Select(value => value.TryGetValue<string>(out var id) ? id : null)
                .Where(id => id is not null)
                .Select(id => id!));
        }

        if (metadata["viewport"] is JsonObject viewportObj)
        {
            session.SetViewportTransform(new ViewportTransform(
                JsonRead.RequireFiniteDouble(viewportObj, "zoom", "$.metadata.viewport"),
                JsonRead.RequireFiniteDouble(viewportObj, "panXPixels", "$.metadata.viewport"),
                JsonRead.RequireFiniteDouble(viewportObj, "panYPixels", "$.metadata.viewport")));
        }
    }
}
