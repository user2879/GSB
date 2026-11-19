using System.Diagnostics;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Nodes;
using LevelWorkbench.Core;
using SampleGenerator = LevelWorkbench.SampleGen.Program;

namespace LevelWorkbench.SelfTest;

public static class Program
{
    private static readonly List<string> Failures = new();
    private static int _checks;
    private static string _root = string.Empty;

    public static int Main(string[] args)
    {
        _root = Path.Combine(Path.GetTempPath(), "level-workbench-selftest-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_root);
        var started = Stopwatch.StartNew();
        try
        {
            EnsureFixture();
            Run("sample generation is deterministic", SampleGenerationIsDeterministic);
            Run("scene parses and identifiers stay unique", SceneParses);
            Run("duplicate stable id returns E_IDENT", DuplicateStableIdReturnsIdent);
            Run("out-of-bounds index returns E_GEOMETRY", OutOfBoundsIndexReturnsGeometry);
            Run("non-finite vertex returns E_GEOMETRY", NonFiniteVertexReturnsGeometry);
            Run("missing layer visibility returns E_LAYER", MissingLayerVisibilityReturnsLayer);
            Run("scene unit mismatch returns E_UNIT", SceneUnitMismatchReturnsUnit);
            Run("unknown record version returns E_VERSION", UnknownRecordVersionReturnsVersion);
            Run("record unit/field mismatch returns E_UNIT", RecordUnitMismatchReturnsUnit);
            Run("legacy digest resolves without number guessing", LegacyDigestResolves);
            Run("legacy digest mismatch reports candidates", LegacyDigestMismatchReportsCandidates);
            Run("ambiguous legacy digest returns E_IDENT", AmbiguousLegacyDigestReturnsIdent);
            Run("stable id conflicts are rejected", StableIdConflictRejected);
            Run("mixed versions normalize together", MixedVersionsNormalize);
            Run("serial and bounded parallel picks match", SerialAndParallelPicksMatch);
            Run("bounded parallel returns the first ordered error", BoundedParallelReturnsFirstOrderedError);
            Run("memory budget above one job is observed", LimitedMemoryBudgetWorks);
            Run("single job over memory budget returns E_RESOURCE", OversizedMemoryRequestReturnsResource);
            Run("layer visibility and filters exclude nearer proxy", FiltersExcludeNearerProxy);
            Run("viewport zoom/pan preserve ray identity", ViewportZoomPanPreservesRay);
            Run("normals are normalized and camera-facing", NormalsAreCameraFacing);
            Run("box selection respects pick filters", BoxSelectionRespectsFilters);
            Run("drag updates pose and undo restores selection", DragUndoRestoresPoseAndSelection);
            Run("mixed history replay succeeds", ReplaySucceeds);
            Run("replay can use a current pose snapshot", ReplayUsesCurrentPoseSnapshot);
            Run("valid chunks are reused", ValidChunksAreReused);
            Run("tampered chunk is rejected and recomputed", TamperedChunkIsRecomputed);
            Run("cancellation writes boundary interrupt and exit code 130", CancellationWritesInterrupt);
            Run("interrupted replay resumes from valid chunks", InterruptedReplayResumes);
            Run("tampered record returns E_REPLAY with location", TamperedRecordReturnsReplay);
            Run("migration archives v1 and preserves unknown fields", MigrationArchivesAndPreserves);
            Run("invalid input writes no record file", InvalidInputWritesNothing);
            Run("pick script writes deterministic current records", PickScriptWritesCurrentRecords);
            Run("pick script chunks resume without recomputation", PickScriptChunksResume);
            Run("pick script cancellation writes interrupt and resumes", PickScriptCancellationResumes);
            Run("record append upgrades existing v1 atomically", AppendUpgradesExisting);
            Run("POSIX write failure preserves old file", WriteFailurePreservesOldFile);
            Run("CLI returns JSON errors and exit codes", CliBehavior);
            Run("CLI SIGTERM interrupts at a chunk boundary", CliSigtermInterrupts);
        }
        catch (Exception ex)
        {
            Failures.Add("UNEXPECTED: " + ex);
        }
        finally
        {
            started.Stop();
            if (Failures.Count == 0)
            {
                Console.WriteLine($"PASS {_checks} checks in {started.Elapsed.TotalSeconds:F2}s");
                TryDelete(_root);
            }
            else
            {
                Console.WriteLine($"FAIL {Failures.Count} of {_checks} checks in {started.Elapsed.TotalSeconds:F2}s");
                foreach (var failure in Failures)
                {
                    Console.WriteLine("  " + failure);
                }

                Console.WriteLine("Artifacts: " + _root);
            }
        }

        return Failures.Count == 0 ? 0 : 1;
    }

    private static void Run(string name, Action test)
    {
        var before = Failures.Count;
        try
        {
            test();
        }
        catch (Exception ex)
        {
            Failures.Add($"{name}: {ex.GetType().Name}: {ex.Message}");
        }

        if (Failures.Count == before)
        {
            _checks++;
            Console.WriteLine("ok  " + name);
        }
        else
        {
            Console.WriteLine("err " + name);
        }
    }

    private static void SampleGenerationIsDeterministic()
    {
        var a = Path.Combine(_root, "det-a");
        var b = Path.Combine(_root, "det-b");
        SampleGenerator.Generate(a);
        SampleGenerator.Generate(b);
        var hashesA = HashTree(a);
        var hashesB = HashTree(b);
        Check(hashesA.SequenceEqual(hashesB), "generated file hashes differ between runs");
        Check(hashesA.Count >= 14, "expected generated sample files were not produced");
    }

    private static void SceneParses()
    {
        var fixture = Fixture();
        Check(fixture.Scene.Proxies.Count == 9, "proxy count mismatch");
        Check(fixture.Scene.Layers.Count == 5, "layer count mismatch");
        Check(fixture.Scene.Proxies.Select(p => p.Id).Distinct().Count() == fixture.Scene.Proxies.Count, "proxy ids not unique");
        Check(fixture.Scene.Proxies.Select(p => p.StableId).Distinct().Count() == fixture.Scene.Proxies.Count, "stable ids not unique");
        Check(fixture.Scene.Proxies.Select(p => p.ProxyNumber).Distinct().Count() == fixture.Scene.Proxies.Count, "proxy numbers not unique");
        Check(Sha256Util.IsDigest(fixture.Scene.Proxies[0].Geometry.SnapshotDigest), "snapshot digest malformed");
    }

    private static void DuplicateStableIdReturnsIdent()
    {
        ExpectCode(ErrorCodes.Ident, () => SceneLoader.Load(Path.Combine(FixtureRoot(), "invalid", "duplicate-stable-id.scene.json")));
    }

    private static void OutOfBoundsIndexReturnsGeometry()
    {
        ExpectCode(ErrorCodes.Geometry, () => SceneLoader.Load(Path.Combine(FixtureRoot(), "invalid", "index-out-of-bounds.scene.json")));
    }

    private static void NonFiniteVertexReturnsGeometry()
    {
        var root = (JsonObject)JsonNode.Parse(File.ReadAllText(Path.Combine(FixtureRoot(), "scene.json")))!;
        var proxies = (JsonArray)root["proxies"]!;
        var proxy = (JsonObject)proxies[0]!;
        var geometry = (JsonObject)proxy["geometry"]!;
        var vertices = (JsonArray)geometry["verticesMeters"]!;
        var vertex = (JsonArray)vertices[0]!;
        ((JsonArray)vertex)[0] = JsonValue.Create("NaN");
        ExpectCode(ErrorCodes.Geometry, () => SceneLoader.Parse(root.ToJsonString()));
    }

    private static void MissingLayerVisibilityReturnsLayer()
    {
        ExpectCode(ErrorCodes.Layer, () => SceneLoader.Load(Path.Combine(FixtureRoot(), "invalid", "missing-layer-visibility.scene.json")));
    }

    private static void SceneUnitMismatchReturnsUnit()
    {
        var root = (JsonObject)JsonNode.Parse(File.ReadAllText(Path.Combine(FixtureRoot(), "scene.json")))!;
        root["unit"] = "centimeter";
        ExpectCode(ErrorCodes.Unit, () => SceneLoader.Parse(root.ToJsonString()));
    }

    private static void UnknownRecordVersionReturnsVersion()
    {
        ExpectCode(ErrorCodes.Version, () => RecordLoader.Load(Path.Combine(FixtureRoot(), "invalid", "unknown-record-version.json")));
    }

    private static void RecordUnitMismatchReturnsUnit()
    {
        var fixture = Fixture();
        var records = RecordLoader.Load(Path.Combine(FixtureRoot(), "invalid", "unit-mismatch.records.json"));
        ExpectCode(ErrorCodes.Unit, () => new RecordNormalizer().NormalizeAll(fixture.Scene, records));
    }

    private static void LegacyDigestResolves()
    {
        var fixture = Fixture();
        var records = RecordLoader.Load(Path.Combine(FixtureRoot(), "records", "legacy-v1.json"));
        var normalized = new RecordNormalizer().NormalizeAll(fixture.Scene, records);
        Check(normalized.Count == 1, "legacy record count mismatch");
        Check(normalized[0].Target.StableId == "stable-proxy-a", "legacy digest bound to the wrong proxy");
        Check(normalized[0].Version == 1, "legacy source version lost");
        var cm = normalized[0].Raw.Node["hitPointCm"]![0]!.GetValue<double>();
        Check(Math.Abs(normalized[0].HitPointMeters.X - cm / 100.0) < 1e-12, "legacy centimeters were not converted to meters");
    }

    private static void LegacyDigestMismatchReportsCandidates()
    {
        var fixture = Fixture();
        var records = RecordLoader.Load(Path.Combine(FixtureRoot(), "invalid", "digest-mismatch.records.json"));
        var error = ExpectCode(ErrorCodes.Ident, () => new RecordNormalizer().NormalizeAll(fixture.Scene, records));
        Check(error.Details["candidates"] is JsonArray candidates && candidates.Count == 1, "digest mismatch did not report the declared-number candidate");
        Check(error.Details["field"]?.GetValue<string>() == "snapshotDigest", "conflict field not reported");
    }

    private static void AmbiguousLegacyDigestReturnsIdent()
    {
        var fixture = Fixture();
        var records = RecordLoader.Load(Path.Combine(FixtureRoot(), "invalid", "ambiguous-digest.records.json"));
        var error = ExpectCode(ErrorCodes.Ident, () => new RecordNormalizer().NormalizeAll(fixture.Scene, records));
        Check(error.Details["candidates"] is JsonArray candidates && candidates.Count == 2, "ambiguous digest did not report both candidates");
    }

    private static void StableIdConflictRejected()
    {
        var fixture = Fixture();
        var set = RecordLoader.Load(Path.Combine(FixtureRoot(), "records", "modern-v2.json"));
        set.Records[0].Node["proxyNumber"] = 102;
        ExpectCode(ErrorCodes.Ident, () => new RecordNormalizer().NormalizeAll(fixture.Scene, set));
    }

    private static void MixedVersionsNormalize()
    {
        var fixture = Fixture();
        var set = RecordLoader.Load(Path.Combine(FixtureRoot(), "records", "mixed.json"));
        var normalized = new RecordNormalizer().NormalizeAll(fixture.Scene, set);
        Check(normalized.Count == 4, "mixed record count mismatch");
        Check(normalized.Select(r => r.Version).OrderBy(v => v).SequenceEqual(new[] { 1, 2, 3, 3 }), "mixed versions not preserved");
        Check(normalized[0].Target.StableId == "stable-proxy-a", "v1 binding wrong in mixed set");
        Check(normalized[1].Target.StableId == "stable-center-far", "v2 binding wrong in mixed set");
        Check(normalized[2].Target.StableId == "stable-proxy-b", "v3 binding wrong in mixed set");
    }

    private static void SerialAndParallelPicksMatch()
    {
        var fixture = Fixture();
        var engine = new PickEngine();
        var screen = Center(fixture.Scene);
        var query = new PickQuery { Filter = "default", ConcurrencyLimit = 3, MemoryBudgetBytes = 8192 };
        var serial = engine.PickSerial(fixture.Snapshot, screen, query);
        var parallel = engine.PickParallelAsync(fixture.Snapshot, screen, query).GetAwaiter().GetResult();
        Check(serial is not null && parallel is not null, "expected a hit at viewport center");
        CheckHitsEqual(serial!, parallel!);
    }

    private static void LimitedMemoryBudgetWorks()
    {
        var fixture = Fixture();
        var hit = new PickEngine().PickParallelAsync(fixture.Snapshot, Center(fixture.Scene), new PickQuery
        {
            Filter = "default",
            ConcurrencyLimit = 2,
            MemoryBudgetBytes = 4096
        }).GetAwaiter().GetResult();
        Check(hit?.StableId == "stable-center-far", "bounded-memory parallel pick produced the wrong target");
    }

    private static void BoundedParallelReturnsFirstOrderedError()
    {
        var items = new[] { "z", "a", "b" };
        var error = ExpectCode("E_TEST_A", () => BoundedParallel.RunAsync(
            items,
            item => item,
            _ => 1,
            concurrencyLimit: 3,
            memoryBudgetBytes: 16,
            work: (item, _) =>
            {
                if (item is "a" or "b")
                {
                    throw new WorkbenchException(item == "a" ? "E_TEST_A" : "E_TEST_B", "synthetic failure " + item);
                }

                return Task.FromResult(item);
            }).GetAwaiter().GetResult());
        Check(error.Message.Contains('a'), "first ordered error was not the 'a' item");
    }

    private static void OversizedMemoryRequestReturnsResource()
    {
        var fixture = Fixture();
        ExpectCode(ErrorCodes.Resource, () => new PickEngine().PickParallelAsync(fixture.Snapshot, Center(fixture.Scene), new PickQuery
        {
            Filter = "default",
            ConcurrencyLimit = 2,
            MemoryBudgetBytes = 100
        }).GetAwaiter().GetResult());
    }

    private static void FiltersExcludeNearerProxy()
    {
        var fixture = Fixture();
        var engine = new PickEngine();
        var center = Center(fixture.Scene);
        var filtered = engine.PickSerial(fixture.Snapshot, center, new PickQuery { Filter = "default" });
        var visibilityIgnored = engine.PickSerial(fixture.Snapshot, center, new PickQuery
        {
            Filter = "default",
            RespectVisibility = false
        });
        Check(filtered?.StableId == "stable-center-far", "nearest visible pick should be the far proxy");
        Check(visibilityIgnored?.StableId == "stable-hidden-center", "visibility-disabled pick should expose the hidden nearer proxy");
        Check(filtered!.ScreenDistanceMeters > visibilityIgnored!.ScreenDistanceMeters, "filtered proxy was not geometrically nearer");
    }

    private static void ViewportZoomPanPreservesRay()
    {
        var fixture = Fixture();
        var engine = new PickEngine();
        var baseScreen = FindHitScreen(fixture.Scene, "proxy-a", "default");
        var transform = new ViewportTransform(1.7, 43.5, -27.25);
        var displayed = transform.ViewportToScreen(baseScreen, fixture.Scene.Viewport);
        var original = engine.PickSerial(fixture.Snapshot, baseScreen, new PickQuery { Filter = "default" });
        var transformed = engine.PickSerial(fixture.Snapshot, displayed, new PickQuery
        {
            Filter = "default",
            ViewportTransform = transform
        });
        Check(original is not null && transformed is not null, "zoom/pan picks unexpectedly missed");
        Check(original!.StableId == transformed!.StableId, "zoom/pan changed the picked object");
        Check(original.FaceIndex == transformed.FaceIndex, "zoom/pan changed the picked face");
        Check(Vec3.Distance(original.HitPointMeters, transformed.HitPointMeters) < 1e-9, "zoom/pan changed the hit point");
        Check(Vec3.Distance(original.FaceNormal, transformed.FaceNormal) < 1e-12, "zoom/pan changed the face normal");
        Check(Math.Abs(original.ScreenDistanceMeters - transformed.ScreenDistanceMeters) < 1e-12, "zoom/pan changed the screen distance");
        Check(transformed.ScreenPoint == displayed, "transformed pick did not retain the displayed screen point");
    }

    private static void NormalsAreCameraFacing()
    {
        var fixture = Fixture();
        var hit = new PickEngine().PickSerial(fixture.Snapshot, Center(fixture.Scene), new PickQuery { Filter = "default" });
        Check(hit is not null, "center pick failed");
        Check(Math.Abs(hit!.FaceNormal.Length - 1.0) < 1e-12, "face normal is not normalized");
        Check(Vec3.Dot(hit.FaceNormal, hit.Ray.Direction) < 0, "face normal does not point toward the camera side");
    }

    private static void BoxSelectionRespectsFilters()
    {
        var fixture = Fixture();
        var session = new EditorSession(fixture.Scene);
        var selected = session.BoxSelect(
            new ScreenRect(0, 0, fixture.Scene.Viewport.Width - 1, fixture.Scene.Viewport.Height - 1),
            "default",
            additive: false);
        Check(selected.Contains("proxy-a") && selected.Contains("proxy-b"), "box selection missed visible main-layer proxies");
        Check(!selected.Contains("proxy-hidden"), "box selection included a hidden proxy");
        Check(!selected.Contains("proxy-nopick"), "box selection included an unpickable proxy");
        Check(!selected.Contains("proxy-gizmo"), "box selection ignored the active filter");
    }

    private static void DragUndoRestoresPoseAndSelection()
    {
        var fixture = Fixture();
        var session = new EditorSession(fixture.Scene);
        var start = FindHitScreen(fixture.Scene, "proxy-a", "default");
        var hit = session.Pick(start, "default");
        Check(hit?.StableId == "stable-proxy-a", "drag setup pick missed proxy A");
        var originalPosition = session.PoseOf("proxy-a").PositionMeters;
        var originalSelection = session.Selection.ToArray();
        session.BeginDrag(hit!, start, "default", additiveSelection: false);
        var finalHit = session.UpdateDrag(new Vec2(start.X + 55, start.Y + 25));
        session.EndDrag();
        Check(Vec3.Distance(session.PoseOf("proxy-a").PositionMeters, originalPosition) > 0.1, "drag did not move the proxy");
        Check(finalHit is not null, "drag did not recompute a hit");
        Check(session.Undo(out var restored), "undo was not available after drag");
        Check(Vec3.Distance(session.PoseOf("proxy-a").PositionMeters, originalPosition) < 1e-12, "undo did not restore proxy position");
        Check(session.Selection.OrderBy(x => x).SequenceEqual(originalSelection.OrderBy(x => x)), "undo did not restore selection");
        Check(restored.Selection.SetEquals(originalSelection), "undo snapshot selection is wrong");
    }

    private static void ReplaySucceeds()
    {
        var fixture = Fixture();
        var set = RecordLoader.Load(Path.Combine(FixtureRoot(), "records", "mixed.json"));
        var normalized = new RecordNormalizer().NormalizeAll(fixture.Scene, set);
        var state = Path.Combine(_root, "replay-success");
        var report = new ReplayService().RunAsync(fixture.Scene, set, normalized, state).GetAwaiter().GetResult();
        Check(report.Ok, "mixed replay should succeed");
        Check(report.Records.Count == 4 && report.Records.All(r => r.Conflicts.Count == 0), "replay record details are not clean");
        Check(report.ComputedChunkCount == 2, "expected two computed chunks");
        Check(Directory.EnumerateFiles(state, "chunk-*.json", SearchOption.AllDirectories).Count() == 2, "chunk files missing");
        Check(File.Exists(Path.Combine(StateRunDirectory(state, report), "report.json")), "completion report missing");
    }

    private static void ValidChunksAreReused()
    {
        var fixture = Fixture();
        var set = RecordLoader.Load(Path.Combine(FixtureRoot(), "records", "mixed.json"));
        var normalized = new RecordNormalizer().NormalizeAll(fixture.Scene, set);
        var state = Path.Combine(_root, "replay-reuse");
        var service = new ReplayService();
        var first = service.RunAsync(fixture.Scene, set, normalized, state).GetAwaiter().GetResult();
        var second = service.RunAsync(fixture.Scene, set, normalized, state).GetAwaiter().GetResult();
        Check(second.Ok && second.ResumedChunkCount == first.ComputedChunkCount && second.ComputedChunkCount == 0, "valid chunks were not reused");
    }

    private static void ReplayUsesCurrentPoseSnapshot()
    {
        var fixture = Fixture();
        var movedPose = fixture.Scene.ProxyById("proxy-a").Transform with
        {
            PositionMeters = fixture.Scene.ProxyById("proxy-a").Transform.PositionMeters + new Vec3(0.35, 0.15, 0.2)
        };
        var poses = fixture.Scene.Proxies.ToDictionary(
            proxy => proxy.Id,
            proxy => proxy.Id == "proxy-a" ? movedPose : proxy.Transform,
            StringComparer.Ordinal);
        var movedSnapshot = SceneSnapshot.FromPoses(fixture.Scene, poses);
        var screen = FindHitScreen(movedSnapshot, "proxy-a", "default");
        var hit = new PickEngine().PickSerial(movedSnapshot, screen, new PickQuery { Filter = "default" });
        Check(hit is not null && hit.ProxyId == "proxy-a", "moved-pose pick failed");
        var record = hit!.ToCurrentRecord("pose-snapshot-test", fixture.Scene.Viewport);
        var set = SingleRecordSet(fixture.Scene, record, Path.Combine(_root, "pose-snapshot", "records.json"));
        var normalized = new RecordNormalizer().NormalizeAll(fixture.Scene, set);
        var report = new ReplayService().RunAsync(
            fixture.Scene,
            set,
            normalized,
            Path.Combine(_root, "pose-snapshot-state"),
            poses: poses).GetAwaiter().GetResult();
        Check(report.Ok, "replay did not evaluate against the supplied pose snapshot");
    }

    private static void TamperedChunkIsRecomputed()
    {
        var fixture = Fixture();
        var set = RecordLoader.Load(Path.Combine(FixtureRoot(), "records", "mixed.json"));
        var normalized = new RecordNormalizer().NormalizeAll(fixture.Scene, set);
        var state = Path.Combine(_root, "replay-tamper-chunk");
        var service = new ReplayService();
        var first = service.RunAsync(fixture.Scene, set, normalized, state).GetAwaiter().GetResult();
        var journal = new ChunkJournal(state, first.VersionId, "replay", first.ChunkSize);
        var path = journal.ChunkPath(0);
        var node = (JsonObject)JsonNode.Parse(File.ReadAllText(path))!;
        ((JsonObject)((JsonArray)node["results"]!)[0]!)["ok"] = false;
        File.WriteAllText(path, node.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
        var second = service.RunAsync(fixture.Scene, set, normalized, state).GetAwaiter().GetResult();
        Check(second.Ok, "tampered chunk caused an incorrect replay failure");
        Check(second.ComputedChunkCount == 1 && second.ResumedChunkCount == 1, "digest-invalid chunk was not recomputed while the valid chunk was reused");
    }

    private static void CancellationWritesInterrupt()
    {
        var fixture = Fixture();
        var set = RecordLoader.Load(Path.Combine(FixtureRoot(), "records", "mixed.json"));
        var normalized = new RecordNormalizer().NormalizeAll(fixture.Scene, set);
        var state = Path.Combine(_root, "replay-cancel");
        using var cts = new CancellationTokenSource();
        cts.Cancel();
        var report = new ReplayService().RunAsync(fixture.Scene, set, normalized, state, cancellationToken: cts.Token)
            .GetAwaiter().GetResult();
        Check(report.Interrupted && report.InterruptExitCode == 130, "replay did not report an interrupted boundary");
        var run = StateRunDirectory(state, report);
        Check(File.Exists(Path.Combine(run, "interrupt.json")), "interrupt marker missing");
        Check(!File.Exists(Path.Combine(run, "report.json")), "interrupted run wrote a completion report");
        var marker = (JsonObject)JsonNode.Parse(File.ReadAllText(Path.Combine(run, "interrupt.json")))!;
        Check(marker["exitCode"]!.GetValue<int>() == 130, "interrupt exit code mismatch");
        Check(report.ToJson()["exitCode"]!.GetValue<int>() == 130, "report exit code mismatch");
    }

    private static void InterruptedReplayResumes()
    {
        var fixture = Fixture();
        var set = RecordLoader.Load(Path.Combine(FixtureRoot(), "records", "mixed.json"));
        var normalized = new RecordNormalizer().NormalizeAll(fixture.Scene, set);
        var state = Path.Combine(_root, "replay-partial");
        using (var cts = new CancellationTokenSource())
        {
            cts.Cancel();
            var interrupted = new ReplayService().RunAsync(fixture.Scene, set, normalized, state, cancellationToken: cts.Token)
                .GetAwaiter().GetResult();
            Check(interrupted.Interrupted, "initial replay was not interrupted");
        }

        var resumed = new ReplayService().RunAsync(fixture.Scene, set, normalized, state).GetAwaiter().GetResult();
        Check(resumed.Ok && resumed.ComputedChunkCount == 2, "resumed replay did not complete missing chunks");
    }

    private static void TamperedRecordReturnsReplay()
    {
        var fixture = Fixture();
        var set = RecordLoader.Load(Path.Combine(FixtureRoot(), "invalid", "unreproducible.records.json"));
        var normalized = new RecordNormalizer().NormalizeAll(fixture.Scene, set);
        var state = Path.Combine(_root, "replay-tampered-record");
        var report = new ReplayService().RunAsync(fixture.Scene, set, normalized, state).GetAwaiter().GetResult();
        Check(!report.Ok && report.FailedCount == 1, "tampered record was not marked unreproducible");
        var failed = report.Records.Single(r => !r.Ok);
        Check(failed.Version == 3 && failed.RecordIndex == 0, "replay report lost version/index");
        Check(failed.Conflicts.Any(c => c.Field is "hitPointMeters" or "faceIndex"), "replay report lacks conflicting field");
        Check(report.ToJson()["error"]?["code"]?.GetValue<string>() == ErrorCodes.Replay, "replay JSON does not return E_REPLAY");
    }

    private static void MigrationArchivesAndPreserves()
    {
        var fixture = Fixture();
        var directory = Path.Combine(_root, "migration");
        Directory.CreateDirectory(directory);
        var target = Path.Combine(directory, "records.json");
        File.Copy(Path.Combine(FixtureRoot(), "records", "legacy-v1.json"), target);
        var originalBytes = File.ReadAllBytes(target);
        var set = RecordLoader.Load(target);
        var result = new PickRecordStore().MigrateAndWrite(target, fixture.Scene, set);
        Check(result.PreviousFormatVersion == 1, "previous version was not read");
        Check(result.PreviousContentDigest == Sha256Util.HashBytesWithPrefix(originalBytes), "previous content digest mismatch");
        Check(result.ArchivedPath is not null && File.Exists(result.ArchivedPath), "v1 archive was not written");
        Check(File.ReadAllBytes(result.ArchivedPath!).SequenceEqual(originalBytes), "archive content differs from old file");
        var migrated = (JsonObject)JsonNode.Parse(File.ReadAllText(target))!;
        Check(migrated["formatVersion"]!.GetValue<int>() == 3, "migrated envelope version is not current");
        var record = (JsonObject)((JsonArray)migrated["records"]!)[0]!;
        Check(record["version"]!.GetValue<int>() == 3, "record version was not upgraded");
        Check(record["sourceVersion"]!.GetValue<int>() == 1, "source version was not retained");
        Check(record["legacyNote"]?.GetValue<string>() == "preserve-this-unknown-field", "unknown legacy field was lost");
        Check(record.ContainsKey("hitPointCm") && record.ContainsKey("snapshotDigest"), "legacy evidence fields were discarded");
        Check(!Directory.EnumerateFiles(directory, "*.tmp", SearchOption.AllDirectories).Any(), "temporary file leaked");
    }

    private static void InvalidInputWritesNothing()
    {
        var fixture = Fixture();
        var set = RecordLoader.Load(Path.Combine(FixtureRoot(), "invalid", "unit-mismatch.records.json"));
        var output = Path.Combine(_root, "must-not-exist", "records.json");
        ExpectCode(ErrorCodes.Unit, () => new PickRecordStore().MigrateAndWrite(output, fixture.Scene, set));
        Check(!File.Exists(output), "invalid input produced a record file");
        Check(!Directory.Exists(Path.GetDirectoryName(output)!), "invalid input created a record output directory");
    }

    private static void PickScriptWritesCurrentRecords()
    {
        var fixture = Fixture();
        var script = PickScriptLoader.Load(Path.Combine(FixtureRoot(), "pick-script.json"));
        var state = Path.Combine(_root, "pick-state");
        var result = new PickScriptService().RunAsync(fixture.Scene, script, state).GetAwaiter().GetResult();
        Check(!result.Interrupted && result.Records.Count == 5, "pick script did not produce expected hit records");
        Check(result.StepResults.Count == 6, "pick script step result count mismatch");
        var output = Path.Combine(_root, "pick-output", "records.json");
        var write = new PickRecordStore().WriteGeneratedCurrent(output, fixture.Scene, result.Records);
        Check(File.Exists(output) && write.RecordCount == 5, "pick records output missing");
        var loaded = RecordLoader.Load(output);
        var normalized = new RecordNormalizer().NormalizeAll(fixture.Scene, loaded);
        Check(normalized.Count == 5 && normalized.All(r => r.Version == 3), "pick script output is not current-version");
    }

    private static void PickScriptChunksResume()
    {
        var fixture = Fixture();
        var script = PickScriptLoader.Load(Path.Combine(FixtureRoot(), "pick-script.json"));
        var state = Path.Combine(_root, "pick-resume");
        var service = new PickScriptService();
        var first = service.RunAsync(fixture.Scene, script, state).GetAwaiter().GetResult();
        var second = service.RunAsync(fixture.Scene, script, state).GetAwaiter().GetResult();
        Check(!second.Interrupted && second.Records.Count == first.Records.Count, "resumed pick script changed the record set");
        Check(second.ResumedChunkCount == first.ComputedChunkCount && second.ComputedChunkCount == 0, "pick-script chunks were not reused");
    }

    private static void PickScriptCancellationResumes()
    {
        var fixture = Fixture();
        var script = PickScriptLoader.Load(Path.Combine(FixtureRoot(), "pick-script.json"));
        var state = Path.Combine(_root, "pick-cancel");
        var service = new PickScriptService();
        using (var cts = new CancellationTokenSource())
        {
            cts.Cancel();
            var interrupted = service.RunAsync(fixture.Scene, script, state, cancellationToken: cts.Token)
                .GetAwaiter().GetResult();
            Check(interrupted.Interrupted && interrupted.InterruptExitCode == 130, "pick script did not stop at the first chunk boundary");
            var journal = new ChunkJournal(state, interrupted.VersionId, "pick", fixture.Scene.ChunkSize);
            Check(File.Exists(Path.Combine(journal.RunDirectory, "interrupt.json")), "pick script interrupt marker missing");
        }

        var resumed = service.RunAsync(fixture.Scene, script, state).GetAwaiter().GetResult();
        Check(!resumed.Interrupted && resumed.Records.Count == 5, "pick script did not complete after resume");
        Check(resumed.ComputedChunkCount == 3, "pick script resume did not compute all missing chunks");
    }

    private static void AppendUpgradesExisting()
    {
        var fixture = Fixture();
        var directory = Path.Combine(_root, "append");
        Directory.CreateDirectory(directory);
        var target = Path.Combine(directory, "records.json");
        File.Copy(Path.Combine(FixtureRoot(), "records", "legacy-v1.json"), target);
        var hit = new PickEngine().PickSerial(fixture.Snapshot, Center(fixture.Scene), new PickQuery { Filter = "default" })!;
        var result = new PickRecordStore().AppendHit(target, fixture.Scene, hit, "desktop-test-001");
        Check(result.RecordCount == 2, "append did not retain the existing record");
        Check(result.ArchivedPath is not null && File.Exists(result.ArchivedPath), "append did not archive existing v1");
        var set = RecordLoader.Load(target);
        var normalized = new RecordNormalizer().NormalizeAll(fixture.Scene, set);
        Check(normalized.Count == 2 && normalized.All(r => r.Version == 3), "append output is not upgraded to v3");
        Check(normalized.Any(r => r.RecordId == "legacy-a-001") && normalized.Any(r => r.RecordId == "desktop-test-001"), "append lost record identity");
    }

    private static void WriteFailurePreservesOldFile()
    {
        if (!OperatingSystem.IsLinux() && !OperatingSystem.IsMacOS())
        {
            return;
        }

        var fixture = Fixture();
        var directory = Path.Combine(_root, "write-failure");
        Directory.CreateDirectory(directory);
        var target = Path.Combine(directory, "records.json");
        File.Copy(Path.Combine(FixtureRoot(), "records", "current-v3.json"), target);
        var original = File.ReadAllBytes(target);
        var mode = File.GetUnixFileMode(directory);
        try
        {
            var set = RecordLoader.Load(target);
            File.SetUnixFileMode(directory, UnixFileMode.UserRead | UnixFileMode.UserExecute);
            ExpectCode(ErrorCodes.Write, () => new PickRecordStore().MigrateAndWrite(target, fixture.Scene, set));
            Check(File.ReadAllBytes(target).SequenceEqual(original), "failed write modified the old record file");
        }
        finally
        {
            File.SetUnixFileMode(directory, mode);
        }
    }

    private static void CliBehavior()
    {
        var cli = FindCliDll();
        var invalid = RunProcess("dotnet", $"\"{cli}\" validate --scene \"{Path.Combine(FixtureRoot(), "scene.json")}\" --records \"{Path.Combine(FixtureRoot(), "invalid", "unit-mismatch.records.json")}\"");
        Check(invalid.ExitCode == 2, "CLI invalid input exit code should be 2");
        Check(invalid.Output.Contains("\"code\": \"E_UNIT\"", StringComparison.Ordinal), "CLI did not emit E_UNIT JSON");
        Check(invalid.Output.Contains("\"ok\": false", StringComparison.Ordinal), "CLI error response is not JSON");

        var success = RunProcess("dotnet", $"\"{cli}\" validate --scene \"{Path.Combine(FixtureRoot(), "scene.json")}\" --records \"{Path.Combine(FixtureRoot(), "records", "mixed.json")}\"");
        Check(success.ExitCode == 0, "CLI valid input exit code should be 0");
        Check(success.Output.Contains("\"versions\"", StringComparison.Ordinal), "CLI validate output lacks version summary");
    }

    private static void CliSigtermInterrupts()
    {
        if (!OperatingSystem.IsLinux())
        {
            return;
        }

        var fixture = Fixture();
        var source = RecordLoader.Load(Path.Combine(FixtureRoot(), "records", "current-v3.json"));
        var template = (JsonObject)source.Records[0].Node.DeepClone();
        var records = new JsonArray();
        for (var i = 0; i < 400; i++)
        {
            var clone = (JsonObject)template.DeepClone();
            clone["recordId"] = $"signal-{i:D4}";
            records.Add(clone);
        }

        var signalRoot = Path.Combine(_root, "signal");
        Directory.CreateDirectory(signalRoot);
        var recordsPath = Path.Combine(signalRoot, "records.json");
        var envelope = new JsonObject
        {
            ["schema"] = "level-workbench.pick-records",
            ["formatVersion"] = 3,
            ["records"] = records
        };
        File.WriteAllText(recordsPath, envelope.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
        var state = Path.Combine(signalRoot, "state");
        var cli = FindCliDll();
        var arguments = $"\"{cli}\" replay --scene \"{Path.Combine(FixtureRoot(), "scene.json")}\" --records \"{recordsPath}\" --state \"{state}\" --chunk-size 1";
        using var process = new Process
        {
            StartInfo = new ProcessStartInfo
            {
                FileName = "dotnet",
                Arguments = arguments,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false
            }
        };
        process.Start();
        var stdout = process.StandardOutput.ReadToEndAsync();
        var stderr = process.StandardError.ReadToEndAsync();
        var sawChunk = false;
        var deadline = DateTime.UtcNow.AddSeconds(30);
        while (DateTime.UtcNow < deadline && !process.HasExited)
        {
            sawChunk = Directory.Exists(state)
                       && Directory.EnumerateFiles(state, "chunk-*.json", SearchOption.AllDirectories).Any();
            if (sawChunk)
            {
                break;
            }

            Thread.Sleep(5);
        }

        Check(sawChunk, "replay produced no chunk before the signal deadline");
        using (var signal = Process.Start(new ProcessStartInfo
               {
                   FileName = "kill",
                   Arguments = $"-TERM {process.Id}",
                   UseShellExecute = false
               }))
        {
            signal?.WaitForExit();
        }

        Check(process.WaitForExit(30_000), "CLI did not exit after SIGTERM");
        Check(process.ExitCode == 130, $"CLI SIGTERM exit code was {process.ExitCode}, expected 130");
        var output = stdout.GetAwaiter().GetResult() + stderr.GetAwaiter().GetResult();
        Check(output.Contains(ErrorCodes.Cancelled, StringComparison.Ordinal), "CLI SIGTERM output did not contain E_CANCELLED");
        Check(Directory.EnumerateFiles(state, "interrupt.json", SearchOption.AllDirectories).Any(), "SIGTERM did not write interrupt marker");
        Check(!Directory.EnumerateFiles(state, "report.json", SearchOption.AllDirectories).Any(), "interrupted CLI wrote a completion report");
    }

    private static FixtureData Fixture()
    {
        var root = FixtureRoot();
        var scene = SceneLoader.Load(Path.Combine(root, "scene.json"));
        return new FixtureData(scene, SceneSnapshot.FromDocument(scene));
    }

    private static string FixtureRoot() => Path.Combine(_root, "fixture");

    private static void EnsureFixture()
    {
        if (!Directory.Exists(Path.Combine(FixtureRoot(), "records")))
        {
            SampleGenerator.Generate(FixtureRoot());
        }
    }

    private static string StateRunDirectory(string state, ReplayReport report) =>
        Path.Combine(state, "runs", report.VersionId.Replace(':', '-'), $"replay-c{report.ChunkSize}");

    private static Vec2 Center(SceneDocument scene) => new(scene.Viewport.Width / 2.0, scene.Viewport.Height / 2.0);

    private static Vec2 FindHitScreen(SceneDocument scene, string proxyId, string filter) =>
        FindHitScreen(SceneSnapshot.FromDocument(scene), proxyId, filter);

    private static Vec2 FindHitScreen(SceneSnapshot snapshot, string proxyId, string filter)
    {
        var scene = snapshot.Scene;
        var engine = new PickEngine();
        var proxy = scene.Proxies.First(p => p.Id == proxyId);
        var basis = CameraBasis.FromCamera(scene.Camera);
        var instance = snapshot.ById(proxyId);
        var projection = basis.ProjectWorld(instance.Transform.TransformPoint(Vec3.Zero), scene.Viewport, ViewportTransform.Identity, scene.Camera.NearMeters)
                         ?? throw new InvalidOperationException("proxy is not projectable");
        for (var radius = 0; radius <= 60; radius += 8)
        {
            for (var x = -radius; x <= radius; x += 8)
            {
                for (var y = -radius; y <= radius; y += 8)
                {
                    var point = new Vec2(projection.Screen.X + x, projection.Screen.Y + y);
                    if (point.X < 0 || point.X >= scene.Viewport.Width || point.Y < 0 || point.Y >= scene.Viewport.Height)
                    {
                        continue;
                    }

                    var hit = engine.PickSerial(snapshot, point, new PickQuery { Filter = filter });
                    if (hit?.ProxyId == proxyId)
                    {
                        return point;
                    }
                }
            }
        }

        throw new InvalidOperationException("no hit point found for " + proxyId);
    }

    private static LoadedRecordSet SingleRecordSet(SceneDocument scene, JsonObject record, string path)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        var root = new JsonObject
        {
            ["schema"] = "level-workbench.pick-records",
            ["formatVersion"] = 3,
            ["records"] = new JsonArray(record.DeepClone())
        };
        File.WriteAllText(path, root.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
        return RecordLoader.Load(path);
    }

    private static void CheckHitsEqual(PickHit expected, PickHit actual)
    {
        Check(expected.StableId == actual.StableId, "stable id mismatch");
        Check(expected.FaceIndex == actual.FaceIndex, "face index mismatch");
        Check(expected.HitPointMeters == actual.HitPointMeters, "hit point mismatch");
        Check(expected.FaceNormal == actual.FaceNormal, "normal mismatch");
        Check(expected.ScreenDistanceMeters == actual.ScreenDistanceMeters, "screen distance mismatch");
        Check(expected.ScreenPoint == actual.ScreenPoint, "screen point mismatch");
    }

    private static WorkbenchException ExpectCode(string code, Action action)
    {
        try
        {
            action();
        }
        catch (WorkbenchException ex)
        {
            Check(ex.Code == code, $"expected {code}, got {ex.Code}: {ex.Message}");
            return ex;
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"expected {code}, got {ex.GetType().Name}: {ex.Message}", ex);
        }

        throw new InvalidOperationException($"expected {code}, but no exception was thrown");
    }

    private static void Check(bool condition, string message)
    {
        if (!condition)
        {
            throw new InvalidOperationException(message);
        }
    }

    private static Dictionary<string, string> HashTree(string root) => Directory
        .EnumerateFiles(root, "*", SearchOption.AllDirectories)
        .OrderBy(path => Path.GetRelativePath(root, path), StringComparer.Ordinal)
        .ToDictionary(
            path => Path.GetRelativePath(root, path),
            path => Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(path))).ToLowerInvariant(),
            StringComparer.Ordinal);

    private static string FindCliDll()
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null)
        {
            var candidate = Path.Combine(directory.FullName, "src", "LevelWorkbench.Cli", "bin", "Debug", "net8.0", "LevelWorkbench.Cli.dll");
            if (File.Exists(candidate))
            {
                return candidate;
            }

            directory = directory.Parent;
        }

        throw new FileNotFoundException("LevelWorkbench.Cli.dll not found; build the solution first.");
    }

    private static (int ExitCode, string Output) RunProcess(string fileName, string arguments)
    {
        using var process = new Process
        {
            StartInfo = new ProcessStartInfo
            {
                FileName = fileName,
                Arguments = arguments,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false
            }
        };
        process.Start();
        var output = process.StandardOutput.ReadToEnd();
        var error = process.StandardError.ReadToEnd();
        process.WaitForExit();
        return (process.ExitCode, output + error);
    }

    private static void TryDelete(string path)
    {
        try
        {
            if (Directory.Exists(path))
            {
                Directory.Delete(path, recursive: true);
            }
        }
        catch
        {
            // Tests keep their artifacts on failure for diagnosis.
        }
    }

    private sealed record FixtureData(SceneDocument Scene, SceneSnapshot Snapshot);
}
