using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using LevelWorkbench.Core;

namespace LevelWorkbench.Cli;

public static class Program
{
    private static readonly JsonSerializerOptions Pretty = new() { WriteIndented = true };

    public static int Main(string[] args)
    {
        Console.OutputEncoding = Encoding.UTF8;
        using var cancellation = new CancellationTokenSource();
        ConsoleCancelEventHandler cancelHandler = (_, e) =>
        {
            e.Cancel = true;
            cancellation.Cancel();
        };
        Console.CancelKeyPress += cancelHandler;
        PosixSignalRegistration? sigterm = null;
        try
        {
            sigterm = PosixSignalRegistration.Create(PosixSignal.SIGTERM, context =>
            {
                context.Cancel = true;
                cancellation.Cancel();
            });
        }
        catch (PlatformNotSupportedException)
        {
            // SIGTERM registration is optional on platforms that do not expose POSIX signals.
        }

        try
        {
            return Run(args, cancellation.Token);
        }
        catch (WorkbenchException ex)
        {
            Write(ex.ToErrorNode());
            return ex.Code == ErrorCodes.Cancelled ? 130 : 2;
        }
        catch (OperationCanceledException)
        {
            Write(new JsonObject
            {
                ["ok"] = false,
                ["error"] = new JsonObject
                {
                    ["code"] = ErrorCodes.Cancelled,
                    ["message"] = "Operation cancelled at a chunk boundary.",
                    ["details"] = new JsonObject { ["exitCode"] = 130 }
                }
            });
            return 130;
        }
        catch (Exception ex)
        {
            Write(new JsonObject
            {
                ["ok"] = false,
                ["error"] = new JsonObject
                {
                    ["code"] = ErrorCodes.Internal,
                    ["message"] = ex.Message,
                    ["details"] = new JsonObject { ["type"] = ex.GetType().Name }
                }
            });
            return 1;
        }
        finally
        {
            Console.CancelKeyPress -= cancelHandler;
            sigterm?.Dispose();
        }
    }

    private static int Run(string[] args, CancellationToken cancellationToken)
    {
        if (args.Length == 0)
        {
            throw new WorkbenchException(ErrorCodes.Schema, "A command is required: validate, pick, replay, or migrate.");
        }

        var command = args[0];
        var options = ParseOptions(args.Skip(1).ToArray());
        return command switch
        {
            "validate" => Validate(options),
            "pick" => Pick(options, cancellationToken),
            "replay" => Replay(options, cancellationToken),
            "migrate" or "normalize" => Migrate(options),
            "help" or "--help" or "-h" => Help(),
            _ => throw new WorkbenchException(ErrorCodes.Schema, "Unknown command.", new JsonObject
            {
                ["command"] = command,
                ["known"] = new JsonArray("validate", "pick", "replay", "migrate")
            })
        };
    }

    private static int Validate(IReadOnlyDictionary<string, string> options)
    {
        var scenePath = RequireOption(options, "--scene");
        var recordsPath = RequireOption(options, "--records");
        var scene = SceneLoader.Load(scenePath);
        var records = RecordLoader.Load(recordsPath);
        var normalized = new RecordNormalizer().NormalizeAll(scene, records);
        Write(new JsonObject
        {
            ["ok"] = true,
            ["scene"] = new JsonObject
            {
                ["name"] = scene.Name,
                ["proxyCount"] = scene.Proxies.Count,
                ["layerCount"] = scene.Layers.Count,
                ["viewport"] = new JsonObject { ["width"] = scene.Viewport.Width, ["height"] = scene.Viewport.Height },
                ["digest"] = SceneSnapshot.FromDocument(scene).ComputeDigest()
            },
            ["records"] = new JsonObject
            {
                ["files"] = records.Files.Count,
                ["count"] = records.Records.Count,
                ["versions"] = new JsonArray(records.Records
                    .GroupBy(record => record.Version)
                    .OrderBy(group => group.Key)
                    .Select(group => (JsonNode?)new JsonObject { ["version"] = group.Key, ["count"] = group.Count() })
                    .ToArray()),
                ["canonicalDigest"] = records.CanonicalDigest,
                ["normalized"] = new JsonArray(normalized.Select(record => (JsonNode?)new JsonObject
                {
                    ["recordIndex"] = record.RecordIndex,
                    ["recordId"] = record.RecordId,
                    ["version"] = record.Version,
                    ["stableId"] = record.Target.StableId,
                    ["proxyNumber"] = record.Target.ProxyNumber,
                    ["faceIndex"] = record.FaceIndex,
                    ["hitPointMeters"] = SceneDigest.VecNode(record.HitPointMeters)
                }).ToArray())
            }
        });
        return 0;
    }

    private static int Pick(IReadOnlyDictionary<string, string> options, CancellationToken cancellationToken)
    {
        var scene = SceneLoader.Load(RequireOption(options, "--scene"));
        var script = PickScriptLoader.Load(RequireOption(options, "--script"));
        var stateDirectory = RequireOption(options, "--state");
        var outputPath = RequireOption(options, "--out");
        var chunkSize = OptionalInt(options, "--chunk-size");
        var result = new PickScriptService().RunAsync(scene, script, stateDirectory, chunkSize, cancellationToken)
            .GetAwaiter().GetResult();
        if (result.Interrupted)
        {
            Write(new JsonObject
            {
                ["ok"] = false,
                ["interrupted"] = true,
                ["versionId"] = result.VersionId,
                ["exitCode"] = 130,
                ["error"] = new JsonObject
                {
                    ["code"] = ErrorCodes.Cancelled,
                    ["message"] = "Pick script interrupted at a chunk boundary. State is resumable.",
                    ["details"] = new JsonObject { ["stateDirectory"] = Path.GetFullPath(stateDirectory) }
                }
            });
            return 130;
        }

        var write = new PickRecordStore().WriteGeneratedCurrent(outputPath, scene, result.Records);
        Write(new JsonObject
        {
            ["ok"] = true,
            ["operation"] = "pick",
            ["versionId"] = result.VersionId,
            ["stepCount"] = script.Steps.Count,
            ["recordCount"] = result.Records.Count,
            ["resumedChunkCount"] = result.ResumedChunkCount,
            ["computedChunkCount"] = result.ComputedChunkCount,
            ["recordsFile"] = write.Path,
            ["contentDigest"] = write.ContentDigest,
            ["steps"] = result.StepResults.DeepClone()
        });
        return 0;
    }

    private static int Replay(IReadOnlyDictionary<string, string> options, CancellationToken cancellationToken)
    {
        var scene = SceneLoader.Load(RequireOption(options, "--scene"));
        var records = RecordLoader.Load(RequireOption(options, "--records"));
        var normalized = new RecordNormalizer().NormalizeAll(scene, records);
        var stateDirectory = RequireOption(options, "--state");
        var chunkSize = OptionalInt(options, "--chunk-size");
        var reportPath = Optional(options, "--report");
        var report = new ReplayService().RunAsync(scene, records, normalized, stateDirectory, chunkSize, cancellationToken)
            .GetAwaiter().GetResult();
        var node = report.ToJson();
        if (report.Interrupted)
        {
            node["ok"] = false;
            node["error"] = new JsonObject
            {
                ["code"] = ErrorCodes.Cancelled,
                ["message"] = "Replay interrupted at a chunk boundary. State is resumable.",
                ["details"] = new JsonObject
                {
                    ["stateDirectory"] = Path.GetFullPath(stateDirectory),
                    ["versionId"] = report.VersionId,
                    ["exitCode"] = 130
                }
            };
            Write(node);
            return 130;
        }

        Write(node);
        if (!string.IsNullOrWhiteSpace(reportPath))
        {
            AtomicFile.WriteAllText(reportPath, node.ToJsonString(Pretty) + "\n");
        }

        return report.Ok ? 0 : 2;
    }

    private static int Migrate(IReadOnlyDictionary<string, string> options)
    {
        var scene = SceneLoader.Load(RequireOption(options, "--scene"));
        var records = RecordLoader.Load(RequireOption(options, "--records"));
        var outputPath = RequireOption(options, "--out");
        var result = new PickRecordStore().MigrateAndWrite(outputPath, scene, records);
        Write(new JsonObject
        {
            ["ok"] = true,
            ["operation"] = "migrate",
            ["path"] = result.Path,
            ["recordCount"] = result.RecordCount,
            ["formatVersion"] = RecordVersions.Current,
            ["contentDigest"] = result.ContentDigest,
            ["previousFormatVersion"] = result.PreviousFormatVersion,
            ["previousContentDigest"] = result.PreviousContentDigest,
            ["archivedPath"] = result.ArchivedPath
        });
        return 0;
    }

    private static int Help()
    {
        Write(new JsonObject
        {
            ["ok"] = true,
            ["commands"] = new JsonObject
            {
                ["validate"] = "validate --scene <scene.json> --records <file-or-directory>",
                ["pick"] = "pick --scene <scene.json> --script <script.json> --state <state-dir> --out <records.json> [--chunk-size N]",
                ["replay"] = "replay --scene <scene.json> --records <file-or-directory> --state <state-dir> [--report <report.json>] [--chunk-size N]",
                ["migrate"] = "migrate --scene <scene.json> --records <file-or-directory> --out <records-v3.json>"
            }
        });
        return 0;
    }

    private static IReadOnlyDictionary<string, string> ParseOptions(string[] args)
    {
        var result = new Dictionary<string, string>(StringComparer.Ordinal);
        for (var i = 0; i < args.Length; i++)
        {
            if (!args[i].StartsWith("--", StringComparison.Ordinal))
            {
                throw new WorkbenchException(ErrorCodes.Schema, "Expected an option name beginning with --.", new JsonObject
                {
                    ["value"] = args[i]
                });
            }

            if (i + 1 >= args.Length || args[i + 1].StartsWith("--", StringComparison.Ordinal))
            {
                throw new WorkbenchException(ErrorCodes.Schema, "Option is missing its value.", new JsonObject
                {
                    ["option"] = args[i]
                });
            }

            result[args[i]] = args[++i];
        }

        return result;
    }

    private static string RequireOption(IReadOnlyDictionary<string, string> options, string name)
    {
        if (!options.TryGetValue(name, out var value) || string.IsNullOrWhiteSpace(value))
        {
            throw new WorkbenchException(ErrorCodes.Schema, "Required option is missing.", new JsonObject { ["option"] = name });
        }

        return value;
    }

    private static string? Optional(IReadOnlyDictionary<string, string> options, string name) =>
        options.TryGetValue(name, out var value) ? value : null;

    private static int? OptionalInt(IReadOnlyDictionary<string, string> options, string name)
    {
        if (!options.TryGetValue(name, out var value))
        {
            return null;
        }

        if (!int.TryParse(value, out var parsed) || parsed <= 0)
        {
            throw new WorkbenchException(ErrorCodes.Schema, "Option must be a positive integer.", new JsonObject { ["option"] = name, ["value"] = value });
        }

        return parsed;
    }

    private static void Write(JsonNode node) => Console.WriteLine(node.ToJsonString(Pretty));
}
