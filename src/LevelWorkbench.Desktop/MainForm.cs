using System.Text.Json;
using LevelWorkbench.Core;

namespace LevelWorkbench.Desktop;

public sealed class MainForm : Form
{
    private readonly ViewportControl _viewport = new() { Dock = DockStyle.Fill };
    private readonly ListBox _recordList = new() { Dock = DockStyle.Fill };
    private readonly TextBox _details = new() { Dock = DockStyle.Fill, Multiline = true, ScrollBars = ScrollBars.Both, ReadOnly = true };
    private readonly ToolStripStatusLabel _status = new("Ready");
    private readonly ComboBox _filter = new() { DropDownStyle = ComboBoxStyle.DropDownList, Width = 110 };
    private readonly Label _sceneLabel = new() { AutoSize = true, Text = "No scene" };
    private readonly IReadOnlyDictionary<string, string> _options;
    private string _scenePath;
    private string _recordsPath;
    private string _scriptPath;
    private string _statePath;
    private string _outputPath;
    private SceneDocument? _scene;
    private EditorSession? _session;
    private IReadOnlyList<NormalizedPickRecord> _normalizedRecords = Array.Empty<NormalizedPickRecord>();

    public MainForm(IReadOnlyDictionary<string, string> options)
    {
        _options = options;
        Text = "Level Workbench · deterministic pick replay";
        Width = 1380;
        Height = 860;
        MinimumSize = new Size(1000, 640);
        StartPosition = FormStartPosition.CenterScreen;
        BackColor = Color.FromArgb(22, 27, 35);
        ForeColor = Color.FromArgb(230, 236, 245);

        var sampleRoot = FindSampleRoot();
        _scenePath = Option("--scene", Path.Combine(sampleRoot, "scene.json"));
        _recordsPath = Option("--records", Path.Combine(sampleRoot, "records", "mixed.json"));
        _scriptPath = Option("--script", Path.Combine(sampleRoot, "pick-script.json"));
        _statePath = Option("--state", Path.Combine(Environment.CurrentDirectory, "workbench-state"));
        _outputPath = Option("--out", Path.Combine(Environment.CurrentDirectory, "workbench-output", "picked-records.json"));

        BuildLayout();
        _filter.Items.AddRange(new object[] { "default", "gizmo", "(any)" });
        _filter.SelectedIndex = 0;
        _filter.SelectedIndexChanged += (_, _) =>
        {
            _viewport.ActiveFilter = _filter.SelectedItem?.ToString() is "(any)" or null ? null : _filter.SelectedItem.ToString();
            _viewport.Invalidate();
        };
        _viewport.ActiveFilter = "default";
        _viewport.StatusChanged += message => SetStatus(message);
        _viewport.HitCommitted += CommitHit;
        Shown += async (_, _) => await LoadInitialAsync();
    }

    private void BuildLayout()
    {
        var toolbar = new FlowLayoutPanel
        {
            Dock = DockStyle.Top,
            Height = 46,
            Padding = new Padding(8, 7, 8, 5),
            WrapContents = false,
            BackColor = Color.FromArgb(28, 35, 46)
        };
        toolbar.Controls.Add(Button("Open scene", async () => await OpenSceneAsync()));
        toolbar.Controls.Add(Button("Load records", async () => await LoadRecordsAsync()));
        toolbar.Controls.Add(Button("Replay history", async () => await ReplayAsync()));
        toolbar.Controls.Add(Button("Run script", async () => await RunScriptAsync()));
        toolbar.Controls.Add(Button("Migrate records", async () => await MigrateAsync()));
        toolbar.Controls.Add(Button("Undo (Ctrl+Z)", () =>
        {
            if (_session is not null && _session.Undo(out _))
            {
                _viewport.PreviewHit = null;
                _viewport.Invalidate();
                SetStatus("Undo restored positions and selection.");
            }
        }));
        toolbar.Controls.Add(Button("Reset view", () =>
        {
            if (_session is not null)
            {
                _session.SetViewportTransform(ViewportTransform.Identity);
                _viewport.Invalidate();
            }
        }));
        toolbar.Controls.Add(new Label { Text = "Filter", AutoSize = true, Padding = new Padding(12, 10, 2, 0), ForeColor = Color.Gainsboro });
        toolbar.Controls.Add(_filter);
        toolbar.Controls.Add(_sceneLabel);
        Controls.Add(toolbar);

        var split = new SplitContainer
        {
            Dock = DockStyle.Fill,
            SplitterDistance = 1010,
            BackColor = Color.FromArgb(34, 41, 53)
        };
        split.Panel1.Controls.Add(_viewport);
        var right = new SplitContainer
        {
            Dock = DockStyle.Fill,
            Orientation = Orientation.Horizontal,
            SplitterDistance = 280
        };
        var recordPanel = new Panel { Dock = DockStyle.Fill, Padding = new Padding(4) };
        var recordHeader = new Label { Text = "Loaded history records", Dock = DockStyle.Top, Height = 24, ForeColor = Color.Gainsboro };
        recordPanel.Controls.Add(_recordList);
        recordPanel.Controls.Add(recordHeader);
        var detailPanel = new Panel { Dock = DockStyle.Fill, Padding = new Padding(4) };
        var detailHeader = new Label { Text = "Diagnostics / replay report", Dock = DockStyle.Top, Height = 24, ForeColor = Color.Gainsboro };
        detailPanel.Controls.Add(_details);
        detailPanel.Controls.Add(detailHeader);
        right.Panel1.Controls.Add(recordPanel);
        right.Panel2.Controls.Add(detailPanel);
        split.Panel2.Controls.Add(right);
        Controls.Add(split);

        var statusStrip = new StatusStrip { BackColor = Color.FromArgb(28, 35, 46) };
        statusStrip.Items.Add(_status);
        Controls.Add(statusStrip);
    }

    private Button Button(string text, Action action)
    {
        var button = new Button
        {
            Text = text,
            AutoSize = true,
            Height = 29,
            FlatStyle = FlatStyle.Flat,
            BackColor = Color.FromArgb(48, 60, 78),
            ForeColor = Color.White
        };
        button.FlatAppearance.BorderColor = Color.FromArgb(86, 108, 140);
        button.Click += (_, _) => action();
        return button;
    }

    private async Task LoadInitialAsync()
    {
        try
        {
            if (File.Exists(_scenePath))
            {
                LoadScene(_scenePath);
            }
            else
            {
                SetStatus("No scene found. Use Open scene.");
            }

            if (_scene is not null && (File.Exists(_recordsPath) || Directory.Exists(_recordsPath)))
            {
                await LoadRecordsAsync();
            }
        }
        catch (Exception ex)
        {
            ShowError(ex);
        }
    }

    private async Task OpenSceneAsync()
    {
        using var dialog = new OpenFileDialog
        {
            Filter = "Scene JSON (*.json)|*.json|All files (*.*)|*.*",
            InitialDirectory = Path.GetDirectoryName(Path.GetFullPath(_scenePath))
        };
        if (dialog.ShowDialog(this) == DialogResult.OK)
        {
            await Task.Yield();
            LoadScene(dialog.FileName);
        }
    }

    private void LoadScene(string path)
    {
        var scene = SceneLoader.Load(path);
        _scene = scene;
        _scenePath = path;
        _session = new EditorSession(scene);
        _viewport.Session = _session;
        _viewport.PreviewHit = null;
        _sceneLabel.Text = $"  {scene.Name} · {scene.Proxies.Count} proxies · {scene.Layers.Count} layers";
        _details.Text = JsonSerializer.Serialize(new
        {
            scene = scene.Name,
            unit = scene.Unit,
            viewport = new { scene.Viewport.Width, scene.Viewport.Height },
            concurrencyLimit = scene.ConcurrencyLimit,
            memoryBudgetBytes = scene.MemoryBudgetBytes,
            chunkSize = scene.ChunkSize,
            sceneDigest = SceneSnapshot.FromDocument(scene).ComputeDigest()
        }, new JsonSerializerOptions { WriteIndented = true });
        SetStatus($"Loaded scene {Path.GetFileName(path)}");
        _viewport.Invalidate();
    }

    private async Task LoadRecordsAsync()
    {
        try
        {
            if (_scene is null)
            {
                throw new WorkbenchException(ErrorCodes.Schema, "Load a scene before loading records.");
            }

            _recordsPath = RequireExistingPath(_recordsPath);
            var records = RecordLoader.Load(_recordsPath);
            _normalizedRecords = new RecordNormalizer().NormalizeAll(_scene, records);
            _recordList.Items.Clear();
            foreach (var record in _normalizedRecords)
            {
                _recordList.Items.Add(
                    $"#{record.RecordIndex:D2} v{record.Version} {record.RecordId} → {record.Target.StableId} f{record.FaceIndex}");
            }

            _details.Text = JsonSerializer.Serialize(new
            {
                files = records.Files.Select(file => Path.GetFileName(file.Path)).ToArray(),
                count = records.Records.Count,
                canonicalDigest = records.CanonicalDigest,
                versions = records.Records.GroupBy(record => record.Version).ToDictionary(group => group.Key.ToString(), group => group.Count())
            }, new JsonSerializerOptions { WriteIndented = true });
            SetStatus($"Loaded {_normalizedRecords.Count} records from {Path.GetFileName(_recordsPath)}");
            await Task.CompletedTask;
        }
        catch (Exception ex)
        {
            ShowError(ex);
        }
    }

    private async Task ReplayAsync()
    {
        if (_scene is null)
        {
            ShowError(new WorkbenchException(ErrorCodes.Schema, "Load a scene first."));
            return;
        }

        try
        {
            SetStatus("Replaying history in chunk mode...");
            _recordsPath = RequireExistingPath(_recordsPath);
            var records = RecordLoader.Load(_recordsPath);
            var normalized = new RecordNormalizer().NormalizeAll(_scene, records);
            var report = await Task.Run(() => new ReplayService()
                .RunAsync(
                    _scene,
                    records,
                    normalized,
                    _statePath,
                    cancellationToken: CancellationToken.None,
                    poses: _session?.Poses)
                .GetAwaiter().GetResult());
            _details.Text = report.ToJson().ToJsonString(new JsonSerializerOptions { WriteIndented = true });
            SetStatus(report.Ok
                ? $"Replay complete · {report.Records.Count} records, {report.ComputedChunkCount} computed chunks, {report.ResumedChunkCount} resumed"
                : $"Replay failed · {report.FailedCount} record(s) unreproducible");
        }
        catch (Exception ex)
        {
            ShowError(ex);
        }
    }

    private async Task MigrateAsync()
    {
        if (_scene is null)
        {
            ShowError(new WorkbenchException(ErrorCodes.Schema, "Load a scene first."));
            return;
        }

        try
        {
            _recordsPath = RequireExistingPath(_recordsPath);
            var records = RecordLoader.Load(_recordsPath);
            var target = Path.Combine(Environment.CurrentDirectory, "workbench-output", "records-v3.json");
            var result = await Task.Run(() => new PickRecordStore().MigrateAndWrite(target, _scene, records));
            _details.Text = JsonSerializer.Serialize(result, new JsonSerializerOptions { WriteIndented = true });
            SetStatus(result.ArchivedPath is null
                ? "Migrated records to v3 atomically."
                : $"Migrated records; archived previous version to {Path.GetFileName(result.ArchivedPath)}");
        }
        catch (Exception ex)
        {
            ShowError(ex);
        }
    }

    private async Task RunScriptAsync()
    {
        if (_scene is null)
        {
            ShowError(new WorkbenchException(ErrorCodes.Schema, "Load a scene first."));
            return;
        }

        try
        {
            if (!File.Exists(_scriptPath))
            {
                using var dialog = new OpenFileDialog
                {
                    Filter = "Pick script JSON (*.json)|*.json|All files (*.*)|*.*",
                    InitialDirectory = Environment.CurrentDirectory
                };
                if (dialog.ShowDialog(this) != DialogResult.OK)
                {
                    return;
                }

                _scriptPath = dialog.FileName;
            }

            var script = PickScriptLoader.Load(_scriptPath);
            SetStatus("Running deterministic pick script in chunk mode...");
            var result = await Task.Run(() => new PickScriptService()
                .RunAsync(_scene, script, _statePath, cancellationToken: CancellationToken.None)
                .GetAwaiter().GetResult());
            if (result.Interrupted)
            {
                SetStatus("Pick script interrupted at a chunk boundary; state can be resumed.");
                return;
            }

            var target = Path.Combine(Environment.CurrentDirectory, "workbench-output", "script-picked-records.json");
            var write = new PickRecordStore().WriteGeneratedCurrent(target, _scene, result.Records);
            _details.Text = result.StepResults.ToJsonString(new JsonSerializerOptions { WriteIndented = true });
            _recordsPath = write.Path;
            SetStatus($"Script complete · {script.Steps.Count} steps, {write.RecordCount} records → {Path.GetFileName(write.Path)}");
            await LoadRecordsAsync();
        }
        catch (Exception ex)
        {
            ShowError(ex);
        }
    }

    private void CommitHit(PickHit hit)
    {
        try
        {
            if (_scene is null)
            {
                return;
            }

            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(_outputPath))!);
            var recordId = $"desktop-{DateTime.UtcNow:yyyyMMddTHHmmssfff}-{Guid.NewGuid():N}";
            var result = new PickRecordStore().AppendHit(_outputPath, _scene, hit, recordId);
            SetStatus($"Wrote {hit.StableId} f{hit.FaceIndex} to {Path.GetFileName(result.Path)} · {result.ContentDigest[..19]}…");
        }
        catch (Exception ex)
        {
            ShowError(ex);
        }
    }

    private void ShowError(Exception ex)
    {
        var node = ex is WorkbenchException workbench
            ? workbench.ToErrorNode()
            : new System.Text.Json.Nodes.JsonObject
            {
                ["ok"] = false,
                ["error"] = new System.Text.Json.Nodes.JsonObject
                {
                    ["code"] = ErrorCodes.Internal,
                    ["message"] = ex.Message
                }
            };
        var text = node.ToJsonString(new JsonSerializerOptions { WriteIndented = true });
        _details.Text = text;
        SetStatus(ex is WorkbenchException wb ? $"{wb.Code}: {wb.Message}" : ex.Message);
        MessageBox.Show(this, text, "Level Workbench", MessageBoxButtons.OK, MessageBoxIcon.Error);
    }

    private void SetStatus(string text) => _status.Text = text;

    private string RequireExistingPath(string path)
    {
        if (File.Exists(path) || Directory.Exists(path))
        {
            return path;
        }

        using var dialog = new OpenFileDialog
        {
            Filter = "Pick records (*.json)|*.json|All files (*.*)|*.*",
            InitialDirectory = Directory.Exists(Path.GetDirectoryName(Path.GetFullPath(path)))
                ? Path.GetDirectoryName(Path.GetFullPath(path))
                : Environment.CurrentDirectory
        };
        if (dialog.ShowDialog(this) == DialogResult.OK)
        {
            _recordsPath = dialog.FileName;
            return _recordsPath;
        }

        throw new WorkbenchException(ErrorCodes.Schema, "Record file was not selected.", new System.Text.Json.Nodes.JsonObject { ["path"] = path });
    }

    private string Option(string name, string fallback) => _options.TryGetValue(name, out var value) ? value : fallback;

    private static string FindSampleRoot()
    {
        var candidates = new[]
        {
            Path.Combine(Environment.CurrentDirectory, "sample-data"),
            Path.Combine(AppContext.BaseDirectory, "sample-data"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "sample-data")
        };
        return candidates.FirstOrDefault(Directory.Exists) ?? Path.Combine(Environment.CurrentDirectory, "sample-data");
    }
}
