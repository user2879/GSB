package engine

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"time"

	"flowledger/internal/ledger"
	"flowledger/internal/model"
)

var ErrSimulatedCrash = &model.CodedError{
	Code:    model.CodeCrash,
	Message: "simulated forced termination: one window was durably marked running but has no committed result; no exit report was produced",
}

type pending struct {
	task    model.WindowJob
	ordinal int64
}

type runningTask struct {
	pending
	startMS int64
}

type computed struct {
	windowStartMS int64
	result        model.WindowResult
	decisions     []model.DecisionRecord
	resultJSON    string
	resultDigest  string
	err           error
}

func Run(ctx context.Context, store *ledger.Store, loaded *model.Loaded, options model.Options, crashAfter int) (RunOutcome, error) {
	if err := options.Validate(); err != nil {
		return RunOutcome{}, err
	}
	stream, err := model.Generate(loaded)
	if err != nil {
		return RunOutcome{}, err
	}
	manifest := model.BuildManifest(loaded, stream, options)
	manifestJSON, err := model.CanonicalJSON(manifest)
	if err != nil {
		return RunOutcome{}, err
	}
	manifestDigest, err := model.DigestJSON(manifest)
	if err != nil {
		return RunOutcome{}, err
	}
	optionsJSON, err := model.CanonicalJSON(options)
	if err != nil {
		return RunOutcome{}, err
	}
	optionsDigest, err := model.DigestJSON(options)
	if err != nil {
		return RunOutcome{}, err
	}
	runID := "run-" + manifestDigest[:20]
	existing, found, err := store.GetRun(runID)
	if err != nil {
		return RunOutcome{}, err
	}
	wasExisting := found
	if found {
		if existing.SemanticDigest != manifestDigest {
			return RunOutcome{}, model.NewError(model.CodeLedger, "/runs/semantic_digest", "run identity collision or corrupted manifest digest")
		}
		if existing.Stale || !existing.IsCurrent {
			return RunOutcome{}, model.NewError(model.CodeSpec, "/run_id", "the requested run is a stale historical view; current views must be created by a newer monotonic version")
		}
		if existing.Status == "completed" {
			report, err := loadStoredReport(existing)
			if err != nil {
				return RunOutcome{}, err
			}
			return RunOutcome{Report: report, RunID: runID, ReusedReport: true, Resumed: wasExisting}, nil
		}
	}
	jobs := model.BuildWindowJobs(stream, options.WindowMS())
	if !found {
		latest, ok, err := store.LatestCurrentRun()
		if err != nil {
			return RunOutcome{}, err
		}
		if ok {
			if err := validateMonotonicVersions(latest, loaded); err != nil {
				return RunOutcome{}, err
			}
		}
		cs, err := ComputeChangeSet(ptrRun(latest, ok), manifest, stream, loaded)
		if err != nil {
			return RunOutcome{}, err
		}
		changeJSON, err := model.CanonicalJSON(cs)
		if err != nil {
			return RunOutcome{}, err
		}
		newRun := ledger.NewRun{
			RunID:          runID,
			FormatVersion:  model.RunFormatVersion,
			SpecVersion:    loaded.Spec.SpecVersion,
			PolicyVersion:  loaded.Policy.PolicyVersion,
			SpecDigest:     loaded.SpecDigest,
			PolicyDigest:   loaded.PolicyDigest,
			ManifestJSON:   string(manifestJSON),
			ManifestDigest: manifestDigest,
			OptionsJSON:    string(optionsJSON),
			OptionsDigest:  optionsDigest,
			SemanticDigest: manifestDigest,
			StreamDigest:   stream.StreamDigest,
			ChangeSetJSON:  string(changeJSON),
			TotalWindows:   int64(len(jobs)),
		}
		if err := store.InsertRun(newRun, jobs, ledger.NowMS()); err != nil {
			return RunOutcome{}, err
		}
		existing, found, err = store.GetRun(runID)
		if err != nil || !found {
			if err == nil {
				err = model.NewError(model.CodeLedger, "/runs", "new run vanished after insertion")
			}
			return RunOutcome{}, err
		}
	}
	if existing.Status == "completed" {
		report, err := loadStoredReport(existing)
		if err != nil {
			return RunOutcome{}, err
		}
		return RunOutcome{Report: report, RunID: runID, ReusedReport: true, Resumed: wasExisting}, nil
	}
	if existing.Status == "superseded" {
		return RunOutcome{}, model.NewError(model.CodeSpec, "/run_id", "run was superseded by a newer current view")
	}
	recovered, err := store.MarkRunningUnknown(runID, ledger.NowMS())
	if err != nil {
		return RunOutcome{}, err
	}
	segmentID := fmt.Sprintf("seg-%d-%d", os.Getpid(), time.Now().UTC().UnixMilli())
	if err := store.CreateSegment(runID, segmentID, ledger.NowMS()); err != nil {
		return RunOutcome{}, err
	}
	segmentStart := time.Now()
	monoNow := func() int64 { return time.Since(segmentStart).Milliseconds() }
	tasks, err := store.ListTasks(runID)
	if err != nil {
		return RunOutcome{}, err
	}
	if err := reconcileTasks(tasks, jobs); err != nil {
		return RunOutcome{}, err
	}
	jobByStart := make(map[int64]model.WindowJob, len(jobs))
	for _, job := range jobs {
		jobByStart[job.WindowStartMS] = job
	}
	var queue []pending
	completed := int64(0)
	for _, task := range tasks {
		if task.Status == "completed" {
			completed++
			continue
		}
		job, ok := jobByStart[task.WindowStartMS]
		if !ok {
			return RunOutcome{}, model.NewError(model.CodeLedger, "/window_tasks", fmt.Sprintf("unknown window %d", task.WindowStartMS))
		}
		queue = append(queue, pending{task: job, ordinal: task.Ordinal})
	}
	if completed == int64(len(jobs)) {
		report, err := finalizeAndLoad(store, runID)
		if err != nil {
			return RunOutcome{}, err
		}
		return RunOutcome{Report: report, RunID: runID, Recovered: recovered, Resumed: wasExisting}, nil
	}
	computedCh := make(chan computed, options.GlobalConcurrency)
	active := make(map[int64]runningTask)
	keyActive := make(map[string]int)
	globalActive := 0
	completedThisInvocation := 0
	release := func(t runningTask) {
		delete(active, t.task.WindowStartMS)
		globalActive--
		for _, key := range t.task.Keys {
			keyActive[key]--
			if keyActive[key] == 0 {
				delete(keyActive, key)
			}
		}
	}
	canStart := func(job model.WindowJob) bool {
		if globalActive >= options.GlobalConcurrency {
			return false
		}
		for _, key := range job.Keys {
			if keyActive[key] >= options.PerKeyConcurrency {
				return false
			}
		}
		return true
	}
	acquire := func(p pending, startMS int64) (runningTask, error) {
		if !canStart(p.task) {
			return runningTask{}, model.NewError(model.CodeLedger, "/scheduler", "attempted to acquire an unavailable quota")
		}
		if err := store.StartTask(runID, p.task.WindowStartMS, segmentID, 0, startMS); err != nil {
			return runningTask{}, err
		}
		rt := runningTask{pending: p, startMS: startMS}
		active[p.task.WindowStartMS] = rt
		globalActive++
		for _, key := range p.task.Keys {
			keyActive[key]++
		}
		return rt, nil
	}
	spawn := func(p pending) error {
		rt, err := acquire(p, monoNow())
		if err != nil {
			return err
		}
		go func(pendingTask pending) {
			res := computeWindow(ctx, loaded, pendingTask.task, options)
			computedCh <- res
		}(p)
		_ = rt
		return nil
	}
	for {
		if err := ctx.Err(); err != nil {
			return RunOutcome{}, model.NewError(model.CodeIO, "/context", err.Error())
		}
		if crashAfter >= 0 && completedThisInvocation >= crashAfter && len(queue) > 0 && len(active) == 0 {
			_, err := acquire(queue[0], monoNow())
			if err != nil {
				return RunOutcome{}, err
			}
			return RunOutcome{RunID: runID, Recovered: recovered, Resumed: wasExisting, CrashInjected: true}, ErrSimulatedCrash
		}
		for i := 0; i < len(queue); {
			if crashAfter >= 0 && completedThisInvocation >= crashAfter {
				break
			}
			if !canStart(queue[i].task) {
				i++
				continue
			}
			p := queue[i]
			queue = append(queue[:i], queue[i+1:]...)
			if err := spawn(p); err != nil {
				return RunOutcome{}, err
			}
		}
		if len(active) == 0 && len(queue) == 0 {
			break
		}
		if len(active) == 0 && len(queue) > 0 {
			return RunOutcome{}, model.NewError(model.CodeLedger, "/scheduler", "no task is runnable; quota scheduling deadlocked")
		}
		c := <-computedCh
		if c.err != nil {
			return RunOutcome{}, c.err
		}
		rt, ok := active[c.windowStartMS]
		if !ok {
			return RunOutcome{}, model.NewError(model.CodeLedger, "/scheduler", "completion for non-running window")
		}
		endMS := monoNow()
		if err := store.CompleteTask(runID, segmentID, endMS, c.result, c.decisions, c.resultJSON, c.resultDigest, ledger.NowMS()); err != nil {
			return RunOutcome{}, err
		}
		release(rt)
		completed++
		completedThisInvocation++
	}
	report, err := finalizeAndLoad(store, runID)
	if err != nil {
		store.SetRunError(runID, err, ledger.NowMS())
		return RunOutcome{}, err
	}
	return RunOutcome{Report: report, RunID: runID, Recovered: recovered, Resumed: wasExisting}, nil
}

func computeWindow(ctx context.Context, loaded *model.Loaded, job model.WindowJob, options model.Options) computed {
	if options.TaskDelayMS > 0 {
		t := time.NewTimer(time.Duration(options.TaskDelayMS) * time.Millisecond)
		select {
		case <-ctx.Done():
			t.Stop()
			return computed{windowStartMS: job.WindowStartMS, err: model.NewError(model.CodeIO, "/context", ctx.Err().Error())}
		case <-t.C:
		}
	}
	result, decisions, err := model.BuildWindowResult(loaded, job, options)
	if err != nil {
		return computed{windowStartMS: job.WindowStartMS, err: err}
	}
	resultJSON, err := model.CanonicalJSON(result)
	if err != nil {
		return computed{windowStartMS: job.WindowStartMS, err: err}
	}
	digest, err := model.DigestJSON(result)
	if err != nil {
		return computed{windowStartMS: job.WindowStartMS, err: err}
	}
	return computed{windowStartMS: job.WindowStartMS, result: result, decisions: decisions, resultJSON: string(resultJSON), resultDigest: digest}
}

func reconcileTasks(tasks []ledger.TaskRecord, jobs []model.WindowJob) error {
	if len(tasks) != len(jobs) {
		return model.NewError(model.CodeLedger, "/window_tasks", fmt.Sprintf("task count %d does not match generated window count %d", len(tasks), len(jobs)))
	}
	for i := range tasks {
		if tasks[i].WindowStartMS != jobs[i].WindowStartMS || tasks[i].WindowEndMS != jobs[i].WindowEndMS {
			return model.NewError(model.CodeLedger, "/window_tasks", fmt.Sprintf("task window mismatch at ordinal %d", i+1))
		}
		switch tasks[i].Status {
		case "queued", "unknown", "running", "completed":
		default:
			return model.NewError(model.CodeLedger, "/window_tasks/status", "unsupported task status "+tasks[i].Status)
		}
	}
	return nil
}

func validateMonotonicVersions(latest ledger.RunRecord, loaded *model.Loaded) error {
	var old model.RunManifest
	if err := json.Unmarshal([]byte(latest.ManifestJSON), &old); err != nil {
		return model.NewError(model.CodeLedger, "/runs/manifest_json", err.Error())
	}
	oldLoaded, err := model.LoadedFromManifest(old)
	if err != nil {
		return err
	}
	if loaded.Spec.SpecVersion < old.Spec.SpecVersion {
		return model.NewError(model.CodeSpec, "/spec_version", fmt.Sprintf("spec_version %d must not decrease from previous %d", loaded.Spec.SpecVersion, old.Spec.SpecVersion))
	}
	if loaded.Policy.PolicyVersion < old.Policy.PolicyVersion {
		return model.NewError(model.CodeSpec, "/policy_version", fmt.Sprintf("policy_version %d must not decrease from previous %d", loaded.Policy.PolicyVersion, old.Policy.PolicyVersion))
	}
	if oldLoaded.SpecDigest != loaded.SpecDigest && loaded.Spec.SpecVersion <= old.Spec.SpecVersion {
		return model.NewError(model.CodeSpec, "/spec_version", fmt.Sprintf("traffic declaration changed but spec_version %d is not greater than previous %d", loaded.Spec.SpecVersion, old.Spec.SpecVersion))
	}
	if oldLoaded.PolicyDigest != loaded.PolicyDigest && loaded.Policy.PolicyVersion <= old.Policy.PolicyVersion {
		return model.NewError(model.CodeSpec, "/policy_version", fmt.Sprintf("sampling policy changed but policy_version %d is not greater than previous %d", loaded.Policy.PolicyVersion, old.Policy.PolicyVersion))
	}
	return nil
}

func ptrRun(r ledger.RunRecord, ok bool) *ledger.RunRecord {
	if !ok {
		return nil
	}
	return &r
}

func finalizeAndLoad(store *ledger.Store, runID string) (Report, error) {
	row, ok, err := store.GetRun(runID)
	if err != nil {
		return Report{}, err
	}
	if !ok {
		return Report{}, model.NewError(model.CodeLedger, "/runs", "run not found")
	}
	if row.Status == "completed" && row.ReportJSON.Valid {
		return loadStoredReport(row)
	}
	if row.CompletedWindows != row.TotalWindows {
		return Report{}, model.NewError(model.CodeLedger, "/progress", "run is not complete")
	}
	var manifest model.RunManifest
	if err := json.Unmarshal([]byte(row.ManifestJSON), &manifest); err != nil {
		return Report{}, model.NewError(model.CodeLedger, "/runs/manifest_json", err.Error())
	}
	var options model.Options
	if err := json.Unmarshal([]byte(row.OptionsJSON), &options); err != nil {
		return Report{}, model.NewError(model.CodeLedger, "/runs/options_json", err.Error())
	}
	var cs ChangeSet
	if err := json.Unmarshal([]byte(row.ChangeSetJSON), &cs); err != nil {
		return Report{}, model.NewError(model.CodeLedger, "/runs/change_set_json", err.Error())
	}
	windows, err := store.ListWindowRows(runID)
	if err != nil {
		return Report{}, err
	}
	tasks, err := store.ListTasks(runID)
	if err != nil {
		return Report{}, err
	}
	report := Report{
		FormatVersion:  model.RunFormatVersion,
		RunID:          runID,
		Status:         "completed",
		IsCurrent:      true,
		Stale:          false,
		SpecVersion:    row.SpecVersion,
		PolicyVersion:  row.PolicyVersion,
		ManifestDigest: row.ManifestDigest,
		SemanticDigest: row.SemanticDigest,
		StreamDigest:   row.StreamDigest,
		Options:        options,
		ChangeSet:      cs,
		Progress: Progress{
			CompletedWindows:      row.CompletedWindows,
			TotalWindows:          row.TotalWindows,
			ProgressWindowStartMS: nullInt(row.ProgressWindowStartMS),
		},
		LedgerDigest: row.LedgerContentDigest,
	}
	sampleParts := []string{"flowledger-run-sample-set-v1"}
	summaryParts := []string{"flowledger-window-summary-v1", row.ManifestDigest, row.OptionsDigest}
	for _, w := range windows {
		var result model.WindowResult
		if err := json.Unmarshal([]byte(w.ResultJSON), &result); err != nil {
			return Report{}, model.NewError(model.CodeLedger, "/window_results/result_json", err.Error())
		}
		report.Windows = append(report.Windows, result)
		sampleParts = append(sampleParts, w.SampleSetDigest)
		summaryParts = append(summaryParts, w.ResultDigest)
	}
	report.SampleSetDigest = model.DigestFields(sampleParts...)
	report.SummaryDigest = model.DigestFields(summaryParts...)
	for _, t := range tasks {
		s, err := scheduleFromTask(t)
		if err != nil {
			return Report{}, err
		}
		report.Schedule = append(report.Schedule, s)
	}
	sort.Slice(report.Schedule, func(i, j int) bool { return report.Schedule[i].WindowStartMS < report.Schedule[j].WindowStartMS })
	report.ReportDigest = ""
	digest, err := model.DigestJSON(report)
	if err != nil {
		return Report{}, err
	}
	report.ReportDigest = digest
	reportJSON, err := model.CanonicalJSON(report)
	if err != nil {
		return Report{}, err
	}
	contentDigest, err := store.RecomputeContentDigest(runID)
	if err != nil {
		return Report{}, err
	}
	if err := store.FinalizeRun(runID, string(reportJSON), digest, contentDigest, ledger.NowMS()); err != nil {
		return Report{}, err
	}
	return report, nil
}

func loadStoredReport(row ledger.RunRecord) (Report, error) {
	if !row.ReportJSON.Valid || row.ReportJSON.String == "" {
		return Report{}, model.NewError(model.CodeLedger, "/runs/report_json", "completed run has no persisted report")
	}
	var report Report
	if err := json.Unmarshal([]byte(row.ReportJSON.String), &report); err != nil {
		return Report{}, model.NewError(model.CodeLedger, "/runs/report_json", err.Error())
	}
	if report.Status != "completed" || report.Stale || !report.IsCurrent {
		return Report{}, model.NewError(model.CodeLedger, "/runs/report_json", "persisted report is not a completed current view")
	}
	rec := report
	rec.ReportDigest = ""
	digest, err := model.DigestJSON(rec)
	if err != nil {
		return Report{}, err
	}
	if report.ReportDigest != "" && digest != report.ReportDigest {
		return Report{}, model.NewError(model.CodeLedger, "/runs/report_json/report_digest", "persisted report digest mismatch")
	}
	return report, nil
}

func MarshalReport(r Report, pretty bool) ([]byte, error) {
	if !pretty {
		return model.CanonicalJSON(r)
	}
	b, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return nil, err
	}
	return append(b, '\n'), nil
}
