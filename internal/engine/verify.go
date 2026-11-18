package engine

import (
	"encoding/json"
	"fmt"
	"sort"
	"strconv"

	"flowledger/internal/ledger"
	"flowledger/internal/model"
)

type VerificationReport struct {
	Status              string         `json:"status"`
	RunID               string         `json:"run_id"`
	LedgerFormatVersion int            `json:"ledger_format_version"`
	CurrentRuns         int            `json:"current_runs"`
	StaleRuns           int            `json:"stale_runs"`
	Windows             int            `json:"windows"`
	Events              int64          `json:"events"`
	MaxGlobalObserved   int            `json:"max_global_observed"`
	MaxPerKeyObserved   map[string]int `json:"max_per_key_observed"`
	RecoveredTasks      int            `json:"recovered_unknown_tasks"`
	SampleSetDigest     string         `json:"sample_set_digest"`
	LedgerContentDigest string         `json:"ledger_content_digest"`
	Checks              []string       `json:"checks"`
}

type quotaInterval struct {
	window int64
	start  int64
	end    int64
	keys   []string
}

func Verify(store *ledger.Store) (VerificationReport, error) {
	format, err := store.LedgerFormatVersion()
	if err != nil {
		return VerificationReport{}, err
	}
	if format != 1 {
		return VerificationReport{}, verifyError("/ledger_format_version", fmt.Sprintf("unsupported ledger format %d", format))
	}
	currentRow, ok, err := store.LatestCurrentRun()
	if err != nil {
		return VerificationReport{}, err
	}
	if !ok {
		return VerificationReport{}, verifyError("/runs", "ledger has no current run")
	}
	if currentRow.Status != "completed" || currentRow.CompletedWindows != currentRow.TotalWindows {
		return VerificationReport{}, verifyError("/runs/status", fmt.Sprintf("current run is %s with %d/%d windows", currentRow.Status, currentRow.CompletedWindows, currentRow.TotalWindows))
	}
	row, manifest, report, err := LoadCurrent(store)
	if err != nil {
		return VerificationReport{}, err
	}
	loaded, err := model.LoadedFromManifest(manifest)
	if err != nil {
		return VerificationReport{}, asVerifyError(err, "/manifest")
	}
	manifestDigest, err := model.DigestJSON(manifest)
	if err != nil {
		return VerificationReport{}, err
	}
	if manifestDigest != row.ManifestDigest || loaded.SpecDigest != row.SpecDigest || loaded.PolicyDigest != row.PolicyDigest {
		return VerificationReport{}, verifyError("/runs/manifest_digest", "run manifest or declaration digests do not match the normalized declaration")
	}
	if err := manifest.Options.Validate(); err != nil {
		return VerificationReport{}, asVerifyError(err, "/manifest/options")
	}
	if err := verifyRunsAndStaleness(store, row); err != nil {
		return VerificationReport{}, err
	}
	if err := verifyReport(report, row); err != nil {
		return VerificationReport{}, err
	}
	var storedChangeSet ChangeSet
	if err := json.Unmarshal([]byte(row.ChangeSetJSON), &storedChangeSet); err != nil {
		return VerificationReport{}, verifyError("/runs/change_set_json", err.Error())
	}
	if path, exp, act, diff := firstJSONDifferenceStruct(report.ChangeSet, storedChangeSet, "report/change_set"); diff {
		return VerificationReport{}, verifyError(path, fmt.Sprintf("report change set differs from run row: expected=%v actual=%v", exp, act))
	}
	tasks, err := store.ListTasks(row.RunID)
	if err != nil {
		return VerificationReport{}, err
	}
	if int64(len(tasks)) != row.TotalWindows {
		return VerificationReport{}, verifyError("/window_tasks", fmt.Sprintf("task count %d != total_windows %d", len(tasks), row.TotalWindows))
	}
	segments, err := store.ListSegments(row.RunID)
	if err != nil {
		return VerificationReport{}, err
	}
	segmentSet := make(map[string]bool)
	for _, s := range segments {
		if s.ClockBasis != "process-monotonic-counter-ms" {
			return VerificationReport{}, verifyError("/scheduler_segments/clock_basis", "unexpected clock basis "+s.ClockBasis)
		}
		segmentSet[s.SegmentID] = true
	}
	windows, err := store.ListWindowRows(row.RunID)
	if err != nil {
		return VerificationReport{}, err
	}
	if int64(len(windows)) != row.TotalWindows {
		return VerificationReport{}, verifyError("/window_results", fmt.Sprintf("window result count %d != total_windows %d", len(windows), row.TotalWindows))
	}
	windowByStart := make(map[int64]ledger.WindowRow, len(windows))
	for _, w := range windows {
		windowByStart[w.WindowStartMS] = w
	}
	reportWindows := make(map[int64]model.WindowResult)
	for _, w := range report.Windows {
		reportWindows[w.WindowStartMS] = w
	}
	intervalsBySegment := make(map[string][]quotaInterval)
	var eventTotal int64
	sampleParts := []string{"flowledger-run-sample-set-v1"}
	summaryParts := []string{"flowledger-window-summary-v1", row.ManifestDigest, row.OptionsDigest}
	expectedSchedules := make([]TaskSchedule, 0, len(tasks))
	progressCursor := int64(0)
	recoveredTasks := 0
	for _, task := range tasks {
		if task.Status != "completed" {
			return VerificationReport{}, verifyError(fmt.Sprintf("/window_tasks/%d/status", task.WindowStartMS), "task is not completed: "+task.Status)
		}
		if task.RecoveredFromUnknown {
			recoveredTasks++
		}
		if task.Attempts < 1 || task.GlobalQuota != 1 {
			return VerificationReport{}, verifyError(fmt.Sprintf("/window_tasks/%d", task.WindowStartMS), "attempts must be >=1 and global_quota must be 1")
		}
		if !task.SegmentID.Valid || !segmentSet[task.SegmentID.String] {
			return VerificationReport{}, verifyError(fmt.Sprintf("/window_tasks/%d/segment_id", task.WindowStartMS), "task references a missing scheduler segment")
		}
		if !task.QueueStartMonoMS.Valid || !task.StartMonoMS.Valid || !task.EndMonoMS.Valid ||
			!task.QueueDurationMS.Valid || !task.DurationMS.Valid {
			return VerificationReport{}, verifyError(fmt.Sprintf("/window_tasks/%d/timing", task.WindowStartMS), "task timing fields must be present")
		}
		if task.StartMonoMS.Int64 < task.QueueStartMonoMS.Int64 || task.EndMonoMS.Int64 < task.StartMonoMS.Int64 {
			return VerificationReport{}, verifyError(fmt.Sprintf("/window_tasks/%d/timing", task.WindowStartMS), "monotonic timestamps are not ordered")
		}
		if task.QueueDurationMS.Int64 != task.StartMonoMS.Int64-task.QueueStartMonoMS.Int64 ||
			task.DurationMS.Int64 != task.EndMonoMS.Int64-task.StartMonoMS.Int64 {
			return VerificationReport{}, verifyError(fmt.Sprintf("/window_tasks/%d/timing", task.WindowStartMS), "queue/duration values do not match monotonic timestamps")
		}
		keyQuota, err := decodeKeyQuota(task.KeyQuotaJSON)
		if err != nil {
			return VerificationReport{}, asVerifyError(err, fmt.Sprintf("/window_tasks/%d/key_quota_json", task.WindowStartMS))
		}
		if err := model.ValidateTaskKeys(keyQuota); err != nil {
			return VerificationReport{}, asVerifyError(err, fmt.Sprintf("/window_tasks/%d/key_quota_json", task.WindowStartMS))
		}
		window, ok := windowByStart[task.WindowStartMS]
		if !ok {
			return VerificationReport{}, verifyError(fmt.Sprintf("/window_results/%d", task.WindowStartMS), "completed task has no window result")
		}
		if !task.ResultDigest.Valid || task.ResultDigest.String != window.ResultDigest {
			return VerificationReport{}, verifyError(fmt.Sprintf("/window_tasks/%d/result_digest", task.WindowStartMS), "task result digest does not match window result")
		}
		decisions, err := store.ListDecisions(row.RunID, task.WindowStartMS)
		if err != nil {
			return VerificationReport{}, err
		}
		if err := verifyDecisions(loaded, task, decisions); err != nil {
			return VerificationReport{}, err
		}
		actualKeys := decisionKeys(decisions)
		if !stringSliceEqual(actualKeys, keyQuota) {
			return VerificationReport{}, verifyError(fmt.Sprintf("/window_tasks/%d/key_quota_json", task.WindowStartMS), "quota keys do not equal the distinct client keys in event decisions")
		}
		if err := verifyWindow(store, loaded, manifest, task, window, decisions); err != nil {
			return VerificationReport{}, err
		}
		reportWindow, ok := reportWindows[task.WindowStartMS]
		if !ok {
			return VerificationReport{}, verifyError(fmt.Sprintf("/report/windows/%d", task.WindowStartMS), "report is missing a completed window")
		}
		if path, exp, act, diff := firstJSONDifferenceStruct(reportWindow, mustWindow(window), "report/windows/"+strconv.FormatInt(task.WindowStartMS, 10)); diff {
			return VerificationReport{}, verifyError(path, fmt.Sprintf("report window differs from ledger: expected=%v actual=%v", exp, act))
		}
		eventTotal += window.TotalEventCount
		sampleParts = append(sampleParts, window.SampleSetDigest)
		summaryParts = append(summaryParts, window.ResultDigest)
		if task.WindowStartMS > progressCursor {
			progressCursor = task.WindowStartMS
		}
		sched, err := scheduleFromTask(task)
		if err != nil {
			return VerificationReport{}, err
		}
		expectedSchedules = append(expectedSchedules, sched)
		if task.DurationMS.Int64 > 0 {
			intervalsBySegment[task.SegmentID.String] = append(intervalsBySegment[task.SegmentID.String], quotaInterval{
				window: task.WindowStartMS, start: task.StartMonoMS.Int64, end: task.EndMonoMS.Int64, keys: keyQuota,
			})
		}
	}
	sort.Slice(expectedSchedules, func(i, j int) bool { return expectedSchedules[i].WindowStartMS < expectedSchedules[j].WindowStartMS })
	if path, exp, act, diff := firstJSONDifferenceStruct(report.Schedule, expectedSchedules, "report/schedule"); diff {
		return VerificationReport{}, verifyError(path, fmt.Sprintf("report schedule differs from ledger task records: expected=%v actual=%v", exp, act))
	}
	if report.Progress.CompletedWindows != row.CompletedWindows || report.Progress.TotalWindows != row.TotalWindows ||
		report.Progress.ProgressWindowStartMS != progressCursor || nullInt(row.ProgressWindowStartMS) != progressCursor {
		return VerificationReport{}, verifyError("/progress", "run, report, or ledger progress cursor is inconsistent")
	}
	expectedSampleDigest := model.DigestFields(sampleParts...)
	expectedSummaryDigest := model.DigestFields(summaryParts...)
	if report.SampleSetDigest != expectedSampleDigest {
		return VerificationReport{}, verifyError("/report/sample_set_digest", "run-level sample set digest mismatch")
	}
	if report.SummaryDigest != expectedSummaryDigest {
		return VerificationReport{}, verifyError("/report/summary_digest", "summary digest mismatch")
	}
	contentDigest, err := store.RecomputeContentDigest(row.RunID)
	if err != nil {
		return VerificationReport{}, err
	}
	if contentDigest != row.LedgerContentDigest || report.LedgerDigest != contentDigest {
		return VerificationReport{}, verifyError("/ledger_content_digest", "ledger content digest mismatch")
	}
	maxGlobal, maxPerKey, err := verifyQuotas(intervalsBySegment, manifest.Options)
	if err != nil {
		return VerificationReport{}, err
	}
	checks := []string{
		"ledger format and current-view uniqueness",
		"manifest, report, and ledger content digests",
		"window task state, monotonic timing, and progress cursor",
		"global and per-client-key concurrency budgets",
		"per-event stable-hash decisions and decision/sample digests",
		"window aggregates, rankings, tie breaks, and error bounds",
		"stale-history flags and overall current-view replacement",
	}
	return VerificationReport{
		Status: "ok", RunID: row.RunID, LedgerFormatVersion: format,
		CurrentRuns: 1, StaleRuns: countStale(store, row.RunID),
		Windows: len(windows), Events: eventTotal,
		MaxGlobalObserved: maxGlobal, MaxPerKeyObserved: maxPerKey,
		RecoveredTasks:  recoveredTasks,
		SampleSetDigest: expectedSampleDigest, LedgerContentDigest: contentDigest,
		Checks: checks,
	}, nil
}

func verifyRunsAndStaleness(store *ledger.Store, current ledger.RunRecord) error {
	runs, err := store.ListRuns()
	if err != nil {
		return err
	}
	currentCount := 0
	var prev *ledger.RunRecord
	for i := range runs {
		r := &runs[i]
		if r.IsCurrent && !r.Stale {
			currentCount++
			if r.RunID != current.RunID {
				return verifyError("/runs/current", "more than one current run")
			}
		}
		if r.RunID != current.RunID && (!r.Stale || r.IsCurrent) {
			return verifyError("/runs/stale", "historical run "+r.RunID+" is not marked stale")
		}
		if r.RunID == current.RunID {
			for j := i - 1; j >= 0; j-- {
				prev = &runs[j]
				break
			}
		}
	}
	if currentCount != 1 {
		return verifyError("/runs/current", fmt.Sprintf("expected exactly one current run, found %d", currentCount))
	}
	var currentManifest model.RunManifest
	if err := json.Unmarshal([]byte(current.ManifestJSON), &currentManifest); err != nil {
		return verifyError("/runs/manifest_json", err.Error())
	}
	var cs ChangeSet
	if err := json.Unmarshal([]byte(current.ChangeSetJSON), &cs); err != nil {
		return verifyError("/runs/change_set_json", err.Error())
	}
	if !cs.OverallReplacement {
		return verifyError("/change_set/overall_replacement", "current view must declare overall replacement")
	}
	if prev != nil {
		if cs.PreviousRunID != prev.RunID {
			return verifyError("/change_set/previous_run_id", "change set does not identify the immediately previous view")
		}
		if !prev.Stale {
			return verifyError("/runs/stale", "previous view is not stale")
		}
		var prevManifest model.RunManifest
		if err := json.Unmarshal([]byte(prev.ManifestJSON), &prevManifest); err != nil {
			return verifyError("/runs/manifest_json", err.Error())
		}
		if prevManifest.Spec.SpecVersion > currentManifest.Spec.SpecVersion || prevManifest.Policy.PolicyVersion > currentManifest.Policy.PolicyVersion {
			return verifyError("/runs/versions", "run versions are not monotonic")
		}
		if prev.SpecDigest != current.SpecDigest && currentManifest.Spec.SpecVersion <= prevManifest.Spec.SpecVersion {
			return verifyError("/runs/spec_version", "changed traffic declaration did not increase spec_version")
		}
		if prev.PolicyDigest != current.PolicyDigest && currentManifest.Policy.PolicyVersion <= prevManifest.Policy.PolicyVersion {
			return verifyError("/runs/policy_version", "changed sampling policy did not increase policy_version")
		}
	} else if cs.PreviousRunID != "" {
		return verifyError("/change_set/previous_run_id", "first run unexpectedly references a previous run")
	}
	return nil
}

func verifyReport(report Report, row ledger.RunRecord) error {
	if report.Status != "completed" || !report.IsCurrent || report.Stale {
		return verifyError("/report/status", "report is not a completed current view")
	}
	if report.RunID != row.RunID || report.SpecVersion != row.SpecVersion || report.PolicyVersion != row.PolicyVersion ||
		report.ManifestDigest != row.ManifestDigest || report.SemanticDigest != row.SemanticDigest ||
		report.StreamDigest != row.StreamDigest {
		return verifyError("/report/identity", "report identity does not match run row")
	}
	if report.ReportDigest == "" || !row.ReportDigest.Valid || report.ReportDigest != row.ReportDigest.String {
		return verifyError("/report/report_digest", "report digest is missing or does not match run row")
	}
	rec := report
	rec.ReportDigest = ""
	digest, err := model.DigestJSON(rec)
	if err != nil {
		return err
	}
	if digest != report.ReportDigest {
		return verifyError("/report/report_digest", "report content digest mismatch")
	}
	if int64(len(report.Windows)) != row.TotalWindows {
		return verifyError("/report/windows", "report window count does not match total_windows")
	}
	return nil
}

func verifyDecisions(loaded *model.Loaded, task ledger.TaskRecord, decisions []model.DecisionRecord) error {
	if len(decisions) == 0 {
		return verifyError(fmt.Sprintf("/event_decisions/%d", task.WindowStartMS), "window has no event decisions")
	}
	lastOrder := ""
	keyDecision := make(map[string]string)
	seenEvents := make(map[string]bool)
	for i, d := range decisions {
		if d.WindowStartMS != task.WindowStartMS {
			return verifyError(fmt.Sprintf("/event_decisions/%d/%d/window_start_ms", task.WindowStartMS, i), "event decision belongs to a different window")
		}
		if model.WindowStart(d.TimestampMS, loadedWindowMS(loaded, task)) != task.WindowStartMS {
			return verifyError(fmt.Sprintf("/event_decisions/%d/%d/timestamp_ms", task.WindowStartMS, i), "event timestamp is outside its window")
		}
		st, ok := loaded.StrategyBySource[d.SourceID]
		if !ok {
			return verifyError(fmt.Sprintf("/event_decisions/%d/%d/source_id", task.WindowStartMS, i), "unknown source "+d.SourceID)
		}
		if d.StrategyID != st.StrategyID || d.PolicyVersion != loaded.Policy.PolicyVersion || d.Threshold != st.QuantizedThreshold {
			return verifyError(fmt.Sprintf("/event_decisions/%d/%d/strategy", task.WindowStartMS, i), "strategy, policy version, or threshold mismatch")
		}
		bucket := model.StableBucket(loaded.Policy.PolicyVersion, loaded.Policy.Salt, st.StrategyID, d.ClientKey)
		if d.Bucket != bucket {
			return verifyError(fmt.Sprintf("/event_decisions/%d/%d/bucket", task.WindowStartMS, i), fmt.Sprintf("bucket %d != recomputed %d", d.Bucket, bucket))
		}
		if d.Sampled != (d.Bucket < d.Threshold) {
			return verifyError(fmt.Sprintf("/event_decisions/%d/%d/sampled", task.WindowStartMS, i), "sampled flag does not match bucket < threshold")
		}
		digest := model.DigestFields(
			"flowledger-decision-v1",
			strconv.FormatInt(d.WindowStartMS, 10), d.SourceID,
			strconv.FormatInt(d.EventSeq, 10), d.EventID,
			strconv.FormatInt(d.TimestampMS, 10), d.ClientKey,
			strconv.FormatInt(d.Weight, 10), d.StrategyID,
			strconv.FormatInt(d.PolicyVersion, 10), strconv.Itoa(d.Bucket),
			strconv.Itoa(d.Threshold), strconv.FormatBool(d.Sampled),
		)
		if d.DecisionDigest != digest {
			return verifyError(fmt.Sprintf("/event_decisions/%d/%d/decision_digest", task.WindowStartMS, i), "decision digest mismatch")
		}
		if d.Weight <= 0 {
			return verifyError(fmt.Sprintf("/event_decisions/%d/%d/weight", task.WindowStartMS, i), "weight must be positive")
		}
		expectedID := fmt.Sprintf("%s#%012d", d.SourceID, d.EventSeq)
		if d.EventID != expectedID {
			return verifyError(fmt.Sprintf("/event_decisions/%d/%d/event_id", task.WindowStartMS, i), "event id does not match source and sequence")
		}
		order := fmt.Sprintf("%020d/%s/%020d", d.TimestampMS, d.SourceID, d.EventSeq)
		if order < lastOrder {
			return verifyError(fmt.Sprintf("/event_decisions/%d/order", task.WindowStartMS), "event decisions are not in canonical order")
		}
		lastOrder = order
		eventKey := d.SourceID + "\x00" + strconv.FormatInt(d.EventSeq, 10)
		if seenEvents[eventKey] {
			return verifyError(fmt.Sprintf("/event_decisions/%d/%d/event_seq", task.WindowStartMS, i), "duplicate event")
		}
		seenEvents[eventKey] = true
		keyKey := d.StrategyID + "\x00" + d.ClientKey
		sig := fmt.Sprintf("%d/%t", d.Bucket, d.Sampled)
		if prev, ok := keyDecision[keyKey]; ok && prev != sig {
			return verifyError(fmt.Sprintf("/event_decisions/%d/%d/client_key", task.WindowStartMS, i), "same strategy and client key produced different sampling decisions")
		}
		keyDecision[keyKey] = sig
	}
	return nil
}

func loadedWindowMS(loaded *model.Loaded, task ledger.TaskRecord) int64 {
	if task.WindowEndMS > task.WindowStartMS {
		return task.WindowEndMS - task.WindowStartMS
	}
	return 60000
}

func verifyWindow(store *ledger.Store, loaded *model.Loaded, manifest model.RunManifest, task ledger.TaskRecord, row ledger.WindowRow, decisions []model.DecisionRecord) error {
	var actual model.WindowResult
	if err := json.Unmarshal([]byte(row.ResultJSON), &actual); err != nil {
		return verifyError(fmt.Sprintf("/window_results/%d/result_json", task.WindowStartMS), err.Error())
	}
	digest, err := model.DigestJSON(actual)
	if err != nil {
		return err
	}
	if digest != row.ResultDigest {
		return verifyError(fmt.Sprintf("/window_results/%d/result_digest", task.WindowStartMS), "result JSON digest mismatch")
	}
	if row.WindowEndMS != task.WindowEndMS || actual.WindowStartMS != task.WindowStartMS || actual.WindowEndMS != task.WindowEndMS {
		return verifyError(fmt.Sprintf("/window_results/%d/window", task.WindowStartMS), "window bounds disagree with task record")
	}
	job := model.WindowJob{WindowStartMS: task.WindowStartMS, WindowEndMS: task.WindowEndMS}
	keySet := make(map[string]bool)
	for _, d := range decisions {
		job.Events = append(job.Events, model.Event{
			SourceID: d.SourceID, Seq: d.EventSeq, EventID: d.EventID,
			TimestampMS: d.TimestampMS, ClientKey: d.ClientKey, Weight: d.Weight,
		})
		keySet[d.ClientKey] = true
	}
	for key := range keySet {
		job.Keys = append(job.Keys, key)
	}
	sort.Strings(job.Keys)
	expected, _, err := model.BuildWindowResult(loaded, job, manifest.Options)
	if err != nil {
		return asVerifyError(err, fmt.Sprintf("/window_results/%d/recompute", task.WindowStartMS))
	}
	if path, exp, act, diff := firstJSONDifferenceStruct(expected, actual, fmt.Sprintf("window_results/%d/result", task.WindowStartMS)); diff {
		return verifyError(path, fmt.Sprintf("stored window result differs from ledger-derived recomputation: expected=%v actual=%v", exp, act))
	}
	if row.TotalEventCount != expected.TotalEventCount || row.SampledEventCount != expected.SampledEventCount ||
		row.UnseenEventCount != expected.UnseenEventCount || row.SampleSetDigest != expected.SampleSetDigest ||
		row.DecisionSetDigest != expected.DecisionSetDigest || row.AggregationDigest != expected.AggregationDigest ||
		row.RankingDigest != expected.RankingDigest || row.EstimateApplicable != expected.EstimateApplicable ||
		row.TopKGuaranteed != expected.TopKGuaranteed {
		return verifyError(fmt.Sprintf("/window_results/%d", task.WindowStartMS), "denormalized window summary columns disagree with result JSON")
	}
	if err := verifyResultErrorBounds(manifest.Options, expected); err != nil {
		return err
	}
	aggRows, err := store.ListKeyAggregates(row.RunID, task.WindowStartMS)
	if err != nil {
		return err
	}
	if len(aggRows) != len(expected.Aggregates) {
		return verifyError(fmt.Sprintf("window_results/%d/key_aggregates", task.WindowStartMS), "key aggregate row count disagrees with result JSON")
	}
	for i, ar := range aggRows {
		a := expected.Aggregates[i]
		var sourceIDs []string
		if err := json.Unmarshal([]byte(ar.SourceIDsJSON), &sourceIDs); err != nil {
			return verifyError(fmt.Sprintf("window_results/%d/key_aggregates/%d/source_ids_json", task.WindowStartMS, i), err.Error())
		}
		if ar.ClientKey != a.ClientKey || !stringSliceEqual(sourceIDs, a.SourceIDs) ||
			ar.SampleCount != a.SampleCount || ar.SampleWeight != a.SampleWeight ||
			ar.EstimatedCountMilli != a.EstimatedCountMilli || ar.EstimatorLowerMilli != a.EstimatorLowerMilli ||
			ar.EstimatorUpperMilli != a.EstimatorUpperMilli || ar.ObservedCountLower != a.ObservedCountLower ||
			ar.ObservedCountUpper != a.ObservedCountUpper || ar.Rank != int64(a.Rank) {
			return verifyError(fmt.Sprintf("window_results/%d/key_aggregates/%d", task.WindowStartMS, i), "key aggregate row disagrees with result JSON")
		}
	}
	return nil
}

func verifyResultErrorBounds(options model.Options, r model.WindowResult) error {
	if !r.EstimateApplicable {
		if r.SampledEventCount >= options.MinSampleEvents {
			return verifyError("/result/estimate_applicable", "estimate is marked inapplicable despite meeting the sample threshold")
		}
		if len(r.Ranking) != 0 {
			return verifyError("/result/ranking", "inapplicable estimates must not publish a full-ranking conclusion")
		}
		if r.InapplicabilityReason == "" {
			return verifyError("/result/inapplicability_reason", "missing inapplicability reason")
		}
	} else if len(r.Ranking) > options.K {
		return verifyError("/result/ranking", "ranking exceeds K")
	}
	seen := make(map[string]bool)
	for i, a := range r.Aggregates {
		if seen[a.ClientKey] {
			return verifyError(fmt.Sprintf("/result/aggregates/%d/client_key", i), "duplicate client key")
		}
		seen[a.ClientKey] = true
		if a.SampleCount <= 0 || a.SampleWeight <= 0 {
			return verifyError(fmt.Sprintf("/result/aggregates/%d", i), "sample count and weight must be positive")
		}
		if a.ObservedCountLower != a.SampleCount || a.ObservedCountUpper != a.SampleCount+r.UnseenEventCount {
			return verifyError(fmt.Sprintf("/result/aggregates/%d/observed_bounds", i), "observed count bounds are inconsistent with cluster sampling")
		}
		if !(a.EstimatorLowerMilli <= a.EstimatedCountMilli && a.EstimatedCountMilli <= a.EstimatorUpperMilli) {
			return verifyError(fmt.Sprintf("/result/aggregates/%d/estimator", i), "estimate lies outside its quantization envelope")
		}
		quantErr := a.EstimatedCountMilli - a.EstimatorLowerMilli
		if a.EstimatorUpperMilli-a.EstimatedCountMilli > quantErr {
			quantErr = a.EstimatorUpperMilli - a.EstimatedCountMilli
		}
		if a.QuantizationErrorBoundMilli != quantErr || a.EstimatedCount != model.FormatMilli(a.EstimatedCountMilli) {
			return verifyError(fmt.Sprintf("/result/aggregates/%d/error_bound", i), "quantization error bound is inconsistent")
		}
	}
	for i := 1; i < len(r.Aggregates); i++ {
		if r.Aggregates[i-1].ClientKey >= r.Aggregates[i].ClientKey {
			return verifyError("/result/aggregates/order", "aggregates are not stably sorted by client key")
		}
	}
	aggByKey := make(map[string]model.KeyAggregate)
	for _, a := range r.Aggregates {
		aggByKey[a.ClientKey] = a
	}
	for i, it := range r.Ranking {
		if it.Rank != i+1 {
			return verifyError(fmt.Sprintf("/result/ranking/%d/rank", i), "rank is not sequential")
		}
		a, ok := aggByKey[it.ClientKey]
		if !ok {
			return verifyError(fmt.Sprintf("/result/ranking/%d/client_key", i), "ranking key has no aggregate")
		}
		if it.ScoreMilli != a.EstimatedCountMilli || it.Score != a.EstimatedCount ||
			it.SampleCount != a.SampleCount || it.SampleWeight != a.SampleWeight ||
			it.EstimatedCountMilli != a.EstimatedCountMilli || it.EstimatorLowerMilli != a.EstimatorLowerMilli ||
			it.EstimatorUpperMilli != a.EstimatorUpperMilli || it.QuantizationErrorBoundMilli != a.QuantizationErrorBoundMilli ||
			it.ObservedCountLower != a.ObservedCountLower || it.ObservedCountUpper != a.ObservedCountUpper {
			return verifyError(fmt.Sprintf("/result/ranking/%d", i), "ranking item disagrees with its aggregate")
		}
		if i > 0 && !rankLess(r.Ranking[i-1], it) {
			return verifyError("/result/ranking/order", "ranking is not in the declared stable order")
		}
	}
	guaranteed, _ := model.EvaluateTopKGuarantee(r.EstimateApplicable, r.InapplicabilityReason, r.Aggregates, r.Ranking, r.UnseenEventCount, options.K)
	if guaranteed != r.TopKGuaranteed {
		return verifyError("/result/top_k_guaranteed", "top-K guarantee flag disagrees with its explicit basis")
	}
	expectedStrategies := len(r.StrategyErrorBounds)
	if expectedStrategies == 0 {
		return verifyError("/result/strategy_error_bounds", "missing strategy error bounds")
	}
	for i, b := range r.StrategyErrorBounds {
		th := model.QuantizedThreshold(b.RatioBP)
		trunc := int64(b.RatioBP)*model.BucketCount - int64(th)*10000
		if b.QuantizedThreshold != th || b.BucketCount != model.BucketCount || b.Quantization != model.QuantizationMode ||
			b.RatioTruncationErrorNumerator != trunc || b.RatioErrorDenominator != int64(model.BucketCount)*10000 ||
			b.HashBucketGranularityNumerator != 1 || b.HashBucketGranularityDenominator != model.BucketCount ||
			b.CombinedErrorNumerator != trunc+10000 || b.CombinedErrorDenominator != int64(model.BucketCount)*10000 ||
			b.EnvelopeDenominator != model.BucketCount || b.Basis == "" {
			return verifyError(fmt.Sprintf("/result/strategy_error_bounds/%d", i), "strategy error bound formula is inconsistent")
		}
	}
	return nil
}

func rankLess(a, b model.RankItem) bool {
	if a.EstimatedCountMilli != b.EstimatedCountMilli {
		return a.EstimatedCountMilli > b.EstimatedCountMilli
	}
	if a.SampleWeight != b.SampleWeight {
		return a.SampleWeight > b.SampleWeight
	}
	if a.SampleCount != b.SampleCount {
		return a.SampleCount > b.SampleCount
	}
	return a.ClientKey < b.ClientKey
}

func verifyQuotas(bySegment map[string][]quotaInterval, options model.Options) (int, map[string]int, error) {
	type event struct {
		time   int64
		start  bool
		keys   []string
		window int64
	}
	maxGlobal := 0
	maxPerKey := make(map[string]int)
	segments := make([]string, 0, len(bySegment))
	for s := range bySegment {
		segments = append(segments, s)
	}
	sort.Strings(segments)
	for _, segment := range segments {
		intervals := bySegment[segment]
		events := make([]event, 0, len(intervals)*2)
		for _, iv := range intervals {
			if iv.end <= iv.start {
				continue
			}
			events = append(events, event{time: iv.start, start: true, keys: iv.keys, window: iv.window})
			events = append(events, event{time: iv.end, start: false, keys: iv.keys, window: iv.window})
		}
		sort.Slice(events, func(i, j int) bool {
			if events[i].time != events[j].time {
				return events[i].time < events[j].time
			}
			if events[i].start != events[j].start {
				return !events[i].start
			}
			return events[i].window < events[j].window
		})
		activeGlobal := 0
		activeKey := make(map[string]int)
		for _, ev := range events {
			if ev.start {
				activeGlobal++
				if activeGlobal > maxGlobal {
					maxGlobal = activeGlobal
				}
				if activeGlobal > options.GlobalConcurrency {
					return 0, nil, verifyError("/schedule/global_concurrency", fmt.Sprintf("running tasks %d exceed global budget %d", activeGlobal, options.GlobalConcurrency))
				}
				for _, key := range ev.keys {
					activeKey[key]++
					if activeKey[key] > maxPerKey[key] {
						maxPerKey[key] = activeKey[key]
					}
					if activeKey[key] > options.PerKeyConcurrency {
						return 0, nil, verifyError("/schedule/per_key_concurrency", fmt.Sprintf("key %q has %d running tasks, budget %d", key, activeKey[key], options.PerKeyConcurrency))
					}
				}
			} else {
				activeGlobal--
				if activeGlobal < 0 {
					return 0, nil, verifyError("/schedule/global_concurrency", "overlapping task intervals are unbalanced")
				}
				for _, key := range ev.keys {
					activeKey[key]--
					if activeKey[key] == 0 {
						delete(activeKey, key)
					}
				}
			}
		}
		if activeGlobal != 0 || len(activeKey) != 0 {
			return 0, nil, verifyError("/schedule/global_concurrency", "task intervals do not balance within a scheduler segment")
		}
	}
	return maxGlobal, maxPerKey, nil
}

func decisionKeys(decisions []model.DecisionRecord) []string {
	set := make(map[string]bool)
	for _, d := range decisions {
		set[d.ClientKey] = true
	}
	out := make([]string, 0, len(set))
	for key := range set {
		out = append(out, key)
	}
	sort.Strings(out)
	return out
}

func decodeKeyQuota(raw string) ([]string, error) {
	var keys []string
	if err := json.Unmarshal([]byte(raw), &keys); err != nil {
		return nil, err
	}
	sort.Strings(keys)
	return keys, nil
}

func mustWindow(row ledger.WindowRow) model.WindowResult {
	var w model.WindowResult
	_ = json.Unmarshal([]byte(row.ResultJSON), &w)
	return w
}

func countStale(store *ledger.Store, currentID string) int {
	runs, err := store.ListRuns()
	if err != nil {
		return 0
	}
	n := 0
	for _, r := range runs {
		if r.RunID != currentID && r.Stale {
			n++
		}
	}
	return n
}

func asVerifyError(err error, path string) error {
	if ce, ok := err.(*model.CodedError); ok {
		return &model.CodedError{Code: model.CodeVerify, Path: path, Message: ce.Message, Details: ce.Details}
	}
	return verifyError(path, err.Error())
}

func verifyError(path, message string) error {
	return &model.CodedError{Code: model.CodeVerify, Path: path, Message: message}
}
