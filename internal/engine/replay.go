package engine

import (
	"bytes"
	"encoding/json"
	"fmt"
	"sort"
	"strconv"
	"strings"

	"flowledger/internal/ledger"
	"flowledger/internal/model"
)

type ReplayReport struct {
	Status          string `json:"status"`
	RunID           string `json:"run_id"`
	CheckedWindows  int    `json:"checked_windows"`
	CheckedEvents   int    `json:"checked_events"`
	StreamDigest    string `json:"event_stream_digest"`
	SampleSetDigest string `json:"sample_set_digest"`
	LedgerDigest    string `json:"ledger_content_digest"`
}

func Replay(store *ledger.Store) (ReplayReport, error) {
	currentRow, ok, err := store.LatestCurrentRun()
	if err != nil {
		return ReplayReport{}, err
	}
	if !ok {
		return ReplayReport{}, replayMismatch("*", 0, "runs", "current", "missing")
	}
	if currentRow.Status != "completed" || currentRow.CompletedWindows != currentRow.TotalWindows {
		return ReplayReport{}, replayMismatch("*", 0, "runs/status", "completed", currentRow.Status)
	}
	row, manifest, _, err := LoadCurrent(store)
	if err != nil {
		return ReplayReport{}, err
	}
	loaded, err := model.LoadedFromManifest(manifest)
	if err != nil {
		return ReplayReport{}, &model.CodedError{Code: model.CodeReplay, Path: "/manifest", Message: err.Error()}
	}
	stream, err := model.Generate(loaded)
	if err != nil {
		return ReplayReport{}, &model.CodedError{Code: model.CodeReplay, Path: "/event_stream", Message: err.Error()}
	}
	if stream.StreamDigest != manifest.StreamDigest || manifest.StreamDigest != row.StreamDigest {
		source := firstSource(manifest)
		return ReplayReport{}, replayMismatch(source, 0, "event_stream_digest", manifest.StreamDigest, row.StreamDigest)
	}
	manifestSources := make(map[string]string)
	for _, sd := range manifest.SourceDigests {
		manifestSources[sd.SourceID] = sd.Digest
	}
	for _, sid := range sortedKeys(stream.SourceDigest) {
		if manifestSources[sid] != stream.SourceDigest[sid] {
			return ReplayReport{}, replayMismatch(sid, 0, "source_digest", stream.SourceDigest[sid], manifestSources[sid])
		}
	}
	jobs := model.BuildWindowJobs(stream, manifest.Options.WindowMS())
	tasks, err := store.ListTasks(row.RunID)
	if err != nil {
		return ReplayReport{}, err
	}
	taskByWindow := make(map[int64]ledger.TaskRecord)
	for _, t := range tasks {
		taskByWindow[t.WindowStartMS] = t
	}
	replay := ReplayReport{
		Status: "ok", RunID: row.RunID, StreamDigest: stream.StreamDigest,
		LedgerDigest: row.LedgerContentDigest,
	}
	sampleParts := []string{"flowledger-run-sample-set-v1"}
	for _, job := range jobs {
		task, ok := taskByWindow[job.WindowStartMS]
		if !ok {
			return ReplayReport{}, replayMismatch(firstJobSource(job), job.WindowStartMS, "window_tasks/status", "present", "missing")
		}
		if task.Status != "completed" {
			return ReplayReport{}, replayMismatch(firstJobSource(job), job.WindowStartMS, "window_tasks/status", "completed", task.Status)
		}
		expected, expectedDecisions, err := model.BuildWindowResult(loaded, job, manifest.Options)
		if err != nil {
			return ReplayReport{}, &model.CodedError{Code: model.CodeReplay, Path: "/recompute", Message: err.Error()}
		}
		actualDecisions, err := store.ListDecisions(row.RunID, job.WindowStartMS)
		if err != nil {
			return ReplayReport{}, err
		}
		if err := compareDecisions(job, expectedDecisions, actualDecisions); err != nil {
			return ReplayReport{}, err
		}
		actualRow, ok, err := store.GetWindowResult(row.RunID, job.WindowStartMS)
		if err != nil {
			return ReplayReport{}, err
		}
		if !ok {
			return ReplayReport{}, replayMismatch(firstJobSource(job), job.WindowStartMS, "window_results", "present", "missing")
		}
		if field, exp, act, diff := compareWindowRow(expected, actualRow); diff {
			return ReplayReport{}, replayMismatch(firstJobSource(job), job.WindowStartMS, field, exp, act)
		}
		var actualResult model.WindowResult
		if err := json.Unmarshal([]byte(actualRow.ResultJSON), &actualResult); err != nil {
			return ReplayReport{}, replayMismatch(firstJobSource(job), job.WindowStartMS, "result_json", "valid JSON", err.Error())
		}
		if path, exp, act, diff := firstJSONDifferenceStruct(expected, actualResult, "result"); diff {
			return ReplayReport{}, replayMismatch(sourceForResultPath(job, expected, path), job.WindowStartMS, path, exp, act)
		}
		if task.ResultDigest.String != actualRow.ResultDigest {
			return ReplayReport{}, replayMismatch(firstJobSource(job), job.WindowStartMS, "window_tasks/result_digest", actualRow.ResultDigest, task.ResultDigest.String)
		}
		replay.CheckedWindows++
		replay.CheckedEvents += int(expected.TotalEventCount)
		sampleParts = append(sampleParts, expected.SampleSetDigest)
	}
	replay.SampleSetDigest = model.DigestFields(sampleParts...)
	return replay, nil
}

func compareDecisions(job model.WindowJob, expected, actual []model.DecisionRecord) error {
	n := len(expected)
	if len(actual) < n {
		n = len(actual)
	}
	for i := 0; i < n; i++ {
		if field, exp, act, diff := firstDecisionDifference(expected[i], actual[i]); diff {
			source := expected[i].SourceID
			if source == "" {
				source = actual[i].SourceID
			}
			return replayMismatch(source, job.WindowStartMS, fmt.Sprintf("event_decisions[%d]/%s", i, field), exp, act)
		}
	}
	if len(expected) > len(actual) {
		d := expected[len(actual)]
		return replayMismatch(d.SourceID, job.WindowStartMS, fmt.Sprintf("event_decisions[%d]", len(actual)), "present", "missing")
	}
	if len(actual) > len(expected) {
		d := actual[len(expected)]
		return replayMismatch(d.SourceID, job.WindowStartMS, fmt.Sprintf("event_decisions[%d]", len(expected)), "missing", "present")
	}
	return nil
}

func firstDecisionDifference(a, b model.DecisionRecord) (string, any, any, bool) {
	switch {
	case a.WindowStartMS != b.WindowStartMS:
		return "window_start_ms", a.WindowStartMS, b.WindowStartMS, true
	case a.SourceID != b.SourceID:
		return "source_id", a.SourceID, b.SourceID, true
	case a.EventSeq != b.EventSeq:
		return "event_seq", a.EventSeq, b.EventSeq, true
	case a.EventID != b.EventID:
		return "event_id", a.EventID, b.EventID, true
	case a.TimestampMS != b.TimestampMS:
		return "timestamp_ms", a.TimestampMS, b.TimestampMS, true
	case a.ClientKey != b.ClientKey:
		return "client_key", a.ClientKey, b.ClientKey, true
	case a.Weight != b.Weight:
		return "weight", a.Weight, b.Weight, true
	case a.StrategyID != b.StrategyID:
		return "strategy_id", a.StrategyID, b.StrategyID, true
	case a.PolicyVersion != b.PolicyVersion:
		return "policy_version", a.PolicyVersion, b.PolicyVersion, true
	case a.Bucket != b.Bucket:
		return "bucket", a.Bucket, b.Bucket, true
	case a.Threshold != b.Threshold:
		return "threshold", a.Threshold, b.Threshold, true
	case a.Sampled != b.Sampled:
		return "sampled", a.Sampled, b.Sampled, true
	case a.DecisionDigest != b.DecisionDigest:
		return "decision_digest", a.DecisionDigest, b.DecisionDigest, true
	default:
		return "", nil, nil, false
	}
}

func compareWindowRow(expected model.WindowResult, actual ledger.WindowRow) (string, any, any, bool) {
	switch {
	case expected.WindowStartMS != actual.WindowStartMS:
		return "window_start_ms", expected.WindowStartMS, actual.WindowStartMS, true
	case expected.WindowEndMS != actual.WindowEndMS:
		return "window_end_ms", expected.WindowEndMS, actual.WindowEndMS, true
	case expected.TotalEventCount != actual.TotalEventCount:
		return "total_event_count", expected.TotalEventCount, actual.TotalEventCount, true
	case expected.SampledEventCount != actual.SampledEventCount:
		return "sampled_event_count", expected.SampledEventCount, actual.SampledEventCount, true
	case expected.UnseenEventCount != actual.UnseenEventCount:
		return "unseen_event_count", expected.UnseenEventCount, actual.UnseenEventCount, true
	case expected.SampleSetDigest != actual.SampleSetDigest:
		return "sample_set_digest", expected.SampleSetDigest, actual.SampleSetDigest, true
	case expected.DecisionSetDigest != actual.DecisionSetDigest:
		return "decision_set_digest", expected.DecisionSetDigest, actual.DecisionSetDigest, true
	case expected.AggregationDigest != actual.AggregationDigest:
		return "aggregation_digest", expected.AggregationDigest, actual.AggregationDigest, true
	case expected.RankingDigest != actual.RankingDigest:
		return "ranking_digest", expected.RankingDigest, actual.RankingDigest, true
	case expected.EstimateApplicable != actual.EstimateApplicable:
		return "estimate_applicable", expected.EstimateApplicable, actual.EstimateApplicable, true
	case expected.TopKGuaranteed != actual.TopKGuaranteed:
		return "top_k_guaranteed", expected.TopKGuaranteed, actual.TopKGuaranteed, true
	}
	digest, _ := model.DigestJSON(expected)
	if digest != actual.ResultDigest {
		return "result_digest", digest, actual.ResultDigest, true
	}
	return "", nil, nil, false
}

func replayMismatch(source string, windowStart int64, field string, expected, actual any) error {
	return &model.CodedError{
		Code:    model.CodeReplay,
		Message: "replay mismatch",
		Details: map[string]any{
			"source_id":       source,
			"window_start_ms": windowStart,
			"field":           field,
			"expected":        expected,
			"actual":          actual,
		},
	}
}

func sourceForResultPath(job model.WindowJob, result model.WindowResult, path string) string {
	parts := strings.Split(path, "/")
	key := ""
	if len(parts) >= 3 && (parts[1] == "aggregates" || parts[1] == "ranking") {
		idx, err := strconv.Atoi(parts[2])
		if err == nil {
			if parts[1] == "aggregates" && idx >= 0 && idx < len(result.Aggregates) {
				key = result.Aggregates[idx].ClientKey
			}
			if parts[1] == "ranking" && idx >= 0 && idx < len(result.Ranking) {
				key = result.Ranking[idx].ClientKey
			}
		}
	}
	if key != "" {
		source := ""
		for _, e := range job.Events {
			if e.ClientKey == key && (source == "" || e.SourceID < source) {
				source = e.SourceID
			}
		}
		if source != "" {
			return source
		}
	}
	if len(parts) >= 3 && parts[1] == "strategy_error_bounds" {
		idx, err := strconv.Atoi(parts[2])
		if err == nil && idx >= 0 && idx < len(result.StrategyErrorBounds) && len(result.StrategyErrorBounds[idx].SourceIDs) > 0 {
			return result.StrategyErrorBounds[idx].SourceIDs[0]
		}
	}
	return firstJobSource(job)
}

func firstSource(m model.RunManifest) string {
	if len(m.SourceDigests) == 0 {
		return "*"
	}
	ids := make([]string, 0, len(m.SourceDigests))
	for _, s := range m.SourceDigests {
		ids = append(ids, s.SourceID)
	}
	sort.Strings(ids)
	return ids[0]
}

func firstJobSource(job model.WindowJob) string {
	source := ""
	for _, e := range job.Events {
		if source == "" || e.SourceID < source {
			source = e.SourceID
		}
	}
	if source == "" {
		return "*"
	}
	return source
}

func sortedKeys[V any](m map[string]V) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func firstJSONDifference(expected, actual any, path string) (string, any, any, bool) {
	switch e := expected.(type) {
	case map[string]any:
		a, ok := actual.(map[string]any)
		if !ok {
			return path, expected, actual, true
		}
		keys := make([]string, 0, len(e))
		for k := range e {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			av, ok := a[k]
			if !ok {
				return path + "/" + k, e[k], "<missing>", true
			}
			if p, ev, av2, diff := firstJSONDifference(e[k], av, path+"/"+k); diff {
				return p, ev, av2, true
			}
		}
		for k := range a {
			if _, ok := e[k]; !ok {
				return path + "/" + k, "<missing>", a[k], true
			}
		}
		return "", nil, nil, false
	case []any:
		a, ok := actual.([]any)
		if !ok {
			return path, expected, actual, true
		}
		n := len(e)
		if len(a) < n {
			n = len(a)
		}
		for i := 0; i < n; i++ {
			if p, ev, av, diff := firstJSONDifference(e[i], a[i], fmt.Sprintf("%s/%d", path, i)); diff {
				return p, ev, av, true
			}
		}
		if len(e) != len(a) {
			return path + "/length", len(e), len(a), true
		}
		return "", nil, nil, false
	default:
		eb, _ := json.Marshal(expected)
		ab, _ := json.Marshal(actual)
		if !bytes.Equal(eb, ab) {
			return path, normalizeScalar(expected), normalizeScalar(actual), true
		}
		return "", nil, nil, false
	}
}

func normalizeScalar(v any) any {
	if n, ok := v.(json.Number); ok {
		return n.String()
	}
	return v
}

func normalizeJSON(v any) any {
	b, err := json.Marshal(v)
	if err != nil {
		return v
	}
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.UseNumber()
	var out any
	if err := dec.Decode(&out); err != nil {
		return v
	}
	return out
}

func firstJSONDifferenceStruct(expected, actual any, path string) (string, any, any, bool) {
	return firstJSONDifference(normalizeJSON(expected), normalizeJSON(actual), path)
}
