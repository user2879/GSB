package engine

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	_ "modernc.org/sqlite"

	"flowledger/internal/ledger"
	"flowledger/internal/model"
)

func loadSamples(t *testing.T, specName, policyName string) *model.Loaded {
	t.Helper()
	spec, err := os.ReadFile(filepath.Join("..", "..", "samples", specName))
	if err != nil {
		t.Fatal(err)
	}
	policy, err := os.ReadFile(filepath.Join("..", "..", "samples", policyName))
	if err != nil {
		t.Fatal(err)
	}
	loaded, err := model.LoadBytes(spec, policy)
	if err != nil {
		t.Fatal(err)
	}
	return loaded
}

func TestCrashResumeReplayVerifyAndEvolution(t *testing.T) {
	ctx := context.Background()
	dbPath := filepath.Join(t.TempDir(), "ledger.db")
	store, err := ledger.Open(dbPath)
	if err != nil {
		t.Fatal(err)
	}
	loaded1 := loadSamples(t, "traffic-v1.json", "policy-v1.json")
	options := model.Options{
		GlobalConcurrency: 2, PerKeyConcurrency: 2, WindowMinutes: 1,
		K: 3, MinSampleEvents: 2, TaskDelayMS: 8,
	}
	out1, err := Run(ctx, store, loaded1, options, -1)
	if err != nil {
		t.Fatal(err)
	}
	if out1.Report.Status != "completed" || out1.Resumed {
		t.Fatalf("unexpected first outcome: status=%s resumed=%t", out1.Report.Status, out1.Resumed)
	}
	if _, err := Verify(store); err != nil {
		t.Fatalf("verify v1: %v", err)
	}
	if _, err := Replay(store); err != nil {
		t.Fatalf("replay v1: %v", err)
	}

	loaded2 := loadSamples(t, "traffic-v2.json", "policy-v2.json")
	outCrash, err := Run(ctx, store, loaded2, options, 0)
	if !errors.Is(err, ErrSimulatedCrash) {
		t.Fatalf("expected simulated crash, got %v", err)
	}
	if outCrash.Report.Status != "" {
		t.Fatal("crash path must not produce a report")
	}
	crashed, ok, err := store.GetRun(outCrash.RunID)
	if err != nil || !ok {
		t.Fatalf("get crashed run: %v", err)
	}
	if crashed.Status != "running" || crashed.ReportJSON.Valid {
		t.Fatalf("crashed run status=%s report_valid=%t", crashed.Status, crashed.ReportJSON.Valid)
	}
	if crashed.CompletedWindows != 0 {
		t.Fatalf("crash-after-0 completed %d windows", crashed.CompletedWindows)
	}
	tasks, err := store.ListTasks(outCrash.RunID)
	if err != nil {
		t.Fatal(err)
	}
	running := 0
	for _, task := range tasks {
		if task.Status == "running" {
			running++
		}
	}
	if running != 1 {
		t.Fatalf("expected exactly one durably running task, got %d", running)
	}
	if _, err := Verify(store); err == nil {
		t.Fatal("verify accepted an incomplete current run")
	} else {
		var ce *model.CodedError
		if !errors.As(err, &ce) || ce.Code != model.CodeVerify {
			t.Fatalf("expected E_VERIFY for incomplete run, got %v", err)
		}
	}
	if _, err := Replay(store); err == nil {
		t.Fatal("replay accepted an incomplete current run")
	} else {
		var ce *model.CodedError
		if !errors.As(err, &ce) || ce.Code != model.CodeReplay {
			t.Fatalf("expected E_REPLAY for incomplete run, got %v", err)
		}
	}

	out2, err := Run(ctx, store, loaded2, options, -1)
	if err != nil {
		t.Fatal(err)
	}
	if !out2.Resumed || out2.Recovered < 1 {
		t.Fatalf("expected resume with recovered unknown task, got resumed=%t recovered=%d", out2.Resumed, out2.Recovered)
	}
	verify, err := Verify(store)
	if err != nil {
		t.Fatalf("verify v2: %v", err)
	}
	if verify.StaleRuns != 1 {
		t.Fatalf("expected one stale run, got %d", verify.StaleRuns)
	}
	if verify.MaxGlobalObserved < 2 {
		t.Fatalf("expected observable parallelism, max global=%d", verify.MaxGlobalObserved)
	}
	if verify.MaxGlobalObserved > options.GlobalConcurrency {
		t.Fatalf("global budget exceeded: %d", verify.MaxGlobalObserved)
	}
	for key, n := range verify.MaxPerKeyObserved {
		if n > options.PerKeyConcurrency {
			t.Fatalf("per-key budget exceeded for %s: %d", key, n)
		}
	}
	replay, err := Replay(store)
	if err != nil {
		t.Fatalf("replay v2: %v", err)
	}
	if replay.CheckedWindows != len(out2.Report.Windows) {
		t.Fatalf("replayed %d of %d windows", replay.CheckedWindows, len(out2.Report.Windows))
	}
	if !out2.Report.ChangeSet.OverallReplacement || out2.Report.ChangeSet.PreviousRunID != out1.RunID {
		t.Fatalf("bad change set: %+v", out2.Report.ChangeSet)
	}
	if len(out2.Report.ChangeSet.AddedSources) != 1 || out2.Report.ChangeSet.AddedSources[0] != "src-north" {
		t.Fatalf("expected src-north addition, got %+v", out2.Report.ChangeSet)
	}
	if len(out2.Report.ChangeSet.RatioChanges) < 3 || len(out2.Report.ChangeSet.AffectedWindows) == 0 {
		t.Fatalf("expected ratio and window changes, got %+v", out2.Report.ChangeSet)
	}
	old, _, err := store.GetRun(out1.RunID)
	if err != nil {
		t.Fatal(err)
	}
	if !old.Stale || old.IsCurrent {
		t.Fatal("v1 run was not marked stale")
	}
	reused, err := Run(ctx, store, loaded2, options, -1)
	if err != nil {
		t.Fatal(err)
	}
	if !reused.ReusedReport || reused.Report.ReportDigest != out2.Report.ReportDigest {
		t.Fatal("completed run was not returned as the same persisted report")
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}

	db, err := sql.Open("sqlite", dbPath)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	var source string
	var window, seq int64
	if err := db.QueryRow(`SELECT source_id,window_start_ms,event_seq FROM event_decisions WHERE run_id=? ORDER BY window_start_ms,timestamp_ms,source_id,event_seq LIMIT 1`, out2.RunID).Scan(&source, &window, &seq); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(`UPDATE event_decisions SET weight=weight+1 WHERE run_id=? AND window_start_ms=? AND source_id=? AND event_seq=?`,
		out2.RunID, window, source, seq); err != nil {
		t.Fatal(err)
	}
	tamperedStore, err := ledger.Open(dbPath)
	if err != nil {
		t.Fatal(err)
	}
	defer tamperedStore.Close()
	_, err = Replay(tamperedStore)
	if err == nil {
		t.Fatal("replay accepted tampered ledger")
	}
	var ce *model.CodedError
	if !errors.As(err, &ce) || ce.Code != model.CodeReplay {
		t.Fatalf("expected E_REPLAY, got %v", err)
	}
	details, _ := json.Marshal(ce.Details)
	if !strings.Contains(string(details), `"source_id":"`+source+`"`) ||
		!strings.Contains(string(details), `"window_start_ms":`+jsonNumber(window)) ||
		!strings.Contains(string(details), `"field":"event_decisions`) {
		t.Fatalf("mismatch did not identify first source/window/field: %s", details)
	}
}

func TestMonotonicVersionRejectsChangedSpecAtSameVersion(t *testing.T) {
	ctx := context.Background()
	store, err := ledger.Open(filepath.Join(t.TempDir(), "ledger.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	loaded1 := loadSamples(t, "traffic-v1.json", "policy-v1.json")
	options := model.Options{GlobalConcurrency: 1, PerKeyConcurrency: 1, WindowMinutes: 1, K: 2, MinSampleEvents: 1}
	if _, err := Run(ctx, store, loaded1, options, -1); err != nil {
		t.Fatal(err)
	}
	spec, _ := os.ReadFile(filepath.Join("..", "..", "samples", "traffic-v2.json"))
	policy, _ := os.ReadFile(filepath.Join("..", "..", "samples", "policy-v2.json"))
	spec = []byte(strings.Replace(string(spec), `"spec_version": 2`, `"spec_version": 1`, 1))
	loadedSameVersion, err := model.LoadBytes(spec, policy)
	if err != nil {
		t.Fatal(err)
	}
	_, err = Run(ctx, store, loadedSameVersion, options, -1)
	if err == nil {
		t.Fatal("changed spec at same version was accepted")
	}
	var ce *model.CodedError
	if !errors.As(err, &ce) || ce.Code != model.CodeSpec || ce.Path != "/spec_version" {
		t.Fatalf("expected E_SPEC at /spec_version, got %v", err)
	}
	current, _, err := store.LatestCurrentRun()
	if err != nil {
		t.Fatal(err)
	}
	if current.SpecVersion != 1 {
		t.Fatal("rejected version change replaced the current view")
	}
}

func jsonNumber(v int64) string { return strconv.FormatInt(v, 10) }
