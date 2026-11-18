package ledger

import (
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	_ "modernc.org/sqlite"

	"flowledger/internal/model"
)

type Store struct {
	db   *sql.DB
	path string
}

type RunRecord struct {
	RunID                 string
	FormatVersion         int
	SpecVersion           int64
	PolicyVersion         int64
	SpecDigest            string
	PolicyDigest          string
	ManifestJSON          string
	ManifestDigest        string
	OptionsJSON           string
	OptionsDigest         string
	SemanticDigest        string
	StreamDigest          string
	ChangeSetJSON         string
	Status                string
	IsCurrent             bool
	Stale                 bool
	CreatedMS             int64
	UpdatedMS             int64
	CompletedWindows      int64
	TotalWindows          int64
	ProgressWindowStartMS sql.NullInt64
	LedgerContentDigest   string
	ReportJSON            sql.NullString
	ReportDigest          sql.NullString
	ErrorJSON             sql.NullString
}

type NewRun struct {
	RunID          string
	FormatVersion  int
	SpecVersion    int64
	PolicyVersion  int64
	SpecDigest     string
	PolicyDigest   string
	ManifestJSON   string
	ManifestDigest string
	OptionsJSON    string
	OptionsDigest  string
	SemanticDigest string
	StreamDigest   string
	ChangeSetJSON  string
	TotalWindows   int64
}

type TaskRecord struct {
	RunID                string
	WindowStartMS        int64
	WindowEndMS          int64
	Ordinal              int64
	Status               string
	Attempts             int64
	RecoveredFromUnknown bool
	SegmentID            sql.NullString
	QueueStartMonoMS     sql.NullInt64
	StartMonoMS          sql.NullInt64
	EndMonoMS            sql.NullInt64
	QueueDurationMS      sql.NullInt64
	DurationMS           sql.NullInt64
	GlobalQuota          int64
	KeyQuotaJSON         string
	ResultDigest         sql.NullString
	ErrorJSON            sql.NullString
}

type WindowRow struct {
	RunID              string
	WindowStartMS      int64
	WindowEndMS        int64
	TotalEventCount    int64
	SampledEventCount  int64
	UnseenEventCount   int64
	SampleSetDigest    string
	DecisionSetDigest  string
	AggregationDigest  string
	RankingDigest      string
	EstimateApplicable bool
	TopKGuaranteed     bool
	ResultJSON         string
	ResultDigest       string
}

type SegmentRecord struct {
	RunID         string
	SegmentID     string
	StartedWallMS int64
	ClockBasis    string
}

func Open(path string) (*Store, error) {
	if strings.TrimSpace(path) == "" {
		return nil, model.NewError(model.CodeIO, "/ledger", "ledger path must not be empty")
	}
	abs, err := filepath.Abs(path)
	if err == nil {
		path = abs
	}
	if dir := filepath.Dir(path); dir != "." && dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return nil, &model.CodedError{Code: model.CodeIO, Path: path, Message: err.Error()}
		}
	}
	db, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, &model.CodedError{Code: model.CodeLedger, Path: path, Message: err.Error()}
	}
	db.SetMaxOpenConns(1)
	db.SetMaxIdleConns(1)
	if err := db.Ping(); err != nil {
		_ = db.Close()
		return nil, &model.CodedError{Code: model.CodeLedger, Path: path, Message: err.Error()}
	}
	for _, pragma := range []string{
		"PRAGMA journal_mode=WAL",
		"PRAGMA synchronous=FULL",
		"PRAGMA foreign_keys=ON",
		"PRAGMA busy_timeout=5000",
	} {
		if _, err := db.Exec(pragma); err != nil {
			_ = db.Close()
			return nil, &model.CodedError{Code: model.CodeLedger, Path: path, Message: err.Error()}
		}
	}
	s := &Store{db: db, path: path}
	if err := s.migrate(); err != nil {
		_ = db.Close()
		return nil, err
	}
	return s, nil
}

func (s *Store) Close() error { return s.db.Close() }
func (s *Store) Path() string { return s.path }

func (s *Store) migrate() error {
	if _, err := s.db.Exec(schemaSQL); err != nil {
		return &model.CodedError{Code: model.CodeLedger, Path: s.path, Message: "schema migration failed: " + err.Error()}
	}
	var existing string
	err := s.db.QueryRow(`SELECT value FROM ledger_meta WHERE key='ledger_format_version'`).Scan(&existing)
	switch {
	case errors.Is(err, sql.ErrNoRows):
		if _, err := s.db.Exec(`INSERT INTO ledger_meta(key,value) VALUES('ledger_format_version',?)`, fmt.Sprint(ledgerFormatVersion)); err != nil {
			return &model.CodedError{Code: model.CodeLedger, Path: s.path, Message: err.Error()}
		}
	case err != nil:
		return &model.CodedError{Code: model.CodeLedger, Path: s.path, Message: err.Error()}
	case existing != fmt.Sprint(ledgerFormatVersion):
		return model.NewError(model.CodeLedger, "/ledger_format_version", fmt.Sprintf("unsupported ledger format version %s (expected %d)", existing, ledgerFormatVersion))
	}
	return nil
}

func (s *Store) GetRun(runID string) (RunRecord, bool, error) {
	row := s.db.QueryRow(runSelectSQL+` WHERE run_id=?`, runID)
	r, err := scanRun(row)
	if errors.Is(err, sql.ErrNoRows) {
		return RunRecord{}, false, nil
	}
	if err != nil {
		return RunRecord{}, false, &model.CodedError{Code: model.CodeLedger, Path: s.path, Message: err.Error()}
	}
	return r, true, nil
}

const runSelectSQL = `SELECT run_id,format_version,spec_version,policy_version,spec_digest,policy_digest,
 manifest_json,manifest_digest,options_json,options_digest,semantic_digest,stream_digest,
 change_set_json,status,is_current,stale,created_ms,updated_ms,completed_windows,total_windows,
 progress_window_start_ms,ledger_content_digest,report_json,report_digest,error_json FROM runs`

type rowScanner interface{ Scan(dest ...any) error }

func scanRun(row rowScanner) (RunRecord, error) {
	var r RunRecord
	err := row.Scan(
		&r.RunID, &r.FormatVersion, &r.SpecVersion, &r.PolicyVersion, &r.SpecDigest, &r.PolicyDigest,
		&r.ManifestJSON, &r.ManifestDigest, &r.OptionsJSON, &r.OptionsDigest, &r.SemanticDigest, &r.StreamDigest,
		&r.ChangeSetJSON, &r.Status, &r.IsCurrent, &r.Stale, &r.CreatedMS, &r.UpdatedMS,
		&r.CompletedWindows, &r.TotalWindows, &r.ProgressWindowStartMS, &r.LedgerContentDigest,
		&r.ReportJSON, &r.ReportDigest, &r.ErrorJSON,
	)
	return r, err
}

func (s *Store) LatestCurrentRun() (RunRecord, bool, error) {
	row := s.db.QueryRow(runSelectSQL + ` WHERE is_current=1 AND stale=0 ORDER BY created_ms DESC LIMIT 1`)
	r, err := scanRun(row)
	if errors.Is(err, sql.ErrNoRows) {
		return RunRecord{}, false, nil
	}
	if err != nil {
		return RunRecord{}, false, &model.CodedError{Code: model.CodeLedger, Path: s.path, Message: err.Error()}
	}
	return r, true, nil
}

func (s *Store) ListRuns() ([]RunRecord, error) {
	rows, err := s.db.Query(runSelectSQL + ` ORDER BY rowid ASC`)
	if err != nil {
		return nil, &model.CodedError{Code: model.CodeLedger, Path: s.path, Message: err.Error()}
	}
	defer rows.Close()
	var out []RunRecord
	for rows.Next() {
		r, err := scanRun(rows)
		if err != nil {
			return nil, &model.CodedError{Code: model.CodeLedger, Path: s.path, Message: err.Error()}
		}
		out = append(out, r)
	}
	return out, rows.Err()
}

type keyTask struct {
	WindowStartMS int64
	WindowEndMS   int64
	Ordinal       int64
	Keys          []string
}

func (s *Store) InsertRun(newRun NewRun, jobs []model.WindowJob, nowMS int64) error {
	tx, err := s.db.Begin()
	if err != nil {
		return ledgerErr(s.path, err)
	}
	defer tx.Rollback()
	if _, err := tx.Exec(`UPDATE runs SET is_current=0, stale=1, updated_ms=?,
		status=CASE WHEN status='completed' THEN status ELSE 'superseded' END WHERE is_current=1`, nowMS); err != nil {
		return ledgerErr(s.path, err)
	}
	if _, err := tx.Exec(`INSERT INTO runs(
		run_id,format_version,spec_version,policy_version,spec_digest,policy_digest,
		manifest_json,manifest_digest,options_json,options_digest,semantic_digest,stream_digest,
		change_set_json,status,is_current,stale,created_ms,updated_ms,completed_windows,total_windows,
		progress_window_start_ms,ledger_content_digest,report_json,report_digest,error_json)
		VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,'running',1,0,?,?,0,?,NULL,?,NULL,NULL,NULL)`,
		newRun.RunID, newRun.FormatVersion, newRun.SpecVersion, newRun.PolicyVersion, newRun.SpecDigest, newRun.PolicyDigest,
		newRun.ManifestJSON, newRun.ManifestDigest, newRun.OptionsJSON, newRun.OptionsDigest, newRun.SemanticDigest, newRun.StreamDigest,
		newRun.ChangeSetJSON, nowMS, nowMS, newRun.TotalWindows, contentDigest(nil)); err != nil {
		return ledgerErr(s.path, err)
	}
	for i, job := range jobs {
		keyJSON, err := model.CanonicalJSON(job.Keys)
		if err != nil {
			return ledgerErr(s.path, err)
		}
		if _, err := tx.Exec(`INSERT INTO window_tasks(
			run_id,window_start_ms,window_end_ms,ordinal,status,attempts,recovered_from_unknown,
			segment_id,queue_start_mono_ms,start_mono_ms,end_mono_ms,queue_duration_ms,duration_ms,
			global_quota,key_quota_json,result_digest,error_json)
			VALUES(?,?,?,?,'queued',0,0,NULL,NULL,NULL,NULL,NULL,NULL,1,?,NULL,NULL)`,
			newRun.RunID, job.WindowStartMS, job.WindowEndMS, int64(i+1), string(keyJSON)); err != nil {
			return ledgerErr(s.path, err)
		}
	}
	if err := tx.Commit(); err != nil {
		return ledgerErr(s.path, err)
	}
	return nil
}

func (s *Store) CreateSegment(runID, segmentID string, wallMS int64) error {
	_, err := s.db.Exec(`INSERT INTO scheduler_segments(run_id,segment_id,started_wall_ms,clock_basis)
		VALUES(?,?,?,?)`, runID, segmentID, wallMS, "process-monotonic-counter-ms")
	if err != nil {
		return ledgerErr(s.path, err)
	}
	return nil
}

func (s *Store) MarkRunningUnknown(runID string, nowMS int64) (int64, error) {
	tx, err := s.db.Begin()
	if err != nil {
		return 0, ledgerErr(s.path, err)
	}
	defer tx.Rollback()
	res, err := tx.Exec(`UPDATE window_tasks
		SET status='unknown', recovered_from_unknown=1, error_json=?
		WHERE run_id=? AND status='running'`,
		`{"code":"E_RECOVERED_UNKNOWN","message":"task was running when the prior process ended; result is unknown and must be rerun"}`,
		runID)
	if err != nil {
		return 0, ledgerErr(s.path, err)
	}
	n, _ := res.RowsAffected()
	if _, err := tx.Exec(`UPDATE runs SET status='running', updated_ms=?, error_json=NULL WHERE run_id=?`, nowMS, runID); err != nil {
		return 0, ledgerErr(s.path, err)
	}
	if err := tx.Commit(); err != nil {
		return 0, ledgerErr(s.path, err)
	}
	return n, nil
}

func (s *Store) ListTasks(runID string) ([]TaskRecord, error) {
	rows, err := s.db.Query(`SELECT run_id,window_start_ms,window_end_ms,ordinal,status,attempts,
		recovered_from_unknown,segment_id,queue_start_mono_ms,start_mono_ms,end_mono_ms,
		queue_duration_ms,duration_ms,global_quota,key_quota_json,result_digest,error_json
		FROM window_tasks WHERE run_id=? ORDER BY window_start_ms ASC`, runID)
	if err != nil {
		return nil, ledgerErr(s.path, err)
	}
	defer rows.Close()
	var out []TaskRecord
	for rows.Next() {
		var t TaskRecord
		if err := rows.Scan(
			&t.RunID, &t.WindowStartMS, &t.WindowEndMS, &t.Ordinal, &t.Status, &t.Attempts,
			&t.RecoveredFromUnknown, &t.SegmentID, &t.QueueStartMonoMS, &t.StartMonoMS, &t.EndMonoMS,
			&t.QueueDurationMS, &t.DurationMS, &t.GlobalQuota, &t.KeyQuotaJSON, &t.ResultDigest, &t.ErrorJSON,
		); err != nil {
			return nil, ledgerErr(s.path, err)
		}
		out = append(out, t)
	}
	return out, rows.Err()
}

func (s *Store) StartTask(runID string, windowStartMS int64, segmentID string, queueStartMS, startMS int64) error {
	queueDur := startMS - queueStartMS
	if queueDur < 0 {
		queueDur = 0
	}
	res, err := s.db.Exec(`UPDATE window_tasks SET
		status='running', attempts=attempts+1, segment_id=?,
		queue_start_mono_ms=?, start_mono_ms=?, end_mono_ms=NULL,
		queue_duration_ms=?, duration_ms=NULL, error_json=NULL
		WHERE run_id=? AND window_start_ms=? AND status IN ('queued','unknown')`,
		segmentID, queueStartMS, startMS, queueDur, runID, windowStartMS)
	if err != nil {
		return ledgerErr(s.path, err)
	}
	n, _ := res.RowsAffected()
	if n != 1 {
		return model.NewError(model.CodeLedger, "/window_tasks", fmt.Sprintf("cannot start window %d from queued/unknown state", windowStartMS))
	}
	return nil
}

func (s *Store) CompleteTask(runID string, segmentID string, endMS int64, result model.WindowResult, decisions []model.DecisionRecord, resultJSON, resultDigest string, nowMS int64) error {
	tx, err := s.db.Begin()
	if err != nil {
		return ledgerErr(s.path, err)
	}
	defer tx.Rollback()
	var status, seg sql.NullString
	var start sql.NullInt64
	if err := tx.QueryRow(`SELECT status,segment_id,start_mono_ms FROM window_tasks WHERE run_id=? AND window_start_ms=?`,
		runID, result.WindowStartMS).Scan(&status, &seg, &start); err != nil {
		return ledgerErr(s.path, err)
	}
	if status.String != "running" || seg.String != segmentID || !start.Valid {
		return model.NewError(model.CodeLedger, "/window_tasks", fmt.Sprintf("window %d is not running in segment %s", result.WindowStartMS, segmentID))
	}
	if endMS < start.Int64 {
		return model.NewError(model.CodeLedger, "/window_tasks/end_mono_ms", "end precedes start")
	}
	if _, err := tx.Exec(`DELETE FROM event_decisions WHERE run_id=? AND window_start_ms=?`, runID, result.WindowStartMS); err != nil {
		return ledgerErr(s.path, err)
	}
	if _, err := tx.Exec(`DELETE FROM key_aggregates WHERE run_id=? AND window_start_ms=?`, runID, result.WindowStartMS); err != nil {
		return ledgerErr(s.path, err)
	}
	if _, err := tx.Exec(`DELETE FROM window_results WHERE run_id=? AND window_start_ms=?`, runID, result.WindowStartMS); err != nil {
		return ledgerErr(s.path, err)
	}
	for _, d := range decisions {
		if _, err := tx.Exec(`INSERT INTO event_decisions(
			run_id,window_start_ms,source_id,event_seq,event_id,timestamp_ms,client_key,weight,
			strategy_id,policy_version,bucket,threshold,sampled,decision_digest)
			VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)`,
			runID, d.WindowStartMS, d.SourceID, d.EventSeq, d.EventID, d.TimestampMS, d.ClientKey, d.Weight,
			d.StrategyID, d.PolicyVersion, d.Bucket, d.Threshold, d.Sampled, d.DecisionDigest); err != nil {
			return ledgerErr(s.path, err)
		}
	}
	for _, a := range result.Aggregates {
		sourceJSON, err := model.CanonicalJSON(a.SourceIDs)
		if err != nil {
			return ledgerErr(s.path, err)
		}
		if _, err := tx.Exec(`INSERT INTO key_aggregates(
			run_id,window_start_ms,client_key,source_ids_json,sample_count,sample_weight,
			estimated_count_milli,estimator_lower_milli,estimator_upper_milli,
			observed_count_lower,observed_count_upper,rank)
			VALUES(?,?,?,?,?,?,?,?,?,?,?,?)`,
			runID, result.WindowStartMS, a.ClientKey, string(sourceJSON), a.SampleCount, a.SampleWeight,
			a.EstimatedCountMilli, a.EstimatorLowerMilli, a.EstimatorUpperMilli,
			a.ObservedCountLower, a.ObservedCountUpper, a.Rank); err != nil {
			return ledgerErr(s.path, err)
		}
	}
	if _, err := tx.Exec(`INSERT INTO window_results(
		run_id,window_start_ms,window_end_ms,total_event_count,sampled_event_count,unseen_event_count,
		sample_set_digest,decision_set_digest,aggregation_digest,ranking_digest,estimate_applicable,
		top_k_guaranteed,result_json,result_digest) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)`,
		runID, result.WindowStartMS, result.WindowEndMS, result.TotalEventCount,
		result.SampledEventCount, result.UnseenEventCount, result.SampleSetDigest,
		result.DecisionSetDigest, result.AggregationDigest, result.RankingDigest,
		result.EstimateApplicable, result.TopKGuaranteed, resultJSON, resultDigest); err != nil {
		return ledgerErr(s.path, err)
	}
	var taskStart sql.NullInt64
	if err := tx.QueryRow(`SELECT start_mono_ms FROM window_tasks WHERE run_id=? AND window_start_ms=?`,
		runID, result.WindowStartMS).Scan(&taskStart); err != nil {
		return ledgerErr(s.path, err)
	}
	duration := endMS - taskStart.Int64
	if duration < 0 {
		duration = 0
	}
	res, err := tx.Exec(`UPDATE window_tasks SET status='completed',end_mono_ms=?,duration_ms=?,
		result_digest=?,error_json=NULL WHERE run_id=? AND window_start_ms=? AND status='running' AND segment_id=?`,
		endMS, duration, resultDigest, runID, result.WindowStartMS, segmentID)
	if err != nil {
		return ledgerErr(s.path, err)
	}
	n, _ := res.RowsAffected()
	if n != 1 {
		return model.NewError(model.CodeLedger, "/window_tasks", "completion lost its running task")
	}
	var completed int64
	var cursor sql.NullInt64
	if err := tx.QueryRow(`SELECT COUNT(*), MAX(window_start_ms) FROM window_results WHERE run_id=?`, runID).Scan(&completed, &cursor); err != nil {
		return ledgerErr(s.path, err)
	}
	digest, err := recomputeContentDigestTx(tx, runID)
	if err != nil {
		return err
	}
	if _, err := tx.Exec(`UPDATE runs SET completed_windows=?,progress_window_start_ms=?,ledger_content_digest=?,updated_ms=? WHERE run_id=?`,
		completed, cursor, digest, nowMS, runID); err != nil {
		return ledgerErr(s.path, err)
	}
	if err := tx.Commit(); err != nil {
		return ledgerErr(s.path, err)
	}
	return nil
}

func (s *Store) ListWindowRows(runID string) ([]WindowRow, error) {
	rows, err := s.db.Query(`SELECT run_id,window_start_ms,window_end_ms,total_event_count,sampled_event_count,
		unseen_event_count,sample_set_digest,decision_set_digest,aggregation_digest,ranking_digest,
		estimate_applicable,top_k_guaranteed,result_json,result_digest
		FROM window_results WHERE run_id=? ORDER BY window_start_ms ASC`, runID)
	if err != nil {
		return nil, ledgerErr(s.path, err)
	}
	defer rows.Close()
	var out []WindowRow
	for rows.Next() {
		var w WindowRow
		if err := rows.Scan(&w.RunID, &w.WindowStartMS, &w.WindowEndMS, &w.TotalEventCount,
			&w.SampledEventCount, &w.UnseenEventCount, &w.SampleSetDigest, &w.DecisionSetDigest,
			&w.AggregationDigest, &w.RankingDigest, &w.EstimateApplicable, &w.TopKGuaranteed,
			&w.ResultJSON, &w.ResultDigest); err != nil {
			return nil, ledgerErr(s.path, err)
		}
		out = append(out, w)
	}
	return out, rows.Err()
}

func (s *Store) GetWindowResult(runID string, windowStartMS int64) (WindowRow, bool, error) {
	row := s.db.QueryRow(`SELECT run_id,window_start_ms,window_end_ms,total_event_count,sampled_event_count,
		unseen_event_count,sample_set_digest,decision_set_digest,aggregation_digest,ranking_digest,
		estimate_applicable,top_k_guaranteed,result_json,result_digest
		FROM window_results WHERE run_id=? AND window_start_ms=?`, runID, windowStartMS)
	var w WindowRow
	err := row.Scan(&w.RunID, &w.WindowStartMS, &w.WindowEndMS, &w.TotalEventCount,
		&w.SampledEventCount, &w.UnseenEventCount, &w.SampleSetDigest, &w.DecisionSetDigest,
		&w.AggregationDigest, &w.RankingDigest, &w.EstimateApplicable, &w.TopKGuaranteed,
		&w.ResultJSON, &w.ResultDigest)
	if errors.Is(err, sql.ErrNoRows) {
		return WindowRow{}, false, nil
	}
	if err != nil {
		return WindowRow{}, false, ledgerErr(s.path, err)
	}
	return w, true, nil
}

func (s *Store) ListDecisions(runID string, windowStartMS int64) ([]model.DecisionRecord, error) {
	rows, err := s.db.Query(`SELECT window_start_ms,source_id,event_seq,event_id,timestamp_ms,client_key,
		weight,strategy_id,policy_version,bucket,threshold,sampled,decision_digest
		FROM event_decisions WHERE run_id=? AND window_start_ms=?
		ORDER BY timestamp_ms ASC,source_id ASC,event_seq ASC`, runID, windowStartMS)
	if err != nil {
		return nil, ledgerErr(s.path, err)
	}
	defer rows.Close()
	var out []model.DecisionRecord
	for rows.Next() {
		var d model.DecisionRecord
		if err := rows.Scan(&d.WindowStartMS, &d.SourceID, &d.EventSeq, &d.EventID, &d.TimestampMS,
			&d.ClientKey, &d.Weight, &d.StrategyID, &d.PolicyVersion, &d.Bucket, &d.Threshold,
			&d.Sampled, &d.DecisionDigest); err != nil {
			return nil, ledgerErr(s.path, err)
		}
		out = append(out, d)
	}
	return out, rows.Err()
}

func (s *Store) ListSegments(runID string) ([]SegmentRecord, error) {
	rows, err := s.db.Query(`SELECT run_id,segment_id,started_wall_ms,clock_basis
		FROM scheduler_segments WHERE run_id=? ORDER BY started_wall_ms ASC,segment_id ASC`, runID)
	if err != nil {
		return nil, ledgerErr(s.path, err)
	}
	defer rows.Close()
	var out []SegmentRecord
	for rows.Next() {
		var r SegmentRecord
		if err := rows.Scan(&r.RunID, &r.SegmentID, &r.StartedWallMS, &r.ClockBasis); err != nil {
			return nil, ledgerErr(s.path, err)
		}
		out = append(out, r)
	}
	return out, rows.Err()
}

func (s *Store) FinalizeRun(runID, reportJSON, reportDigest, summaryDigest string, nowMS int64) error {
	tx, err := s.db.Begin()
	if err != nil {
		return ledgerErr(s.path, err)
	}
	defer tx.Rollback()
	var completed, total int64
	if err := tx.QueryRow(`SELECT completed_windows,total_windows FROM runs WHERE run_id=?`, runID).Scan(&completed, &total); err != nil {
		return ledgerErr(s.path, err)
	}
	if completed != total {
		return model.NewError(model.CodeLedger, "/progress", fmt.Sprintf("cannot finalize: %d of %d windows committed", completed, total))
	}
	if _, err := tx.Exec(`UPDATE runs SET status='completed',report_json=?,report_digest=?,
		ledger_content_digest=?,updated_ms=? WHERE run_id=? AND is_current=1 AND stale=0`,
		reportJSON, reportDigest, summaryDigest, nowMS, runID); err != nil {
		return ledgerErr(s.path, err)
	}
	if err := tx.Commit(); err != nil {
		return ledgerErr(s.path, err)
	}
	return nil
}

func (s *Store) SetRunError(runID string, err error, nowMS int64) {
	b := model.ErrorJSON(err)
	_, _ = s.db.Exec(`UPDATE runs SET error_json=?,updated_ms=? WHERE run_id=?`, string(b), nowMS, runID)
}

func (s *Store) AllRunIDs() ([]string, error) {
	rows, err := s.db.Query(`SELECT run_id FROM runs ORDER BY created_ms ASC`)
	if err != nil {
		return nil, ledgerErr(s.path, err)
	}
	defer rows.Close()
	var out []string
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return nil, ledgerErr(s.path, err)
		}
		out = append(out, id)
	}
	return out, rows.Err()
}

func (s *Store) CountRows(table, runID string) (int64, error) {
	allowed := map[string]bool{"event_decisions": true, "key_aggregates": true, "window_results": true, "window_tasks": true}
	if !allowed[table] {
		return 0, fmt.Errorf("unsupported table")
	}
	var n int64
	if err := s.db.QueryRow(`SELECT COUNT(*) FROM `+table+` WHERE run_id=?`, runID).Scan(&n); err != nil {
		return 0, ledgerErr(s.path, err)
	}
	return n, nil
}

func contentDigest(parts []string) string {
	all := []string{"flowledger-run-content-v1"}
	all = append(all, parts...)
	return model.DigestFields(all...)
}

func recomputeContentDigestTx(tx *sql.Tx, runID string) (string, error) {
	rows, err := tx.Query(`SELECT result_digest FROM window_results WHERE run_id=? ORDER BY window_start_ms ASC`, runID)
	if err != nil {
		return "", ledgerErr("", err)
	}
	defer rows.Close()
	var parts []string
	for rows.Next() {
		var d string
		if err := rows.Scan(&d); err != nil {
			return "", ledgerErr("", err)
		}
		parts = append(parts, d)
	}
	if err := rows.Err(); err != nil {
		return "", ledgerErr("", err)
	}
	return contentDigest(parts), nil
}

func (s *Store) RecomputeContentDigest(runID string) (string, error) {
	rows, err := s.db.Query(`SELECT result_digest FROM window_results WHERE run_id=? ORDER BY window_start_ms ASC`, runID)
	if err != nil {
		return "", ledgerErr(s.path, err)
	}
	defer rows.Close()
	var parts []string
	for rows.Next() {
		var d string
		if err := rows.Scan(&d); err != nil {
			return "", ledgerErr(s.path, err)
		}
		parts = append(parts, d)
	}
	if err := rows.Err(); err != nil {
		return "", ledgerErr(s.path, err)
	}
	return contentDigest(parts), nil
}

func ledgerErr(path string, err error) error {
	if err == nil {
		return nil
	}
	return &model.CodedError{Code: model.CodeLedger, Path: path, Message: err.Error()}
}

// DecodeJSON decodes a stored canonical JSON value and attaches the SQL path.
func DecodeJSON(raw, path string, v any) error {
	if err := json.Unmarshal([]byte(raw), v); err != nil {
		return &model.CodedError{Code: model.CodeVerify, Path: path, Message: "invalid stored JSON: " + err.Error()}
	}
	return nil
}

func SortedStrings(keys []string) []string {
	cp := append([]string(nil), keys...)
	sort.Strings(cp)
	return cp
}

func NowMS() int64 { return time.Now().UTC().UnixMilli() }

type KeyAggregateRow struct {
	RunID               string
	WindowStartMS       int64
	ClientKey           string
	SourceIDsJSON       string
	SampleCount         int64
	SampleWeight        int64
	EstimatedCountMilli int64
	EstimatorLowerMilli int64
	EstimatorUpperMilli int64
	ObservedCountLower  int64
	ObservedCountUpper  int64
	Rank                int64
}

func (s *Store) ListKeyAggregates(runID string, windowStartMS int64) ([]KeyAggregateRow, error) {
	rows, err := s.db.Query(`SELECT run_id,window_start_ms,client_key,source_ids_json,sample_count,sample_weight,
		estimated_count_milli,estimator_lower_milli,estimator_upper_milli,observed_count_lower,observed_count_upper,rank
		FROM key_aggregates WHERE run_id=? AND window_start_ms=? ORDER BY client_key ASC`, runID, windowStartMS)
	if err != nil {
		return nil, ledgerErr(s.path, err)
	}
	defer rows.Close()
	var out []KeyAggregateRow
	for rows.Next() {
		var r KeyAggregateRow
		if err := rows.Scan(&r.RunID, &r.WindowStartMS, &r.ClientKey, &r.SourceIDsJSON, &r.SampleCount, &r.SampleWeight,
			&r.EstimatedCountMilli, &r.EstimatorLowerMilli, &r.EstimatorUpperMilli,
			&r.ObservedCountLower, &r.ObservedCountUpper, &r.Rank); err != nil {
			return nil, ledgerErr(s.path, err)
		}
		out = append(out, r)
	}
	return out, rows.Err()
}

func (s *Store) LedgerFormatVersion() (int, error) {
	var raw string
	if err := s.db.QueryRow(`SELECT value FROM ledger_meta WHERE key='ledger_format_version'`).Scan(&raw); err != nil {
		return 0, ledgerErr(s.path, err)
	}
	var v int
	if _, err := fmt.Sscanf(raw, "%d", &v); err != nil {
		return 0, ledgerErr(s.path, err)
	}
	return v, nil
}
