package engine

import (
	"database/sql"
	"fmt"
	"sort"

	"flowledger/internal/ledger"
	"flowledger/internal/model"
)

type ChangeSet struct {
	Baseline           bool           `json:"baseline"`
	PreviousRunID      string         `json:"previous_run_id,omitempty"`
	SpecVersionFrom    int64          `json:"spec_version_from,omitempty"`
	SpecVersionTo      int64          `json:"spec_version_to"`
	PolicyVersionFrom  int64          `json:"policy_version_from,omitempty"`
	PolicyVersionTo    int64          `json:"policy_version_to"`
	AddedSources       []string       `json:"added_sources"`
	RemovedSources     []string       `json:"removed_sources"`
	ModifiedSources    []SourceChange `json:"modified_sources"`
	RatioChanges       []RatioChange  `json:"ratio_changes"`
	AffectedWindows    []int64        `json:"affected_windows"`
	OverallReplacement bool           `json:"overall_replacement"`
}

type SourceChange struct {
	SourceID string   `json:"source_id"`
	Fields   []string `json:"fields"`
}

type RatioChange struct {
	SourceID    string `json:"source_id"`
	OldStrategy string `json:"old_strategy_id"`
	NewStrategy string `json:"new_strategy_id"`
	OldRatioBP  int    `json:"old_ratio_bp"`
	NewRatioBP  int    `json:"new_ratio_bp"`
}

type TaskSchedule struct {
	WindowStartMS        int64    `json:"window_start_ms"`
	WindowEndMS          int64    `json:"window_end_ms"`
	Ordinal              int64    `json:"ordinal"`
	Status               string   `json:"status"`
	SegmentID            string   `json:"segment_id"`
	Attempts             int64    `json:"attempts"`
	RecoveredFromUnknown bool     `json:"recovered_from_unknown"`
	QueueStartMonoMS     int64    `json:"queue_start_mono_ms"`
	StartMonoMS          int64    `json:"start_mono_ms"`
	EndMonoMS            int64    `json:"end_mono_ms"`
	QueueDurationMS      int64    `json:"queue_duration_ms"`
	DurationMS           int64    `json:"duration_ms"`
	GlobalQuota          int64    `json:"global_quota"`
	KeyQuota             []string `json:"key_quota"`
	ClockBasis           string   `json:"clock_basis"`
	ResultDigest         string   `json:"result_digest"`
}

type Report struct {
	FormatVersion   int                  `json:"format_version"`
	RunID           string               `json:"run_id"`
	Status          string               `json:"status"`
	IsCurrent       bool                 `json:"is_current"`
	Stale           bool                 `json:"stale"`
	SpecVersion     int64                `json:"spec_version"`
	PolicyVersion   int64                `json:"policy_version"`
	ManifestDigest  string               `json:"manifest_digest"`
	SemanticDigest  string               `json:"semantic_digest"`
	StreamDigest    string               `json:"event_stream_digest"`
	Options         model.Options        `json:"options"`
	ChangeSet       ChangeSet            `json:"change_set"`
	Progress        Progress             `json:"progress"`
	SampleSetDigest string               `json:"sample_set_digest"`
	SummaryDigest   string               `json:"summary_digest"`
	Windows         []model.WindowResult `json:"windows"`
	Schedule        []TaskSchedule       `json:"schedule"`
	LedgerDigest    string               `json:"ledger_content_digest"`
	ReportDigest    string               `json:"report_digest,omitempty"`
}

type Progress struct {
	CompletedWindows      int64 `json:"completed_windows"`
	TotalWindows          int64 `json:"total_windows"`
	ProgressWindowStartMS int64 `json:"progress_window_start_ms"`
}

type RunOutcome struct {
	Report        Report
	RunID         string
	Resumed       bool
	Recovered     int64
	ReusedReport  bool
	CrashInjected bool
}

type pendingTask struct {
	job     model.WindowJob
	ordinal int64
}

type taskResult struct {
	key       int64
	endMS     int64
	result    model.WindowResult
	decisions []model.DecisionRecord
	json      string
	digest    string
	err       error
}

func scheduleFromTask(t ledger.TaskRecord) (TaskSchedule, error) {
	keys := []string{}
	if t.KeyQuotaJSON != "" {
		if err := model.CanonicalJSONUnmarshal([]byte(t.KeyQuotaJSON), &keys); err != nil {
			return TaskSchedule{}, model.NewError(model.CodeVerify, "/window_tasks/key_quota_json", err.Error())
		}
	}
	sort.Strings(keys)
	return TaskSchedule{
		WindowStartMS:        t.WindowStartMS,
		WindowEndMS:          t.WindowEndMS,
		Ordinal:              t.Ordinal,
		Status:               t.Status,
		SegmentID:            nullString(t.SegmentID),
		Attempts:             t.Attempts,
		RecoveredFromUnknown: t.RecoveredFromUnknown,
		QueueStartMonoMS:     nullInt(t.QueueStartMonoMS),
		StartMonoMS:          nullInt(t.StartMonoMS),
		EndMonoMS:            nullInt(t.EndMonoMS),
		QueueDurationMS:      nullInt(t.QueueDurationMS),
		DurationMS:           nullInt(t.DurationMS),
		GlobalQuota:          t.GlobalQuota,
		KeyQuota:             keys,
		ClockBasis:           "process-monotonic-counter-ms",
		ResultDigest:         nullString(t.ResultDigest),
	}, nil
}

func nullString(v sql.NullString) string {
	if !v.Valid {
		return ""
	}
	return v.String
}

func nullInt(v sql.NullInt64) int64 {
	if !v.Valid {
		return 0
	}
	return v.Int64
}

func (r Report) ValidateCompleted() error {
	if r.Status != "completed" {
		return model.NewError(model.CodeVerify, "/status", fmt.Sprintf("run status is %q, not completed", r.Status))
	}
	if r.Stale || !r.IsCurrent {
		return model.NewError(model.CodeVerify, "/stale", "report is not the current, non-stale view")
	}
	return nil
}
