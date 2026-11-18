package engine

import (
	"encoding/json"

	"flowledger/internal/ledger"
	"flowledger/internal/model"
)

func LoadCurrent(store *ledger.Store) (ledger.RunRecord, model.RunManifest, Report, error) {
	row, ok, err := store.LatestCurrentRun()
	if err != nil {
		return ledger.RunRecord{}, model.RunManifest{}, Report{}, err
	}
	if !ok {
		return ledger.RunRecord{}, model.RunManifest{}, Report{}, model.NewError(model.CodeVerify, "/runs", "ledger has no current run")
	}
	var manifest model.RunManifest
	if err := json.Unmarshal([]byte(row.ManifestJSON), &manifest); err != nil {
		return ledger.RunRecord{}, model.RunManifest{}, Report{}, model.NewError(model.CodeVerify, "/runs/manifest_json", err.Error())
	}
	report, err := loadStoredReport(row)
	if err != nil {
		return ledger.RunRecord{}, model.RunManifest{}, Report{}, err
	}
	return row, manifest, report, nil
}
