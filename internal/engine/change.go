package engine

import (
	"encoding/json"
	"sort"

	"flowledger/internal/ledger"
	"flowledger/internal/model"
)

func ComputeChangeSet(previous *ledger.RunRecord, newManifest model.RunManifest, newStream *model.Stream, loaded *model.Loaded) (ChangeSet, error) {
	cs := ChangeSet{
		Baseline:           previous == nil,
		SpecVersionTo:      newManifest.Spec.SpecVersion,
		PolicyVersionTo:    newManifest.Policy.PolicyVersion,
		AddedSources:       []string{},
		RemovedSources:     []string{},
		ModifiedSources:    []SourceChange{},
		RatioChanges:       []RatioChange{},
		AffectedWindows:    []int64{},
		OverallReplacement: true,
	}
	newWindows := windowsBySource(newStream, newManifest.Options.WindowMS())
	if previous == nil {
		seen := make(map[int64]bool)
		for _, starts := range newWindows {
			for _, start := range starts {
				seen[start] = true
			}
		}
		cs.AffectedWindows = sortedWindowSet(seen)
		return cs, nil
	}
	cs.PreviousRunID = previous.RunID
	var oldManifest model.RunManifest
	if err := json.Unmarshal([]byte(previous.ManifestJSON), &oldManifest); err != nil {
		return cs, model.NewError(model.CodeLedger, "/runs/manifest_json", err.Error())
	}
	cs.SpecVersionFrom = oldManifest.Spec.SpecVersion
	cs.PolicyVersionFrom = oldManifest.Policy.PolicyVersion
	oldLoaded, err := model.LoadedFromManifest(oldManifest)
	if err != nil {
		return cs, err
	}
	oldStream, err := model.Generate(oldLoaded)
	if err != nil {
		return cs, err
	}
	oldWindows := windowsBySource(oldStream, oldManifest.Options.WindowMS())
	oldSources := make(map[string]model.Source)
	for _, s := range oldManifest.Spec.Sources {
		oldSources[s.SourceID] = s
	}
	newSources := make(map[string]model.Source)
	for _, s := range newManifest.Spec.Sources {
		newSources[s.SourceID] = s
	}
	affected := make(map[int64]bool)
	for id := range newSources {
		if _, ok := oldSources[id]; !ok {
			cs.AddedSources = append(cs.AddedSources, id)
			addWindows(affected, newWindows[id])
		}
	}
	for id := range oldSources {
		if _, ok := newSources[id]; !ok {
			cs.RemovedSources = append(cs.RemovedSources, id)
			addWindows(affected, oldWindows[id])
		}
	}
	for id, ns := range newSources {
		os, ok := oldSources[id]
		if !ok {
			continue
		}
		fields := changedSourceFields(os, ns)
		if len(fields) > 0 {
			cs.ModifiedSources = append(cs.ModifiedSources, SourceChange{SourceID: id, Fields: fields})
			addWindows(affected, oldWindows[id])
			addWindows(affected, newWindows[id])
		}
	}
	sort.Strings(cs.AddedSources)
	sort.Strings(cs.RemovedSources)
	sort.Slice(cs.ModifiedSources, func(i, j int) bool { return cs.ModifiedSources[i].SourceID < cs.ModifiedSources[j].SourceID })
	oldRatios := ratiosBySource(oldManifest.Policy)
	newRatios := ratiosBySource(newManifest.Policy)
	allIDs := make(map[string]bool)
	for id := range oldRatios {
		allIDs[id] = true
	}
	for id := range newRatios {
		allIDs[id] = true
	}
	var ratioIDs []string
	for id := range allIDs {
		ratioIDs = append(ratioIDs, id)
	}
	sort.Strings(ratioIDs)
	for _, id := range ratioIDs {
		o := oldRatios[id]
		n := newRatios[id]
		if o.ratio != n.ratio || o.strategy != n.strategy {
			cs.RatioChanges = append(cs.RatioChanges, RatioChange{
				SourceID: id, OldStrategy: o.strategy, NewStrategy: n.strategy, OldRatioBP: o.ratio, NewRatioBP: n.ratio,
			})
		}
	}
	// A policy version or salt changes the hash domain. The affected window
	// set must therefore include all sources, even if every numeric ratio is
	// unchanged.
	if oldManifest.Policy.PolicyVersion != newManifest.Policy.PolicyVersion || oldManifest.Policy.Salt != newManifest.Policy.Salt {
		for id, starts := range oldWindows {
			addWindows(affected, starts)
			_ = id
		}
		for _, starts := range newWindows {
			addWindows(affected, starts)
		}
	} else {
		for _, rc := range cs.RatioChanges {
			addWindows(affected, oldWindows[rc.SourceID])
			addWindows(affected, newWindows[rc.SourceID])
		}
	}
	cs.AffectedWindows = sortedWindowSet(affected)
	_ = loaded
	return cs, nil
}

type ratioInfo struct {
	strategy string
	ratio    int
}

func ratiosBySource(p model.Policy) map[string]ratioInfo {
	out := make(map[string]ratioInfo)
	for _, st := range p.Strategies {
		for _, sid := range st.SourceIDs {
			out[sid] = ratioInfo{strategy: st.StrategyID, ratio: st.RatioBP}
		}
	}
	return out
}

func changedSourceFields(a, b model.Source) []string {
	var fields []string
	if a.Seed != b.Seed {
		fields = append(fields, "seed")
	}
	if a.EventCount != b.EventCount {
		fields = append(fields, "event_count")
	}
	if a.StartTimestampMS != b.StartTimestampMS {
		fields = append(fields, "start_timestamp_ms")
	}
	if a.StepMS != b.StepMS {
		fields = append(fields, "step_ms")
	}
	if !stringSliceEqual(a.Keys, b.Keys) {
		fields = append(fields, "keys")
	}
	if a.WeightMin != b.WeightMin {
		fields = append(fields, "weight_min")
	}
	if a.WeightMax != b.WeightMax {
		fields = append(fields, "weight_max")
	}
	return fields
}

func stringSliceEqual(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func windowsBySource(stream *model.Stream, windowMS int64) map[string][]int64 {
	out := make(map[string]map[int64]bool)
	for _, e := range stream.Events {
		if out[e.SourceID] == nil {
			out[e.SourceID] = make(map[int64]bool)
		}
		out[e.SourceID][model.WindowStart(e.TimestampMS, windowMS)] = true
	}
	result := make(map[string][]int64, len(out))
	for id, set := range out {
		result[id] = sortedWindowSet(set)
	}
	return result
}

func addWindows(dst map[int64]bool, starts []int64) {
	for _, start := range starts {
		dst[start] = true
	}
}

func sortedWindowSet(set map[int64]bool) []int64 {
	out := make([]int64, 0, len(set))
	for start := range set {
		out = append(out, start)
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}
