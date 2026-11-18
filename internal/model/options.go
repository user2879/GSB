package model

import (
	"fmt"
	"sort"
)

const RunFormatVersion = 1

type Options struct {
	GlobalConcurrency int   `json:"global_concurrency"`
	PerKeyConcurrency int   `json:"per_key_concurrency"`
	WindowMinutes     int64 `json:"window_minutes"`
	K                 int   `json:"k"`
	MinSampleEvents   int64 `json:"min_sample_events"`
	TaskDelayMS       int64 `json:"task_delay_ms"`
}

func (o Options) Validate() error {
	if o.GlobalConcurrency <= 0 {
		return NewError(CodeSpec, "/options/global_concurrency", "must be a positive integer")
	}
	if o.PerKeyConcurrency <= 0 {
		return NewError(CodeSpec, "/options/per_key_concurrency", "must be a positive integer")
	}
	if o.WindowMinutes <= 0 {
		return NewError(CodeSpec, "/options/window_minutes", "must be a positive integer")
	}
	if o.WindowMinutes > (1<<63-1)/60000 {
		return NewError(CodeSpec, "/options/window_minutes", "window size overflows int64 milliseconds")
	}
	if o.K <= 0 {
		return NewError(CodeSpec, "/options/k", "must be a positive integer")
	}
	if o.MinSampleEvents < 0 {
		return NewError(CodeSpec, "/options/min_sample_events", "must be non-negative")
	}
	if o.TaskDelayMS < 0 {
		return NewError(CodeSpec, "/options/task_delay_ms", "must be non-negative")
	}
	return nil
}

func (o Options) WindowMS() int64 { return o.WindowMinutes * 60000 }

type RunManifest struct {
	FormatVersion int            `json:"format_version"`
	Spec          Spec           `json:"spec"`
	Policy        Policy         `json:"policy"`
	Options       Options        `json:"options"`
	StreamDigest  string         `json:"event_stream_digest"`
	SourceDigests []SourceDigest `json:"source_digests"`
}

type SourceDigest struct {
	SourceID string `json:"source_id"`
	Digest   string `json:"digest"`
}

func BuildManifest(loaded *Loaded, stream *Stream, options Options) RunManifest {
	ids := make([]string, 0, len(stream.SourceDigest))
	for id := range stream.SourceDigest {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	sourceDigests := make([]SourceDigest, 0, len(ids))
	for _, id := range ids {
		sourceDigests = append(sourceDigests, SourceDigest{SourceID: id, Digest: stream.SourceDigest[id]})
	}
	return RunManifest{
		FormatVersion: RunFormatVersion,
		Spec:          loaded.Spec,
		Policy:        loaded.Policy,
		Options:       options,
		StreamDigest:  stream.StreamDigest,
		SourceDigests: sourceDigests,
	}
}

func LoadedFromManifest(m RunManifest) (*Loaded, error) {
	if m.FormatVersion != RunFormatVersion {
		return nil, NewError(CodeSpec, "/format_version", fmt.Sprintf("unsupported run format %d", m.FormatVersion))
	}
	specBytes, err := CanonicalJSON(m.Spec)
	if err != nil {
		return nil, NewError(CodeSpec, "/spec", err.Error())
	}
	policyBytes, err := CanonicalJSON(m.Policy)
	if err != nil {
		return nil, NewError(CodeSpec, "/policy", err.Error())
	}
	loaded, err := LoadBytes(specBytes, policyBytes)
	if err != nil {
		return nil, err
	}
	return loaded, nil
}
