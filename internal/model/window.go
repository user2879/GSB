package model

import (
	"fmt"
	"sort"
)

type WindowJob struct {
	WindowStartMS int64    `json:"window_start_ms"`
	WindowEndMS   int64    `json:"window_end_ms"`
	Events        []Event  `json:"-"`
	Keys          []string `json:"keys"`
}

func WindowStart(timestampMS, windowMS int64) int64 {
	if timestampMS >= 0 {
		return (timestampMS / windowMS) * windowMS
	}
	return -(((-timestampMS + windowMS - 1) / windowMS) * windowMS)
}

func BuildWindowJobs(stream *Stream, windowMS int64) []WindowJob {
	byWindow := make(map[int64][]Event)
	for _, e := range stream.Events {
		start := WindowStart(e.TimestampMS, windowMS)
		byWindow[start] = append(byWindow[start], e)
	}
	starts := make([]int64, 0, len(byWindow))
	for start := range byWindow {
		starts = append(starts, start)
	}
	sort.Slice(starts, func(i, j int) bool { return starts[i] < starts[j] })
	jobs := make([]WindowJob, 0, len(starts))
	for _, start := range starts {
		events := byWindow[start]
		sort.Slice(events, func(i, j int) bool {
			if events[i].TimestampMS != events[j].TimestampMS {
				return events[i].TimestampMS < events[j].TimestampMS
			}
			if events[i].SourceID != events[j].SourceID {
				return events[i].SourceID < events[j].SourceID
			}
			return events[i].Seq < events[j].Seq
		})
		keySet := make(map[string]bool)
		for _, e := range events {
			keySet[e.ClientKey] = true
		}
		keys := make([]string, 0, len(keySet))
		for key := range keySet {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		jobs = append(jobs, WindowJob{
			WindowStartMS: start,
			WindowEndMS:   start + windowMS,
			Events:        events,
			Keys:          keys,
		})
	}
	return jobs
}

func AllWindowStarts(stream *Stream, windowMS int64) []int64 {
	jobs := BuildWindowJobs(stream, windowMS)
	out := make([]int64, len(jobs))
	for i, j := range jobs {
		out[i] = j.WindowStartMS
	}
	return out
}

func ValidateTaskKeys(keys []string) error {
	if len(keys) == 0 {
		return NewError(CodeVerify, "/task/key_quota", "window task has no client-key quota")
	}
	if !sort.StringsAreSorted(keys) {
		return NewError(CodeVerify, "/task/key_quota", "key quota is not lexicographically sorted")
	}
	for i := 1; i < len(keys); i++ {
		if keys[i] == keys[i-1] {
			return NewError(CodeVerify, "/task/key_quota", fmt.Sprintf("duplicate key %q", keys[i]))
		}
	}
	return nil
}
