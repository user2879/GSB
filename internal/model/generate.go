package model

import (
	"fmt"
	"sort"
	"strconv"
)

const splitMixGamma uint64 = 0x9e3779b97f4a7c15

type Event struct {
	SourceID    string `json:"source_id"`
	Seq         int64  `json:"event_seq"`
	EventID     string `json:"event_id"`
	TimestampMS int64  `json:"timestamp_ms"`
	ClientKey   string `json:"client_key"`
	Weight      int64  `json:"weight"`
}

type Stream struct {
	Events       []Event           `json:"events"`
	SourceDigest map[string]string `json:"source_digest"`
	StreamDigest string            `json:"stream_digest"`
	TotalEvents  int64             `json:"total_events"`
	MinTimestamp int64             `json:"min_timestamp_ms"`
	MaxTimestamp int64             `json:"max_timestamp_ms"`
}

func splitMix64(x uint64) uint64 {
	x += splitMixGamma
	x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9
	x = (x ^ (x >> 27)) * 0x94d049bb133111eb
	return x ^ (x >> 31)
}

func Generate(loaded *Loaded) (*Stream, error) {
	sources := append([]Source(nil), loaded.Spec.Sources...)
	sort.Slice(sources, func(i, j int) bool { return sources[i].SourceID < sources[j].SourceID })
	stream := &Stream{SourceDigest: make(map[string]string)}
	var all []Event
	for _, src := range sources {
		events, digest, err := generateSource(src)
		if err != nil {
			return nil, err
		}
		stream.SourceDigest[src.SourceID] = digest
		all = append(all, events...)
	}
	sort.Slice(all, func(i, j int) bool {
		if all[i].TimestampMS != all[j].TimestampMS {
			return all[i].TimestampMS < all[j].TimestampMS
		}
		if all[i].SourceID != all[j].SourceID {
			return all[i].SourceID < all[j].SourceID
		}
		return all[i].Seq < all[j].Seq
	})
	stream.Events = all
	stream.TotalEvents = int64(len(all))
	if len(all) > 0 {
		stream.MinTimestamp = all[0].TimestampMS
		stream.MaxTimestamp = all[len(all)-1].TimestampMS
	}
	sourceIDs := make([]string, 0, len(stream.SourceDigest))
	for sid := range stream.SourceDigest {
		sourceIDs = append(sourceIDs, sid)
	}
	sort.Strings(sourceIDs)
	parts := []string{"flowledger-event-stream-v1", loaded.SpecDigest}
	for _, sid := range sourceIDs {
		parts = append(parts, sid, stream.SourceDigest[sid])
	}
	stream.StreamDigest = DigestFields(parts...)
	return stream, nil
}

func generateSource(src Source) ([]Event, string, error) {
	var state uint64 = uint64(src.Seed)
	events := make([]Event, 0, src.EventCount)
	digestParts := []string{"flowledger-source-stream-v1", src.SourceID, strconv.FormatInt(src.Seed, 10)}
	weightSpan := uint64(src.WeightMax-src.WeightMin) + 1
	for i := int64(0); i < src.EventCount; i++ {
		state = splitMix64(state)
		keyIdx := int(state % uint64(len(src.Keys)))
		state = splitMix64(state)
		weight := src.WeightMin + int64(state%weightSpan)
		ts := src.StartTimestampMS + i*src.StepMS
		e := Event{
			SourceID:    src.SourceID,
			Seq:         i + 1,
			EventID:     fmt.Sprintf("%s#%012d", src.SourceID, i+1),
			TimestampMS: ts,
			ClientKey:   src.Keys[keyIdx],
			Weight:      weight,
		}
		events = append(events, e)
		digestParts = append(digestParts,
			e.SourceID, strconv.FormatInt(e.Seq, 10), e.EventID,
			strconv.FormatInt(e.TimestampMS, 10), e.ClientKey, strconv.FormatInt(e.Weight, 10),
		)
	}
	return events, DigestFields(digestParts...), nil
}
