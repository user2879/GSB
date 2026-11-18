package model

import (
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"sync"
	"testing"
)

const baseSpec = `{
  "spec_version": 1,
  "sources": [
    {"source_id":"src-a","seed":11,"event_count":90,"start_timestamp_ms":1700000000000,"step_ms":1000,
     "keys":["hot","hot","warm","cold"],"weight_min":1,"weight_max":5},
    {"source_id":"src-b","seed":22,"event_count":80,"start_timestamp_ms":1700000000500,"step_ms":1500,
     "keys":["hot","warm","vip"],"weight_min":2,"weight_max":8}
  ]
}`

const basePolicy = `{
  "policy_version": 1,
  "salt": "test-salt",
  "strategies": [
    {"strategy_id":"a","source_ids":["src-a"],"ratio_bp":5000},
    {"strategy_id":"b","source_ids":["src-b"],"ratio_bp":7500}
  ]
}`

func TestQuantizedThresholdBoundaries(t *testing.T) {
	tests := []struct{ ratio, want int }{
		{0, 0}, {1, 1}, {5000, 5000}, {9999, 9999}, {10000, 10001},
	}
	for _, tc := range tests {
		if got := QuantizedThreshold(tc.ratio); got != tc.want {
			t.Errorf("QuantizedThreshold(%d)=%d want %d", tc.ratio, got, tc.want)
		}
	}
}

func TestSpecValidationPaths(t *testing.T) {
	tests := []struct {
		name   string
		spec   string
		policy string
		path   string
	}{
		{
			name:   "duplicate source",
			spec:   `{"spec_version":1,"sources":[{"source_id":"x","seed":1,"event_count":1,"start_timestamp_ms":0,"step_ms":1,"keys":["k"],"weight_min":1,"weight_max":1},{"source_id":"x","seed":2,"event_count":1,"start_timestamp_ms":0,"step_ms":1,"keys":["k"],"weight_min":1,"weight_max":1}]}`,
			policy: `{"policy_version":1,"salt":"s","strategies":[{"strategy_id":"s","source_ids":["x"],"ratio_bp":1}]}`,
			path:   "/sources/1/source_id",
		},
		{
			name:   "zero seed",
			spec:   `{"spec_version":1,"sources":[{"source_id":"x","seed":0,"event_count":1,"start_timestamp_ms":0,"step_ms":1,"keys":["k"],"weight_min":1,"weight_max":1}]}`,
			policy: `{"policy_version":1,"salt":"s","strategies":[{"strategy_id":"s","source_ids":["x"],"ratio_bp":1}]}`,
			path:   "/sources/0/seed",
		},
		{
			name:   "zero events",
			spec:   `{"spec_version":1,"sources":[{"source_id":"x","seed":1,"event_count":0,"start_timestamp_ms":0,"step_ms":1,"keys":["k"],"weight_min":1,"weight_max":1}]}`,
			policy: `{"policy_version":1,"salt":"s","strategies":[{"strategy_id":"s","source_ids":["x"],"ratio_bp":1}]}`,
			path:   "/sources/0/event_count",
		},
		{
			name:   "unsupported client key",
			spec:   `{"spec_version":1,"sources":[{"source_id":"x","seed":1,"event_count":1,"start_timestamp_ms":0,"step_ms":1,"keys":["bad key"],"weight_min":1,"weight_max":1}]}`,
			policy: `{"policy_version":1,"salt":"s","strategies":[{"strategy_id":"s","source_ids":["x"],"ratio_bp":1}]}`,
			path:   "/sources/0/keys/0",
		},
		{
			name:   "ratio above range",
			spec:   baseSpec,
			policy: `{"policy_version":1,"salt":"s","strategies":[{"strategy_id":"a","source_ids":["src-a","src-b"],"ratio_bp":10001}]}`,
			path:   "/strategies/0/ratio_bp",
		},
		{
			name:   "missing source coverage",
			spec:   baseSpec,
			policy: `{"policy_version":1,"salt":"s","strategies":[{"strategy_id":"a","source_ids":["src-a"],"ratio_bp":1}]}`,
			path:   "/strategies",
		},
		{
			name:   "unknown source path preserves input strategy order",
			spec:   `{"spec_version":1,"sources":[{"source_id":"x","seed":1,"event_count":1,"start_timestamp_ms":0,"step_ms":1,"keys":["k"],"weight_min":1,"weight_max":1}]}`,
			policy: `{"policy_version":1,"salt":"s","strategies":[{"strategy_id":"z","source_ids":["x"],"ratio_bp":1},{"strategy_id":"a","source_ids":["missing"],"ratio_bp":1}]}`,
			path:   "/strategies/1/source_ids/0",
		},
		{
			name:   "unknown field",
			spec:   `{"spec_version":1,"extra":true,"sources":[]}`,
			policy: basePolicy,
			path:   "/spec/extra",
		},
		{
			name:   "decimal integer",
			spec:   `{"spec_version":1,"sources":[{"source_id":"x","seed":1.5,"event_count":1,"start_timestamp_ms":0,"step_ms":1,"keys":["k"],"weight_min":1,"weight_max":1}]}`,
			policy: `{"policy_version":1,"salt":"s","strategies":[{"strategy_id":"s","source_ids":["x"],"ratio_bp":1}]}`,
			path:   "/sources/0/seed",
		},
		{
			name:   "trailing JSON",
			spec:   baseSpec + ` {}`,
			policy: basePolicy,
			path:   "/spec",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := LoadBytes([]byte(tc.spec), []byte(tc.policy))
			if err == nil {
				t.Fatal("expected E_SPEC")
			}
			var ce *CodedError
			if !errors.As(err, &ce) || ce.Code != CodeSpec {
				t.Fatalf("expected E_SPEC, got %v", err)
			}
			if ce.Path != tc.path {
				t.Fatalf("path=%q want %q (%v)", ce.Path, tc.path, err)
			}
		})
	}
}

func TestGenerationAndSamplingAreDeterministic(t *testing.T) {
	loaded, err := LoadBytes([]byte(baseSpec), []byte(basePolicy))
	if err != nil {
		t.Fatal(err)
	}
	s1, err := Generate(loaded)
	if err != nil {
		t.Fatal(err)
	}
	s2, err := Generate(loaded)
	if err != nil {
		t.Fatal(err)
	}
	if s1.StreamDigest != s2.StreamDigest {
		t.Fatalf("stream digest changed: %s != %s", s1.StreamDigest, s2.StreamDigest)
	}
	b1, _ := json.Marshal(s1.Events)
	b2, _ := json.Marshal(s2.Events)
	if string(b1) != string(b2) {
		t.Fatal("event bytes changed across generation")
	}
	decisions := make(map[string]string)
	for _, e := range s1.Events {
		d, err := Decide(loaded, e.SourceID, e.ClientKey)
		if err != nil {
			t.Fatal(err)
		}
		key := d.StrategyID + "\x00" + e.ClientKey
		sig := fmt.Sprintf("%d/%t", d.Bucket, d.Sampled)
		if prev, ok := decisions[key]; ok && prev != sig {
			t.Fatalf("unstable decision for %s: %s vs %s", key, prev, sig)
		}
		decisions[key] = sig
	}
}

func TestWindowAggregationIsOrderIndependent(t *testing.T) {
	loaded, err := LoadBytes([]byte(baseSpec), []byte(basePolicy))
	if err != nil {
		t.Fatal(err)
	}
	stream, err := Generate(loaded)
	if err != nil {
		t.Fatal(err)
	}
	options := Options{GlobalConcurrency: 2, PerKeyConcurrency: 2, WindowMinutes: 1, K: 3, MinSampleEvents: 1}
	jobs := BuildWindowJobs(stream, options.WindowMS())
	if len(jobs) < 2 {
		t.Fatalf("expected multiple windows, got %d", len(jobs))
	}
	for _, job := range jobs {
		want, _, err := BuildWindowResult(loaded, job, options)
		if err != nil {
			t.Fatal(err)
		}
		for i, j := 0, len(job.Events)-1; i < j; i, j = i+1, j-1 {
			job.Events[i], job.Events[j] = job.Events[j], job.Events[i]
		}
		got, _, err := BuildWindowResult(loaded, job, options)
		if err != nil {
			t.Fatal(err)
		}
		wb, _ := json.Marshal(want)
		gb, _ := json.Marshal(got)
		if string(wb) != string(gb) {
			t.Fatalf("window %d changed under reversed input", job.WindowStartMS)
		}
	}
}

func TestLowSampleEstimateIsInapplicable(t *testing.T) {
	loaded, err := LoadBytes([]byte(baseSpec), []byte(basePolicy))
	if err != nil {
		t.Fatal(err)
	}
	stream, err := Generate(loaded)
	if err != nil {
		t.Fatal(err)
	}
	options := Options{GlobalConcurrency: 1, PerKeyConcurrency: 1, WindowMinutes: 1, K: 2, MinSampleEvents: stream.TotalEvents + 1}
	job := BuildWindowJobs(stream, options.WindowMS())[0]
	result, _, err := BuildWindowResult(loaded, job, options)
	if err != nil {
		t.Fatal(err)
	}
	if result.EstimateApplicable {
		t.Fatal("estimate should be inapplicable")
	}
	if len(result.Ranking) != 0 {
		t.Fatal("inapplicable estimate must not publish ranking")
	}
	if result.TopKGuaranteed {
		t.Fatal("inapplicable estimate cannot guarantee top-K")
	}
	if result.InapplicabilityReason == "" {
		t.Fatal("missing reason")
	}
}

func TestFullSamplingProducesExactTopKGuarantee(t *testing.T) {
	const fullPolicy = `{
	  "policy_version": 1,
	  "salt": "test-salt",
	  "strategies": [
	    {"strategy_id":"all","source_ids":["src-a","src-b"],"ratio_bp":10000}
	  ]
	}`
	loaded, err := LoadBytes([]byte(baseSpec), []byte(fullPolicy))
	if err != nil {
		t.Fatal(err)
	}
	stream, err := Generate(loaded)
	if err != nil {
		t.Fatal(err)
	}
	options := Options{GlobalConcurrency: 1, PerKeyConcurrency: 1, WindowMinutes: 1, K: 1, MinSampleEvents: 1}
	jobs := BuildWindowJobs(stream, options.WindowMS())
	for _, job := range jobs {
		result, _, err := BuildWindowResult(loaded, job, options)
		if err != nil {
			t.Fatal(err)
		}
		if result.SampledEventCount != result.TotalEventCount || result.UnseenEventCount != 0 {
			t.Fatalf("10000 bp did not sample every event: %+v", result)
		}
		if !result.TopKGuaranteed {
			t.Fatalf("full sample did not guarantee K=1: %s", result.GuaranteeBasis)
		}
		counts := make(map[string]int64)
		weights := make(map[string]int64)
		for _, e := range job.Events {
			counts[e.ClientKey]++
			weights[e.ClientKey] += e.Weight
		}
		keys := make([]string, 0, len(counts))
		for key := range counts {
			keys = append(keys, key)
		}
		sort.Slice(keys, func(i, j int) bool {
			if counts[keys[i]] != counts[keys[j]] {
				return counts[keys[i]] > counts[keys[j]]
			}
			if weights[keys[i]] != weights[keys[j]] {
				return weights[keys[i]] > weights[keys[j]]
			}
			return keys[i] < keys[j]
		})
		if len(result.Ranking) == 0 || result.Ranking[0].ClientKey != keys[0] {
			t.Fatalf("guaranteed top key %q != true top key %q", result.Ranking[0].ClientKey, keys[0])
		}
		for _, a := range result.Aggregates {
			if a.EstimatorLowerMilli != a.EstimatedCountMilli || a.EstimatorUpperMilli != a.EstimatedCountMilli || a.QuantizationErrorBoundMilli != 0 {
				t.Fatalf("full-sample estimate is not exact for %s: %+v", a.ClientKey, a)
			}
		}
	}
}

func TestSamplingConsistentUnderConcurrentScheduling(t *testing.T) {
	loaded, err := LoadBytes([]byte(baseSpec), []byte(basePolicy))
	if err != nil {
		t.Fatal(err)
	}
	stream, err := Generate(loaded)
	if err != nil {
		t.Fatal(err)
	}
	options := Options{GlobalConcurrency: 8, PerKeyConcurrency: 8, WindowMinutes: 1, K: 3, MinSampleEvents: 1}
	jobs := BuildWindowJobs(stream, options.WindowMS())
	baseline := make(map[int64]string)
	for _, job := range jobs {
		result, _, err := BuildWindowResult(loaded, job, options)
		if err != nil {
			t.Fatal(err)
		}
		b, _ := json.Marshal(result)
		baseline[job.WindowStartMS] = string(b)
	}
	type outcome struct {
		window int64
		body   string
		err    error
	}
	ch := make(chan outcome, len(jobs)*16)
	var wg sync.WaitGroup
	for round := 0; round < 16; round++ {
		for _, original := range jobs {
			wg.Add(1)
			job := original
			rotate := round % len(job.Events)
			events := append(append([]Event(nil), job.Events[rotate:]...), job.Events[:rotate]...)
			job.Events = events
			go func() {
				defer wg.Done()
				result, _, err := BuildWindowResult(loaded, job, options)
				if err != nil {
					ch <- outcome{window: job.WindowStartMS, err: err}
					return
				}
				b, _ := json.Marshal(result)
				ch <- outcome{window: job.WindowStartMS, body: string(b)}
			}()
		}
	}
	wg.Wait()
	close(ch)
	for got := range ch {
		if got.err != nil {
			t.Fatal(got.err)
		}
		if got.body != baseline[got.window] {
			t.Fatalf("window %d changed under concurrent/non-canonical scheduling", got.window)
		}
	}
}
