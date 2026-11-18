package model

import (
	"fmt"
	"math/big"
	"sort"
	"strconv"
)

type DecisionRecord struct {
	WindowStartMS  int64  `json:"window_start_ms"`
	SourceID       string `json:"source_id"`
	EventSeq       int64  `json:"event_seq"`
	EventID        string `json:"event_id"`
	TimestampMS    int64  `json:"timestamp_ms"`
	ClientKey      string `json:"client_key"`
	Weight         int64  `json:"weight"`
	StrategyID     string `json:"strategy_id"`
	PolicyVersion  int64  `json:"policy_version"`
	Bucket         int    `json:"bucket"`
	Threshold      int    `json:"threshold"`
	Sampled        bool   `json:"sampled"`
	DecisionDigest string `json:"decision_digest"`
}

type KeyAggregate struct {
	ClientKey                   string   `json:"client_key"`
	SourceIDs                   []string `json:"source_ids"`
	SampleCount                 int64    `json:"sample_count"`
	SampleWeight                int64    `json:"sample_weight"`
	EstimatedCountMilli         int64    `json:"estimated_count_milli"`
	EstimatedCount              string   `json:"estimated_count"`
	EstimatorLowerMilli         int64    `json:"estimator_lower_milli"`
	EstimatorUpperMilli         int64    `json:"estimator_upper_milli"`
	QuantizationErrorBoundMilli int64    `json:"quantization_error_bound_milli"`
	ObservedCountLower          int64    `json:"observed_count_lower"`
	ObservedCountUpper          int64    `json:"observed_count_upper"`
	StrategyIDs                 []string `json:"strategy_ids"`
	Rank                        int      `json:"rank,omitempty"`
}

type RankItem struct {
	Rank                        int    `json:"rank"`
	ClientKey                   string `json:"client_key"`
	ScoreMilli                  int64  `json:"score_milli"`
	Score                       string `json:"score"`
	SampleCount                 int64  `json:"sample_count"`
	SampleWeight                int64  `json:"sample_weight"`
	EstimatedCountMilli         int64  `json:"estimated_count_milli"`
	EstimatorLowerMilli         int64  `json:"estimator_lower_milli"`
	EstimatorUpperMilli         int64  `json:"estimator_upper_milli"`
	QuantizationErrorBoundMilli int64  `json:"quantization_error_bound_milli"`
	ObservedCountLower          int64  `json:"observed_count_lower"`
	ObservedCountUpper          int64  `json:"observed_count_upper"`
	TieBreak                    string `json:"tie_break"`
}

type StrategyErrorBound struct {
	StrategyID                       string   `json:"strategy_id"`
	SourceIDs                        []string `json:"source_ids"`
	RatioBP                          int      `json:"ratio_bp"`
	QuantizedThreshold               int      `json:"quantized_threshold"`
	BucketCount                      int      `json:"bucket_count"`
	Quantization                     string   `json:"quantization"`
	RatioTruncationErrorNumerator    int64    `json:"ratio_truncation_error_numerator"`
	RatioErrorDenominator            int64    `json:"ratio_error_denominator"`
	HashBucketGranularityNumerator   int64    `json:"hash_bucket_granularity_numerator"`
	HashBucketGranularityDenominator int64    `json:"hash_bucket_granularity_denominator"`
	CombinedErrorNumerator           int64    `json:"combined_error_numerator"`
	CombinedErrorDenominator         int64    `json:"combined_error_denominator"`
	EnvelopeLowerNumerator           int64    `json:"envelope_lower_numerator"`
	EnvelopeUpperNumerator           int64    `json:"envelope_upper_numerator"`
	EnvelopeDenominator              int64    `json:"envelope_denominator"`
	Basis                            string   `json:"basis"`
}

type WindowResult struct {
	WindowStartMS         int64                `json:"window_start_ms"`
	WindowEndMS           int64                `json:"window_end_ms"`
	TotalEventCount       int64                `json:"total_event_count"`
	SampledEventCount     int64                `json:"sampled_event_count"`
	UnseenEventCount      int64                `json:"unseen_event_count"`
	SampledWeight         int64                `json:"sampled_weight"`
	DecisionSetDigest     string               `json:"decision_set_digest"`
	SampleSetDigest       string               `json:"sample_set_digest"`
	AggregationDigest     string               `json:"aggregation_digest"`
	RankingDigest         string               `json:"ranking_digest"`
	EstimateApplicable    bool                 `json:"estimate_applicable"`
	InapplicabilityReason string               `json:"inapplicability_reason,omitempty"`
	TopKGuaranteed        bool                 `json:"top_k_guaranteed"`
	GuaranteeBasis        string               `json:"guarantee_basis"`
	Aggregates            []KeyAggregate       `json:"aggregates"`
	Ranking               []RankItem           `json:"ranking"`
	StrategyErrorBounds   []StrategyErrorBound `json:"strategy_error_bounds"`
}

type keyWork struct {
	count      int64
	weight     int64
	strategies map[string]int64
	sources    map[string]bool
}

func BuildWindowResult(loaded *Loaded, job WindowJob, options Options) (WindowResult, []DecisionRecord, error) {
	events := append([]Event(nil), job.Events...)
	sort.Slice(events, func(i, j int) bool {
		if events[i].TimestampMS != events[j].TimestampMS {
			return events[i].TimestampMS < events[j].TimestampMS
		}
		if events[i].SourceID != events[j].SourceID {
			return events[i].SourceID < events[j].SourceID
		}
		return events[i].Seq < events[j].Seq
	})
	decisions := make([]DecisionRecord, 0, len(events))
	decisionParts := []string{"flowledger-window-decisions-v1", strconv.FormatInt(job.WindowStartMS, 10)}
	sampleParts := []string{"flowledger-window-sample-set-v1", strconv.FormatInt(job.WindowStartMS, 10)}
	keyMap := make(map[string]*keyWork)
	var sampledEvents int64
	var sampledWeight int64
	for _, e := range events {
		dec, err := Decide(loaded, e.SourceID, e.ClientKey)
		if err != nil {
			return WindowResult{}, nil, err
		}
		rec := DecisionRecord{
			WindowStartMS: job.WindowStartMS,
			SourceID:      e.SourceID,
			EventSeq:      e.Seq,
			EventID:       e.EventID,
			TimestampMS:   e.TimestampMS,
			ClientKey:     e.ClientKey,
			Weight:        e.Weight,
			StrategyID:    dec.StrategyID,
			PolicyVersion: dec.PolicyVersion,
			Bucket:        dec.Bucket,
			Threshold:     dec.Threshold,
			Sampled:       dec.Sampled,
		}
		rec.DecisionDigest = DigestFields(
			"flowledger-decision-v1",
			strconv.FormatInt(rec.WindowStartMS, 10), rec.SourceID,
			strconv.FormatInt(rec.EventSeq, 10), rec.EventID,
			strconv.FormatInt(rec.TimestampMS, 10), rec.ClientKey,
			strconv.FormatInt(rec.Weight, 10), rec.StrategyID,
			strconv.FormatInt(rec.PolicyVersion, 10), strconv.Itoa(rec.Bucket),
			strconv.Itoa(rec.Threshold), strconv.FormatBool(rec.Sampled),
		)
		decisions = append(decisions, rec)
		decisionParts = append(decisionParts, rec.DecisionDigest)
		if rec.Sampled {
			sampledEvents++
			sampledWeight += rec.Weight
			sampleParts = append(sampleParts, rec.DecisionDigest)
			kw := keyMap[rec.ClientKey]
			if kw == nil {
				kw = &keyWork{strategies: make(map[string]int64), sources: make(map[string]bool)}
				keyMap[rec.ClientKey] = kw
			}
			kw.count++
			kw.weight += rec.Weight
			kw.strategies[rec.StrategyID]++
			kw.sources[rec.SourceID] = true
		}
	}
	totalEvents := int64(len(events))
	unseen := totalEvents - sampledEvents
	aggregates := make([]KeyAggregate, 0, len(keyMap))
	for key, kw := range keyMap {
		agg, err := buildKeyAggregate(loaded, key, kw, unseen)
		if err != nil {
			return WindowResult{}, nil, err
		}
		aggregates = append(aggregates, agg)
	}
	sort.Slice(aggregates, func(i, j int) bool { return aggregates[i].ClientKey < aggregates[j].ClientKey })
	applicable := sampledEvents >= options.MinSampleEvents
	ranking := []RankItem{}
	reason := ""
	if !applicable {
		reason = fmt.Sprintf("sampled_event_count %d is below min_sample_events %d; sampled values are diagnostic only", sampledEvents, options.MinSampleEvents)
	} else {
		ordered := append([]KeyAggregate(nil), aggregates...)
		sort.SliceStable(ordered, func(i, j int) bool {
			if ordered[i].EstimatedCountMilli != ordered[j].EstimatedCountMilli {
				return ordered[i].EstimatedCountMilli > ordered[j].EstimatedCountMilli
			}
			if ordered[i].SampleWeight != ordered[j].SampleWeight {
				return ordered[i].SampleWeight > ordered[j].SampleWeight
			}
			if ordered[i].SampleCount != ordered[j].SampleCount {
				return ordered[i].SampleCount > ordered[j].SampleCount
			}
			return ordered[i].ClientKey < ordered[j].ClientKey
		})
		n := options.K
		if n > len(ordered) {
			n = len(ordered)
		}
		for i := 0; i < n; i++ {
			a := ordered[i]
			ranking = append(ranking, RankItem{
				Rank:                        i + 1,
				ClientKey:                   a.ClientKey,
				ScoreMilli:                  a.EstimatedCountMilli,
				Score:                       a.EstimatedCount,
				SampleCount:                 a.SampleCount,
				SampleWeight:                a.SampleWeight,
				EstimatedCountMilli:         a.EstimatedCountMilli,
				EstimatorLowerMilli:         a.EstimatorLowerMilli,
				EstimatorUpperMilli:         a.EstimatorUpperMilli,
				QuantizationErrorBoundMilli: a.QuantizationErrorBoundMilli,
				ObservedCountLower:          a.ObservedCountLower,
				ObservedCountUpper:          a.ObservedCountUpper,
				TieBreak:                    "estimated_count desc, sample_weight desc, sample_count desc, client_key asc",
			})
		}
		for i := range aggregates {
			for r := range ranking {
				if aggregates[i].ClientKey == ranking[r].ClientKey {
					aggregates[i].Rank = ranking[r].Rank
					break
				}
			}
		}
	}
	aggDigest, err := DigestJSON(aggregates)
	if err != nil {
		return WindowResult{}, nil, err
	}
	rankingDigest, err := DigestJSON(ranking)
	if err != nil {
		return WindowResult{}, nil, err
	}
	guaranteed, basis := EvaluateTopKGuarantee(applicable, reason, aggregates, ranking, unseen, options.K)
	bounds := make([]StrategyErrorBound, 0, len(loaded.Policy.Strategies))
	for _, st := range loaded.Policy.Strategies {
		bounds = append(bounds, buildStrategyErrorBound(st))
	}
	sort.Slice(bounds, func(i, j int) bool { return bounds[i].StrategyID < bounds[j].StrategyID })
	result := WindowResult{
		WindowStartMS:         job.WindowStartMS,
		WindowEndMS:           job.WindowEndMS,
		TotalEventCount:       totalEvents,
		SampledEventCount:     sampledEvents,
		UnseenEventCount:      unseen,
		SampledWeight:         sampledWeight,
		DecisionSetDigest:     DigestFields(decisionParts...),
		SampleSetDigest:       DigestFields(sampleParts...),
		AggregationDigest:     aggDigest,
		RankingDigest:         rankingDigest,
		EstimateApplicable:    applicable,
		InapplicabilityReason: reason,
		TopKGuaranteed:        guaranteed,
		GuaranteeBasis:        basis,
		Aggregates:            aggregates,
		Ranking:               ranking,
		StrategyErrorBounds:   bounds,
	}
	return result, decisions, nil
}

func buildKeyAggregate(loaded *Loaded, key string, kw *keyWork, unseen int64) (KeyAggregate, error) {
	strategyIDs := make([]string, 0, len(kw.strategies))
	for id := range kw.strategies {
		strategyIDs = append(strategyIDs, id)
	}
	sort.Strings(strategyIDs)
	sources := make([]string, 0, len(kw.sources))
	for id := range kw.sources {
		sources = append(sources, id)
	}
	sort.Strings(sources)
	var est, lower, upper int64
	for _, sid := range strategyIDs {
		resolved, ok := loaded.StrategyByID[sid]
		if !ok {
			return KeyAggregate{}, NewError(CodeSpec, "/strategies", "sampled key references unknown strategy")
		}
		th := resolved.QuantizedThreshold
		if th <= 0 {
			return KeyAggregate{}, NewError(CodeSpec, "/strategies", "sampled key has zero threshold strategy")
		}
		c := kw.strategies[sid]
		estPart, err := mulDivFloor(c, int64(BucketCount)*1000, int64(th))
		if err != nil {
			return KeyAggregate{}, err
		}
		lowDen := th + 1
		if lowDen > BucketCount {
			lowDen = BucketCount
		}
		lowPart, err := mulDivFloor(c, int64(BucketCount)*1000, int64(lowDen))
		if err != nil {
			return KeyAggregate{}, err
		}
		highDen := th - 1
		if highDen < 1 {
			highDen = 1
		}
		highPart, err := mulDivCeil(c, int64(BucketCount)*1000, int64(highDen))
		if err != nil {
			return KeyAggregate{}, err
		}
		if th == BucketCount {
			lowPart, highPart = estPart, estPart
		}
		est += estPart
		lower += lowPart
		upper += highPart
	}
	quantErr := est - lower
	if upper-est > quantErr {
		quantErr = upper - est
	}
	return KeyAggregate{
		ClientKey:                   key,
		SourceIDs:                   sources,
		SampleCount:                 kw.count,
		SampleWeight:                kw.weight,
		EstimatedCountMilli:         est,
		EstimatedCount:              FormatMilli(est),
		EstimatorLowerMilli:         lower,
		EstimatorUpperMilli:         upper,
		QuantizationErrorBoundMilli: quantErr,
		ObservedCountLower:          kw.count,
		ObservedCountUpper:          kw.count + unseen,
		StrategyIDs:                 strategyIDs,
	}, nil
}

func buildStrategyErrorBound(st Strategy) StrategyErrorBound {
	th := QuantizedThreshold(st.RatioBP)
	truncNum := int64(st.RatioBP)*BucketCount - int64(th)*10000
	lowNum := th - 1
	if lowNum < 0 {
		lowNum = 0
	}
	highNum := th + 1
	if highNum > BucketCount {
		highNum = BucketCount
	}
	if th == 0 {
		lowNum, highNum = 0, 0
	}
	if th == BucketCount {
		lowNum, highNum = BucketCount, BucketCount
	}
	combinedNum := truncNum + 10000
	return StrategyErrorBound{
		StrategyID:                       st.StrategyID,
		SourceIDs:                        append([]string(nil), st.SourceIDs...),
		RatioBP:                          st.RatioBP,
		QuantizedThreshold:               th,
		BucketCount:                      BucketCount,
		Quantization:                     QuantizationMode,
		RatioTruncationErrorNumerator:    truncNum,
		RatioErrorDenominator:            int64(BucketCount) * 10000,
		HashBucketGranularityNumerator:   1,
		HashBucketGranularityDenominator: BucketCount,
		CombinedErrorNumerator:           combinedNum,
		CombinedErrorDenominator:         int64(BucketCount) * 10000,
		EnvelopeLowerNumerator:           int64(lowNum),
		EnvelopeUpperNumerator:           int64(highNum),
		EnvelopeDenominator:              BucketCount,
		Basis:                            "effective p lies in [max(T-1,0)/10001, min(T+1,10001)/10001]; this envelope includes one bucket for floor-ratio truncation and one bucket for stable-hash granularity",
	}
}

func EvaluateTopKGuarantee(applicable bool, reason string, aggregates []KeyAggregate, ranking []RankItem, unseen int64, k int) (bool, string) {
	if !applicable {
		return false, "estimate is not applicable: " + reason
	}
	if len(aggregates) == 0 {
		return false, "no sampled client keys are available"
	}
	if len(ranking) < k && unseen > 0 {
		return false, fmt.Sprintf("only %d sampled keys are available for K=%d and %d events are unseen", len(ranking), k, unseen)
	}
	if len(ranking) < k {
		return true, "all events are sampled and all distinct keys are observed, so the ranking is exact"
	}
	var maxNonCandidate int64
	inCandidate := make(map[string]bool, len(ranking))
	for _, r := range ranking {
		inCandidate[r.ClientKey] = true
	}
	for _, a := range aggregates {
		if !inCandidate[a.ClientKey] && a.SampleCount > maxNonCandidate {
			maxNonCandidate = a.SampleCount
		}
	}
	competitorUpper := maxNonCandidate + unseen
	kthLower := ranking[len(ranking)-1].ObservedCountLower
	if kthLower > competitorUpper {
		return true, fmt.Sprintf("K-th sampled count lower bound %d is strictly greater than every competitor upper bound %d (max non-candidate sampled count %d + unseen events %d)", kthLower, competitorUpper, maxNonCandidate, unseen)
	}
	return false, fmt.Sprintf("K-th sampled count lower bound %d does not exceed competitor upper bound %d; the sample cannot certify top-K inclusion", kthLower, competitorUpper)
}

func mulDivFloor(a, b, c int64) (int64, error) {
	if c <= 0 {
		return 0, NewError(CodeSpec, "/error_bound", "division by non-positive denominator")
	}
	n := new(big.Int).Mul(big.NewInt(a), big.NewInt(b))
	n.Quo(n, big.NewInt(c))
	if !n.IsInt64() {
		return 0, NewError(CodeSpec, "/error_bound", "scaled estimate overflows int64")
	}
	return n.Int64(), nil
}

func mulDivCeil(a, b, c int64) (int64, error) {
	if c <= 0 {
		return 0, NewError(CodeSpec, "/error_bound", "division by non-positive denominator")
	}
	n := new(big.Int).Mul(big.NewInt(a), big.NewInt(b))
	den := big.NewInt(c)
	q, r := new(big.Int).QuoRem(n, den, new(big.Int))
	if r.Sign() > 0 {
		q.Add(q, big.NewInt(1))
	}
	if !q.IsInt64() {
		return 0, NewError(CodeSpec, "/error_bound", "scaled estimate overflows int64")
	}
	return q.Int64(), nil
}
