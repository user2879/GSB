package model

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sort"
	"strconv"
	"strings"
)

const (
	BucketCount         = 10001
	QuantizationMode    = "floor"
	MaxClientKeyLength  = 128
	MaxSourceIDLength   = 64
	MaxStrategyIDLength = 64
	MaxSaltLength       = 128
)

type Source struct {
	SourceID         string   `json:"source_id"`
	Seed             int64    `json:"seed"`
	EventCount       int64    `json:"event_count"`
	StartTimestampMS int64    `json:"start_timestamp_ms"`
	StepMS           int64    `json:"step_ms"`
	Keys             []string `json:"keys"`
	WeightMin        int64    `json:"weight_min"`
	WeightMax        int64    `json:"weight_max"`
}

type Spec struct {
	SpecVersion int64    `json:"spec_version"`
	Sources     []Source `json:"sources"`
}

type Strategy struct {
	StrategyID string   `json:"strategy_id"`
	SourceIDs  []string `json:"source_ids"`
	RatioBP    int      `json:"ratio_bp"`
}

type Policy struct {
	PolicyVersion int64      `json:"policy_version"`
	Salt          string     `json:"salt"`
	Strategies    []Strategy `json:"strategies"`
}

type ResolvedStrategy struct {
	StrategyID          string   `json:"strategy_id"`
	SourceIDs           []string `json:"source_ids"`
	RatioBP             int      `json:"ratio_bp"`
	BucketCount         int      `json:"bucket_count"`
	QuantizedThreshold  int      `json:"quantized_threshold"`
	Quantization        string   `json:"quantization"`
	TruncationNumerator int64    `json:"truncation_error_numerator"`
	ErrorDenominator    int64    `json:"error_denominator"`
}

type Loaded struct {
	Spec             Spec
	Policy           Policy
	SpecDigest       string
	PolicyDigest     string
	ManifestDigest   string
	SourceByID       map[string]Source
	StrategyByID     map[string]ResolvedStrategy
	StrategyBySource map[string]ResolvedStrategy
}

func LoadFiles(specPath, policyPath string) (*Loaded, error) {
	specBytes, err := os.ReadFile(specPath)
	if err != nil {
		return nil, &CodedError{Code: CodeIO, Path: specPath, Message: err.Error()}
	}
	policyBytes, err := os.ReadFile(policyPath)
	if err != nil {
		return nil, &CodedError{Code: CodeIO, Path: policyPath, Message: err.Error()}
	}
	return LoadBytes(specBytes, policyBytes)
}

func LoadBytes(specBytes, policyBytes []byte) (*Loaded, error) {
	spec, err := parseSpec(specBytes)
	if err != nil {
		return nil, err
	}
	policy, err := parsePolicy(policyBytes)
	if err != nil {
		return nil, err
	}
	if err := validateCross(spec, policy); err != nil {
		return nil, err
	}
	spec = normalizeSpec(spec)
	policy = normalizePolicy(policy)
	specDigest, err := DigestJSON(spec)
	if err != nil {
		return nil, &CodedError{Code: CodeSpec, Path: "/", Message: err.Error()}
	}
	policyDigest, err := DigestJSON(policy)
	if err != nil {
		return nil, &CodedError{Code: CodeSpec, Path: "/", Message: err.Error()}
	}
	manifestDigest := DigestFields(specDigest, policyDigest)
	sourceByID := make(map[string]Source, len(spec.Sources))
	for _, s := range spec.Sources {
		sourceByID[s.SourceID] = s
	}
	strategyByID := make(map[string]ResolvedStrategy, len(policy.Strategies))
	strategyBySource := make(map[string]ResolvedStrategy)
	for _, st := range policy.Strategies {
		th := QuantizedThreshold(st.RatioBP)
		rs := ResolvedStrategy{
			StrategyID:          st.StrategyID,
			SourceIDs:           append([]string(nil), st.SourceIDs...),
			RatioBP:             st.RatioBP,
			BucketCount:         BucketCount,
			QuantizedThreshold:  th,
			Quantization:        QuantizationMode,
			TruncationNumerator: int64(st.RatioBP)*BucketCount - int64(th)*10000,
			ErrorDenominator:    int64(BucketCount) * 10000,
		}
		strategyByID[st.StrategyID] = rs
		for _, sid := range st.SourceIDs {
			strategyBySource[sid] = rs
		}
	}
	return &Loaded{
		Spec:             spec,
		Policy:           policy,
		SpecDigest:       specDigest,
		PolicyDigest:     policyDigest,
		ManifestDigest:   manifestDigest,
		SourceByID:       sourceByID,
		StrategyByID:     strategyByID,
		StrategyBySource: strategyBySource,
	}, nil
}

func QuantizedThreshold(ratioBP int) int {
	return int((int64(ratioBP) * int64(BucketCount)) / 10000)
}

func normalizeSpec(spec Spec) Spec {
	for i := range spec.Sources {
		cp := append([]string(nil), spec.Sources[i].Keys...)
		spec.Sources[i].Keys = cp
	}
	sort.Slice(spec.Sources, func(i, j int) bool { return spec.Sources[i].SourceID < spec.Sources[j].SourceID })
	return spec
}

func normalizePolicy(policy Policy) Policy {
	for i := range policy.Strategies {
		cp := append([]string(nil), policy.Strategies[i].SourceIDs...)
		sort.Strings(cp)
		policy.Strategies[i].SourceIDs = cp
	}
	sort.Slice(policy.Strategies, func(i, j int) bool { return policy.Strategies[i].StrategyID < policy.Strategies[j].StrategyID })
	return policy
}

func parseSpec(b []byte) (Spec, error) {
	root, err := decodeObject(b, "/spec")
	if err != nil {
		return Spec{}, err
	}
	if err := onlyKeys(root, "/spec", "spec_version", "sources"); err != nil {
		return Spec{}, err
	}
	version, err := getInt(root, "spec_version", "/spec_version")
	if err != nil {
		return Spec{}, err
	}
	if version <= 0 {
		return Spec{}, NewError(CodeSpec, "/spec_version", "must be a positive integer")
	}
	rawSources, err := getArray(root, "sources", "/sources")
	if err != nil {
		return Spec{}, err
	}
	if len(rawSources) == 0 {
		return Spec{}, NewError(CodeSpec, "/sources", "must contain at least one source")
	}
	seenSources := make(map[string]int)
	spec := Spec{SpecVersion: version}
	for i, raw := range rawSources {
		base := fmt.Sprintf("/sources/%d", i)
		obj, err := rawObject(raw, base)
		if err != nil {
			return Spec{}, err
		}
		if err := onlyKeys(obj, base, "source_id", "seed", "event_count", "start_timestamp_ms", "step_ms", "keys", "weight_min", "weight_max"); err != nil {
			return Spec{}, err
		}
		sourceID, err := getString(obj, "source_id", base+"/source_id")
		if err != nil {
			return Spec{}, err
		}
		if err := validateToken(sourceID, MaxSourceIDLength, true); err != nil {
			return Spec{}, NewError(CodeSpec, base+"/source_id", err.Error())
		}
		if prev, ok := seenSources[sourceID]; ok {
			return Spec{}, NewError(CodeSpec, base+"/source_id", fmt.Sprintf("duplicate source_id; first declared at /sources/%d/source_id", prev))
		}
		seenSources[sourceID] = i
		seed, err := getInt(obj, "seed", base+"/seed")
		if err != nil {
			return Spec{}, err
		}
		if seed <= 0 {
			return Spec{}, NewError(CodeSpec, base+"/seed", "must be a positive integer")
		}
		count, err := getInt(obj, "event_count", base+"/event_count")
		if err != nil {
			return Spec{}, err
		}
		if count <= 0 {
			return Spec{}, NewError(CodeSpec, base+"/event_count", "must be a positive integer")
		}
		start, err := getInt(obj, "start_timestamp_ms", base+"/start_timestamp_ms")
		if err != nil {
			return Spec{}, err
		}
		if start < 0 {
			return Spec{}, NewError(CodeSpec, base+"/start_timestamp_ms", "must be a non-negative integer millisecond timestamp")
		}
		step, err := getInt(obj, "step_ms", base+"/step_ms")
		if err != nil {
			return Spec{}, err
		}
		if step <= 0 {
			return Spec{}, NewError(CodeSpec, base+"/step_ms", "must be a positive integer")
		}
		if count-1 > (int64(^uint64(0)>>1)-start)/step {
			return Spec{}, NewError(CodeSpec, base+"/event_count", "timestamp range overflows int64")
		}
		rawKeys, err := getArray(obj, "keys", base+"/keys")
		if err != nil {
			return Spec{}, err
		}
		if len(rawKeys) == 0 {
			return Spec{}, NewError(CodeSpec, base+"/keys", "must contain at least one client key")
		}
		keys := make([]string, len(rawKeys))
		for j, rk := range rawKeys {
			key, err := rawString(rk, fmt.Sprintf("%s/keys/%d", base, j))
			if err != nil {
				return Spec{}, err
			}
			if err := validateToken(key, MaxClientKeyLength, false); err != nil {
				return Spec{}, NewError(CodeSpec, fmt.Sprintf("%s/keys/%d", base, j), err.Error())
			}
			keys[j] = key
		}
		wmin, err := getInt(obj, "weight_min", base+"/weight_min")
		if err != nil {
			return Spec{}, err
		}
		wmax, err := getInt(obj, "weight_max", base+"/weight_max")
		if err != nil {
			return Spec{}, err
		}
		if wmin <= 0 {
			return Spec{}, NewError(CodeSpec, base+"/weight_min", "must be a positive integer")
		}
		if wmax < wmin {
			return Spec{}, NewError(CodeSpec, base+"/weight_max", "must be greater than or equal to weight_min")
		}
		spec.Sources = append(spec.Sources, Source{
			SourceID: sourceID, Seed: seed, EventCount: count,
			StartTimestampMS: start, StepMS: step, Keys: keys,
			WeightMin: wmin, WeightMax: wmax,
		})
	}
	return spec, nil
}

func parsePolicy(b []byte) (Policy, error) {
	root, err := decodeObject(b, "/policy")
	if err != nil {
		return Policy{}, err
	}
	if err := onlyKeys(root, "/policy", "policy_version", "salt", "strategies"); err != nil {
		return Policy{}, err
	}
	version, err := getInt(root, "policy_version", "/policy_version")
	if err != nil {
		return Policy{}, err
	}
	if version <= 0 {
		return Policy{}, NewError(CodeSpec, "/policy_version", "must be a positive integer")
	}
	salt, err := getString(root, "salt", "/salt")
	if err != nil {
		return Policy{}, err
	}
	if err := validateToken(salt, MaxSaltLength, true); err != nil {
		return Policy{}, NewError(CodeSpec, "/salt", err.Error())
	}
	rawStrategies, err := getArray(root, "strategies", "/strategies")
	if err != nil {
		return Policy{}, err
	}
	if len(rawStrategies) == 0 {
		return Policy{}, NewError(CodeSpec, "/strategies", "must contain at least one sampling strategy")
	}
	// First parse all IDs so duplicate coverage can be reported against the
	// first declaration, even if an invalid source is found later.
	seenStrategy := make(map[string]int)
	seenSource := make(map[string]string)
	policy := Policy{PolicyVersion: version, Salt: salt}
	for i, raw := range rawStrategies {
		base := fmt.Sprintf("/strategies/%d", i)
		obj, err := rawObject(raw, base)
		if err != nil {
			return Policy{}, err
		}
		if err := onlyKeys(obj, base, "strategy_id", "source_ids", "ratio_bp"); err != nil {
			return Policy{}, err
		}
		strategyID, err := getString(obj, "strategy_id", base+"/strategy_id")
		if err != nil {
			return Policy{}, err
		}
		if err := validateToken(strategyID, MaxStrategyIDLength, true); err != nil {
			return Policy{}, NewError(CodeSpec, base+"/strategy_id", err.Error())
		}
		if prev, ok := seenStrategy[strategyID]; ok {
			return Policy{}, NewError(CodeSpec, base+"/strategy_id", fmt.Sprintf("duplicate strategy_id; first declared at /strategies/%d/strategy_id", prev))
		}
		seenStrategy[strategyID] = i
		ratioRaw, ok := obj["ratio_bp"]
		if !ok {
			return Policy{}, NewError(CodeSpec, base+"/ratio_bp", "field is required")
		}
		ratio64, err := rawInt(ratioRaw, base+"/ratio_bp")
		if err != nil {
			return Policy{}, err
		}
		if ratio64 < 0 || ratio64 > 10000 {
			return Policy{}, NewError(CodeSpec, base+"/ratio_bp", "must be in the inclusive range 0..10000")
		}
		rawIDs, err := getArray(obj, "source_ids", base+"/source_ids")
		if err != nil {
			return Policy{}, err
		}
		if len(rawIDs) == 0 {
			return Policy{}, NewError(CodeSpec, base+"/source_ids", "must contain at least one source_id")
		}
		ids := make([]string, 0, len(rawIDs))
		local := make(map[string]bool)
		for j, rid := range rawIDs {
			idPath := fmt.Sprintf("%s/source_ids/%d", base, j)
			id, err := rawString(rid, idPath)
			if err != nil {
				return Policy{}, err
			}
			if err := validateToken(id, MaxSourceIDLength, true); err != nil {
				return Policy{}, NewError(CodeSpec, idPath, err.Error())
			}
			if local[id] {
				return Policy{}, NewError(CodeSpec, idPath, "duplicate source coverage in strategy")
			}
			if prev, ok := seenSource[id]; ok {
				return Policy{}, NewError(CodeSpec, idPath, fmt.Sprintf("source already covered by strategy %q", prev))
			}
			local[id] = true
			seenSource[id] = strategyID
			ids = append(ids, id)
		}
		policy.Strategies = append(policy.Strategies, Strategy{StrategyID: strategyID, SourceIDs: ids, RatioBP: int(ratio64)})
	}
	return policy, nil
}

func validateCross(spec Spec, policy Policy) error {
	known := make(map[string]bool, len(spec.Sources))
	for _, s := range spec.Sources {
		known[s.SourceID] = true
	}
	for i, st := range policy.Strategies {
		for j, sid := range st.SourceIDs {
			if !known[sid] {
				return NewError(CodeSpec, fmt.Sprintf("/strategies/%d/source_ids/%d", i, j), fmt.Sprintf("unknown source_id %q", sid))
			}
		}
	}
	for _, s := range spec.Sources {
		if !contains(policyCoverage(policy), s.SourceID) {
			return NewError(CodeSpec, fmt.Sprintf("/strategies"), fmt.Sprintf("source %q is not covered by any sampling strategy", s.SourceID))
		}
	}
	return nil
}

func policyCoverage(policy Policy) map[string]bool {
	m := make(map[string]bool)
	for _, st := range policy.Strategies {
		for _, sid := range st.SourceIDs {
			m[sid] = true
		}
	}
	return m
}

func contains(m map[string]bool, key string) bool { return m[key] }

func validateToken(s string, max int, allowLeadingDash bool) error {
	if s == "" {
		return fmt.Errorf("must not be empty")
	}
	if len(s) > max {
		return fmt.Errorf("must be at most %d bytes", max)
	}
	for i, r := range s {
		ok := (r >= 'A' && r <= 'Z') || (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') || r == '.' || r == '_' || r == ':' || r == '-'
		if !ok {
			return fmt.Errorf("contains unsupported character %q at byte %d; supported: A-Z a-z 0-9 . _ : -", r, i)
		}
		if i == 0 && r == '-' && !allowLeadingDash {
			return fmt.Errorf("must not start with '-'")
		}
	}
	return nil
}

func decodeObject(b []byte, path string) (map[string]json.RawMessage, error) {
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.UseNumber()
	var v any
	if err := dec.Decode(&v); err != nil {
		return nil, NewError(CodeSpec, path, "invalid JSON: "+err.Error())
	}
	var trailing any
	if err := dec.Decode(&trailing); err == nil {
		return nil, NewError(CodeSpec, path, "unexpected trailing JSON value")
	} else if err != io.EOF {
		return nil, NewError(CodeSpec, path, "invalid trailing JSON: "+err.Error())
	}
	obj, ok := v.(map[string]any)
	if !ok {
		return nil, NewError(CodeSpec, path, "must be a JSON object")
	}
	_ = obj
	raw, err := json.Marshal(v)
	if err != nil {
		return nil, NewError(CodeSpec, path, err.Error())
	}
	var out map[string]json.RawMessage
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, NewError(CodeSpec, path, err.Error())
	}
	return out, nil
}

func onlyKeys(obj map[string]json.RawMessage, base string, allowed ...string) error {
	ok := make(map[string]bool, len(allowed))
	for _, k := range allowed {
		ok[k] = true
	}
	var unknown []string
	for k := range obj {
		if !ok[k] {
			unknown = append(unknown, k)
		}
	}
	if len(unknown) > 0 {
		sort.Strings(unknown)
		return NewError(CodeSpec, base+"/"+unknown[0], "unknown field")
	}
	return nil
}

func getString(obj map[string]json.RawMessage, key, path string) (string, error) {
	raw, ok := obj[key]
	if !ok {
		return "", NewError(CodeSpec, path, "field is required")
	}
	return rawString(raw, path)
}

func getInt(obj map[string]json.RawMessage, key, path string) (int64, error) {
	raw, ok := obj[key]
	if !ok {
		return 0, NewError(CodeSpec, path, "field is required")
	}
	return rawInt(raw, path)
}

func getArray(obj map[string]json.RawMessage, key, path string) ([]json.RawMessage, error) {
	raw, ok := obj[key]
	if !ok {
		return nil, NewError(CodeSpec, path, "field is required")
	}
	var arr []json.RawMessage
	if err := json.Unmarshal(raw, &arr); err != nil || arr == nil {
		return nil, NewError(CodeSpec, path, "must be a JSON array")
	}
	return arr, nil
}

func rawObject(raw json.RawMessage, path string) (map[string]json.RawMessage, error) {
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(raw, &obj); err != nil || obj == nil {
		return nil, NewError(CodeSpec, path, "must be a JSON object")
	}
	return obj, nil
}

func rawString(raw json.RawMessage, path string) (string, error) {
	var s string
	if err := json.Unmarshal(raw, &s); err != nil {
		return "", NewError(CodeSpec, path, "must be a JSON string")
	}
	return s, nil
}

func rawInt(raw json.RawMessage, path string) (int64, error) {
	s := strings.TrimSpace(string(raw))
	if s == "" || strings.ContainsAny(s, ".eE") {
		return 0, NewError(CodeSpec, path, "must be a JSON integer")
	}
	v, err := strconv.ParseInt(s, 10, 64)
	if err != nil {
		return 0, NewError(CodeSpec, path, "must be an integer in int64 range")
	}
	return v, nil
}
