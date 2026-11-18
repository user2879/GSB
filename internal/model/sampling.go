package model

import (
	"crypto/sha256"
	"encoding/binary"
	"strconv"
)

const SamplingDomain = "flowledger/head-sample/v1"

type SamplingDecision struct {
	StrategyID    string `json:"strategy_id"`
	PolicyVersion int64  `json:"policy_version"`
	Bucket        int    `json:"bucket"`
	Threshold     int    `json:"threshold"`
	Sampled       bool   `json:"sampled"`
}

func StableBucket(policyVersion int64, salt, strategyID, clientKey string) int {
	h := sha256.New()
	for _, part := range []string{
		SamplingDomain,
		strconv.FormatInt(policyVersion, 10),
		salt,
		strategyID,
		clientKey,
	} {
		var n [8]byte
		binary.BigEndian.PutUint64(n[:], uint64(len(part)))
		_, _ = h.Write(n[:])
		_, _ = h.Write([]byte(part))
	}
	sum := h.Sum(nil)
	return int(binary.BigEndian.Uint64(sum[:8]) % uint64(BucketCount))
}

func Decide(loaded *Loaded, sourceID, clientKey string) (SamplingDecision, error) {
	st, ok := loaded.StrategyBySource[sourceID]
	if !ok {
		return SamplingDecision{}, NewError(CodeSpec, "/strategies", "source has no strategy")
	}
	bucket := StableBucket(loaded.Policy.PolicyVersion, loaded.Policy.Salt, st.StrategyID, clientKey)
	return SamplingDecision{
		StrategyID:    st.StrategyID,
		PolicyVersion: loaded.Policy.PolicyVersion,
		Bucket:        bucket,
		Threshold:     st.QuantizedThreshold,
		Sampled:       bucket < st.QuantizedThreshold,
	}, nil
}
