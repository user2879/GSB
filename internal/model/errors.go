package model

import (
	"encoding/json"
	"errors"
	"fmt"
)

const (
	CodeSpec   = "E_SPEC"
	CodeReplay = "E_REPLAY"
	CodeVerify = "E_VERIFY"
	CodeLedger = "E_LEDGER"
	CodeIO     = "E_IO"
	CodeCrash  = "E_CRASH"
)

// CodedError is the stable machine-readable error envelope used by both CLIs.
type CodedError struct {
	Code    string         `json:"code"`
	Message string         `json:"message"`
	Path    string         `json:"path,omitempty"`
	Details map[string]any `json:"details,omitempty"`
}

func (e *CodedError) Error() string {
	if e.Path == "" {
		return fmt.Sprintf("%s: %s", e.Code, e.Message)
	}
	return fmt.Sprintf("%s at %s: %s", e.Code, e.Path, e.Message)
}

func NewError(code, path, message string) *CodedError {
	return &CodedError{Code: code, Path: path, Message: message}
}

func WithDetail(err error, key string, value any) error {
	var ce *CodedError
	if errors.As(err, &ce) {
		if ce.Details == nil {
			ce.Details = make(map[string]any)
		}
		ce.Details[key] = value
		return ce
	}
	return err
}

func ErrorJSON(err error) []byte {
	var ce *CodedError
	if !errors.As(err, &ce) {
		ce = &CodedError{Code: CodeIO, Message: err.Error()}
	}
	b, _ := json.MarshalIndent(ce, "", "  ")
	return b
}
