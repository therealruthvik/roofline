// Package model holds LLM architecture presets and the per-token
// FLOPs/bytes math used for decode-step roofline analysis.
package model

import (
	"fmt"
	"sort"
)

// Config describes the architecture parameters needed for decode-step
// FLOPs and KV-cache byte estimates.
type Config struct {
	Name       string
	NumLayers  int
	HiddenDim  int
	NumParams  float64 // total parameter count
}

var configs = map[string]Config{
	"mistral-7b": {
		Name:      "mistral-7b",
		NumLayers: 32,
		HiddenDim: 4096,
		NumParams: 7.24e9,
	},
	"llama-3.1-8b": {
		Name:      "llama-3.1-8b",
		NumLayers: 32,
		HiddenDim: 4096,
		NumParams: 8.03e9,
	},
}

// Lookup finds a model preset by name.
func Lookup(name string) (Config, error) {
	c, ok := configs[name]
	if !ok {
		return Config{}, fmt.Errorf("unknown model %q, valid options: %s", name, validNames())
	}
	return c, nil
}

func validNames() string {
	names := make([]string, 0, len(configs))
	for n := range configs {
		names = append(names, n)
	}
	sort.Strings(names)
	out := ""
	for i, n := range names {
		if i > 0 {
			out += ", "
		}
		out += n
	}
	return out
}

// Precision is a weight/activation quantization scheme.
type Precision string

const (
	FP16 Precision = "fp16"
	BF16 Precision = "bf16"
	INT8 Precision = "int8"
	INT4 Precision = "int4"
)

// BytesPerParam returns weight storage bytes per parameter for the precision.
func BytesPerParam(p Precision) (float64, error) {
	switch p {
	case FP16, BF16:
		return 2, nil
	case INT8:
		return 1, nil
	case INT4:
		return 0.5, nil
	default:
		return 0, fmt.Errorf("unknown precision %q, valid options: fp16, bf16, int8, int4", p)
	}
}

// FLOPsPerToken is the decode-step FLOPs approximation: 2 * num_params.
// Batch size does not change per-token FLOPs (it scales total FLOPs
// linearly, but throughput here is expressed per-token per-sequence).
func (c Config) FLOPsPerToken() float64 {
	return 2 * c.NumParams
}

// KVCacheBytesPerToken is the read traffic added per generated token by
// the KV cache: seq_len * batch * num_layers * hidden_dim * 2 (K and V) *
// bytes_per_element. Here we compute the per-token-position contribution;
// callers multiply by seq_len for the full read at a given position.
func (c Config) KVCacheBytesPerToken(batch, seqLen int, bytesPerElem float64) float64 {
	return float64(batch) * float64(c.NumLayers) * float64(c.HiddenDim) * 2 * bytesPerElem * float64(seqLen)
}

// BytesPerToken is total memory traffic per generated token: weight bytes
// read once per token (batch-independent, weights are shared/broadcast)
// plus the KV-cache read for the current sequence length and batch size.
func (c Config) BytesPerToken(precision Precision, batch, seqLen int) (float64, error) {
	bpp, err := BytesPerParam(precision)
	if err != nil {
		return 0, err
	}
	weightBytes := c.NumParams * bpp
	kvBytes := c.KVCacheBytesPerToken(batch, seqLen, bpp)
	return weightBytes + kvBytes, nil
}
