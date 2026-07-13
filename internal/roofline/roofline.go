// Package roofline computes arithmetic intensity, bound classification,
// and throughput/latency predictions from FLOPs/bytes-per-token figures.
package roofline

import (
	"roofline/internal/hardware"
	"roofline/internal/model"
)

// BoundType classifies whether a workload is compute- or memory-bound
// under the roofline model.
type BoundType string

const (
	ComputeBound BoundType = "compute-bound"
	MemoryBound  BoundType = "memory-bound"
)

// Result holds the full roofline prediction for one configuration.
type Result struct {
	ArithmeticIntensity   float64 // FLOPs/byte
	MachineBalance        float64 // FLOPs/byte
	Bound                 BoundType
	PredictedTokensPerSec float64
	PredictedLatencyMs    float64
}

// ArithmeticIntensity is FLOPs/token divided by bytes/token.
func ArithmeticIntensity(flopsPerToken, bytesPerToken float64) float64 {
	return flopsPerToken / bytesPerToken
}

// MachineBalance is peak FLOPS divided by peak memory bandwidth (bytes/sec).
func MachineBalance(peakFLOPS, peakBandwidth float64) float64 {
	return peakFLOPS / peakBandwidth
}

// Classify returns MemoryBound when AI is below machine balance, else
// ComputeBound (ties go to compute-bound, matching AI >= balance).
func Classify(ai, machineBalance float64) BoundType {
	if ai < machineBalance {
		return MemoryBound
	}
	return ComputeBound
}

// Predict runs the full roofline calculation for one configuration.
func Predict(flopsPerToken, bytesPerToken, peakFLOPS, peakBandwidth float64) Result {
	ai := ArithmeticIntensity(flopsPerToken, bytesPerToken)
	balance := MachineBalance(peakFLOPS, peakBandwidth)
	bound := Classify(ai, balance)

	var throughput float64
	if bound == MemoryBound {
		throughput = peakBandwidth / bytesPerToken
	} else {
		throughput = peakFLOPS / flopsPerToken
	}

	return Result{
		ArithmeticIntensity:   ai,
		MachineBalance:        balance,
		Bound:                 bound,
		PredictedTokensPerSec: throughput,
		PredictedLatencyMs:    1000 / throughput,
	}
}

// Compute ties together a GPU spec and model config to produce a roofline
// Result for the given precision, batch size, and sequence length.
func Compute(gpu hardware.GPU, cfg model.Config, precision model.Precision, batch, seqLen int) (Result, error) {
	flopsPerToken := cfg.FLOPsPerToken()
	bytesPerToken, err := cfg.BytesPerToken(precision, batch, seqLen)
	if err != nil {
		return Result{}, err
	}
	return Predict(flopsPerToken, bytesPerToken, gpu.PeakFLOPS, gpu.PeakBandwidth), nil
}
