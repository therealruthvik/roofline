package roofline

import (
	"math"
	"testing"
)

func almostEqual(a, b float64) bool {
	if b == 0 {
		return math.Abs(a) < 1e-9
	}
	return math.Abs(a-b)/math.Abs(b) < 1e-9
}

func TestArithmeticIntensity(t *testing.T) {
	cases := []struct {
		name          string
		flopsPerToken float64
		bytesPerToken float64
		want          float64
	}{
		{"simple", 100, 10, 10},
		{"fractional", 7.24e9 * 2, 7.24e9 * 2, 1},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := ArithmeticIntensity(c.flopsPerToken, c.bytesPerToken)
			if !almostEqual(got, c.want) {
				t.Errorf("ArithmeticIntensity(%v, %v) = %v, want %v", c.flopsPerToken, c.bytesPerToken, got, c.want)
			}
		})
	}
}

func TestMachineBalance(t *testing.T) {
	got := MachineBalance(125e12, 600e9)
	want := 125e12 / 600e9
	if !almostEqual(got, want) {
		t.Errorf("MachineBalance = %v, want %v", got, want)
	}
}

func TestClassify(t *testing.T) {
	cases := []struct {
		name           string
		ai             float64
		machineBalance float64
		want           BoundType
	}{
		{"below balance is memory-bound", 10, 200, MemoryBound},
		{"above balance is compute-bound", 500, 200, ComputeBound},
		{"exact tie goes compute-bound", 200, 200, ComputeBound},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := Classify(c.ai, c.machineBalance)
			if got != c.want {
				t.Errorf("Classify(%v, %v) = %v, want %v", c.ai, c.machineBalance, got, c.want)
			}
		})
	}
}

func TestPredict_MemoryBound(t *testing.T) {
	// Low AI (small model, huge bytes/token) on a fast-compute GPU should
	// land memory-bound, with throughput = bandwidth / bytes_per_token.
	flopsPerToken := 100.0
	bytesPerToken := 100.0
	peakFLOPS := 125e12
	peakBandwidth := 600e9

	r := Predict(flopsPerToken, bytesPerToken, peakFLOPS, peakBandwidth)

	if r.Bound != MemoryBound {
		t.Fatalf("expected MemoryBound, got %v (ai=%v, balance=%v)", r.Bound, r.ArithmeticIntensity, r.MachineBalance)
	}
	wantThroughput := peakBandwidth / bytesPerToken
	if !almostEqual(r.PredictedTokensPerSec, wantThroughput) {
		t.Errorf("throughput = %v, want %v", r.PredictedTokensPerSec, wantThroughput)
	}
	wantLatency := 1000 / wantThroughput
	if !almostEqual(r.PredictedLatencyMs, wantLatency) {
		t.Errorf("latency = %v, want %v", r.PredictedLatencyMs, wantLatency)
	}
}

func TestPredict_ComputeBound(t *testing.T) {
	// Very high FLOPs relative to bytes (high AI) forces compute-bound.
	flopsPerToken := 1e15
	bytesPerToken := 100.0
	peakFLOPS := 125e12
	peakBandwidth := 600e9

	r := Predict(flopsPerToken, bytesPerToken, peakFLOPS, peakBandwidth)

	if r.Bound != ComputeBound {
		t.Fatalf("expected ComputeBound, got %v (ai=%v, balance=%v)", r.Bound, r.ArithmeticIntensity, r.MachineBalance)
	}
	wantThroughput := peakFLOPS / flopsPerToken
	if !almostEqual(r.PredictedTokensPerSec, wantThroughput) {
		t.Errorf("throughput = %v, want %v", r.PredictedTokensPerSec, wantThroughput)
	}
}

func TestPredict_MistralA10GRealistic(t *testing.T) {
	// Mistral-7B fp16 on A10G at batch=1, short seqlen: should be
	// heavily memory-bound (typical for single-batch decode).
	flopsPerToken := 2 * 7.24e9
	weightBytes := 7.24e9 * 2
	kvBytes := 1.0 * 32 * 4096 * 2 * 2 * 128 // batch=1, seqlen=128
	bytesPerToken := weightBytes + kvBytes

	r := Predict(flopsPerToken, bytesPerToken, 125e12, 600e9)

	if r.Bound != MemoryBound {
		t.Fatalf("expected MemoryBound for batch=1 decode, got %v", r.Bound)
	}
	if r.PredictedTokensPerSec <= 0 {
		t.Errorf("expected positive throughput, got %v", r.PredictedTokensPerSec)
	}
}
