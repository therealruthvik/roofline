// Package hardware holds GPU specs used for roofline analysis.
package hardware

import (
	"fmt"
	"sort"
)

// GPU describes peak compute/memory specs for roofline math.
type GPU struct {
	Name          string
	PeakFLOPS     float64 // FP16 peak, in FLOPS/sec
	PeakBandwidth float64 // bytes/sec
	VRAMBytes     float64
}

// MachineBalance is peak_FLOPS / peak_mem_bandwidth (FLOPs/byte).
func (g GPU) MachineBalance() float64 {
	return g.PeakFLOPS / g.PeakBandwidth
}

const (
	tflops = 1e12
	gbps   = 1e9
	gb     = 1e9
)

// gpus is the built-in preset table. Extend via config/flags as needed.
var gpus = map[string]GPU{
	"A10G": {
		Name:          "A10G",
		PeakFLOPS:     125 * tflops,
		PeakBandwidth: 600 * gbps,
		VRAMBytes:     24 * gb,
	},
	"T4": {
		Name:          "T4",
		PeakFLOPS:     65 * tflops,
		PeakBandwidth: 320 * gbps,
		VRAMBytes:     16 * gb,
	},
	"A100-40GB": {
		Name:          "A100-40GB",
		PeakFLOPS:     312 * tflops,
		PeakBandwidth: 1555 * gbps,
		VRAMBytes:     40 * gb,
	},
	"A100-80GB": {
		Name:          "A100-80GB",
		PeakFLOPS:     312 * tflops,
		PeakBandwidth: 2039 * gbps,
		VRAMBytes:     80 * gb,
	},
	"H100-SXM": {
		Name:          "H100-SXM",
		PeakFLOPS:     990 * tflops,
		PeakBandwidth: 3350 * gbps,
		VRAMBytes:     80 * gb,
	},
}

// Lookup finds a GPU preset by name (case-sensitive key match).
func Lookup(name string) (GPU, error) {
	g, ok := gpus[name]
	if !ok {
		return GPU{}, fmt.Errorf("unknown GPU %q, valid options: %s", name, validNames())
	}
	return g, nil
}

func validNames() string {
	names := make([]string, 0, len(gpus))
	for n := range gpus {
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
