// Package validate loads real benchmark measurements, joins them against
// roofline predictions, and computes prediction error.
package validate

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"strconv"
)

// BreakdownThresholdPct is the error magnitude above which a row is
// flagged as a "model breakdown" case.
const BreakdownThresholdPct = 25.0

// Actual is one real measured benchmark data point.
type Actual struct {
	GPU                  string  `json:"gpu"`
	Model                string  `json:"model"`
	BatchSize            int     `json:"batch_size"`
	MeasuredTokensPerSec float64 `json:"measured_tokens_per_sec"`
}

// Predicted is one row of a sweep prediction CSV.
type Predicted struct {
	BatchSize             int
	ArithmeticIntensity   float64
	BoundType             string
	PredictedTokensPerSec float64
	PredictedLatencyMs    float64
}

// Joined is one predicted/actual pair with computed error.
type Joined struct {
	BatchSize             int
	ArithmeticIntensity   float64
	BoundType             string
	PredictedTokensPerSec float64
	MeasuredTokensPerSec  float64
	PercentError          float64 // signed: (predicted-measured)/measured * 100
	ModelBreakdown        bool
}

// LoadActual reads real benchmark results from a JSON file. It accepts
// either a top-level array of records, or an object with a "results" or
// "benchmarks" key holding that array.
func LoadActual(path string) ([]Actual, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("opening actual benchmark file: %w", err)
	}
	defer f.Close()

	data, err := io.ReadAll(f)
	if err != nil {
		return nil, fmt.Errorf("reading actual benchmark file: %w", err)
	}

	var direct []Actual
	if err := json.Unmarshal(data, &direct); err == nil {
		return direct, nil
	}

	var wrapped struct {
		Results    []Actual `json:"results"`
		Benchmarks []Actual `json:"benchmarks"`
	}
	if err := json.Unmarshal(data, &wrapped); err != nil {
		return nil, fmt.Errorf("parsing actual benchmark JSON (expected array or {results:[...]}/{benchmarks:[...]}): %w", err)
	}
	if len(wrapped.Results) > 0 {
		return wrapped.Results, nil
	}
	return wrapped.Benchmarks, nil
}

// LoadPredictions reads a sweep CSV (as written by `roofline sweep`).
func LoadPredictions(path string) ([]Predicted, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("opening predictions file: %w", err)
	}
	defer f.Close()

	r := csv.NewReader(f)
	records, err := r.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("parsing predictions CSV: %w", err)
	}
	if len(records) < 1 {
		return nil, fmt.Errorf("predictions CSV is empty")
	}

	rows := make([]Predicted, 0, len(records)-1)
	for _, rec := range records[1:] {
		if len(rec) < 5 {
			continue
		}
		batch, err := strconv.Atoi(rec[0])
		if err != nil {
			return nil, fmt.Errorf("invalid batch_size %q: %w", rec[0], err)
		}
		ai, err := strconv.ParseFloat(rec[1], 64)
		if err != nil {
			return nil, fmt.Errorf("invalid arithmetic_intensity %q: %w", rec[1], err)
		}
		tps, err := strconv.ParseFloat(rec[3], 64)
		if err != nil {
			return nil, fmt.Errorf("invalid predicted_tokens_per_sec %q: %w", rec[3], err)
		}
		lat, err := strconv.ParseFloat(rec[4], 64)
		if err != nil {
			return nil, fmt.Errorf("invalid predicted_latency_ms %q: %w", rec[4], err)
		}
		rows = append(rows, Predicted{
			BatchSize:             batch,
			ArithmeticIntensity:   ai,
			BoundType:             rec[2],
			PredictedTokensPerSec: tps,
			PredictedLatencyMs:    lat,
		})
	}
	return rows, nil
}

// Join matches predicted rows (already scoped to one gpu+model sweep)
// against actual measurements filtered to the same gpu+model, keyed on
// batch_size, and computes percent error per row.
func Join(gpu, model string, predicted []Predicted, actual []Actual) []Joined {
	actualByBatch := make(map[int]Actual)
	for _, a := range actual {
		if a.GPU == gpu && a.Model == model {
			actualByBatch[a.BatchSize] = a
		}
	}

	joined := make([]Joined, 0, len(predicted))
	for _, p := range predicted {
		a, ok := actualByBatch[p.BatchSize]
		if !ok {
			continue
		}
		pctErr := (p.PredictedTokensPerSec - a.MeasuredTokensPerSec) / a.MeasuredTokensPerSec * 100
		joined = append(joined, Joined{
			BatchSize:             p.BatchSize,
			ArithmeticIntensity:   p.ArithmeticIntensity,
			BoundType:             p.BoundType,
			PredictedTokensPerSec: p.PredictedTokensPerSec,
			MeasuredTokensPerSec:  a.MeasuredTokensPerSec,
			PercentError:          pctErr,
			ModelBreakdown:        math.Abs(pctErr) > BreakdownThresholdPct,
		})
	}

	sort.Slice(joined, func(i, j int) bool {
		return math.Abs(joined[i].PercentError) > math.Abs(joined[j].PercentError)
	})
	return joined
}
