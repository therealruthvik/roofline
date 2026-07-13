// Package output formats roofline results as tables and CSV.
package output

import (
	"encoding/csv"
	"fmt"
	"io"
	"strconv"
	"text/tabwriter"
)

// Table writes rows as an aligned, padded table to w. header and each row
// must have the same column count.
func Table(w io.Writer, header []string, rows [][]string) error {
	tw := tabwriter.NewWriter(w, 0, 2, 2, ' ', 0)
	fmt.Fprintln(tw, joinTab(header))
	for _, r := range rows {
		fmt.Fprintln(tw, joinTab(r))
	}
	return tw.Flush()
}

func joinTab(cols []string) string {
	out := ""
	for i, c := range cols {
		if i > 0 {
			out += "\t"
		}
		out += c
	}
	return out
}

// SweepRow is one row of the sweep CSV output.
type SweepRow struct {
	BatchSize             int
	ArithmeticIntensity   float64
	BoundType             string
	PredictedTokensPerSec float64
	PredictedLatencyMs    float64
}

// WriteSweepCSV writes sweep rows to w with a fixed header.
func WriteSweepCSV(w io.Writer, rows []SweepRow) error {
	cw := csv.NewWriter(w)
	defer cw.Flush()

	header := []string{"batch_size", "arithmetic_intensity", "bound_type", "predicted_tokens_per_sec", "predicted_latency_ms"}
	if err := cw.Write(header); err != nil {
		return err
	}
	for _, r := range rows {
		rec := []string{
			strconv.Itoa(r.BatchSize),
			strconv.FormatFloat(r.ArithmeticIntensity, 'f', 4, 64),
			r.BoundType,
			strconv.FormatFloat(r.PredictedTokensPerSec, 'f', 4, 64),
			strconv.FormatFloat(r.PredictedLatencyMs, 'f', 4, 64),
		}
		if err := cw.Write(rec); err != nil {
			return err
		}
	}
	return cw.Error()
}

// FormatFloat renders a float with fixed precision for table display.
func FormatFloat(f float64, prec int) string {
	return strconv.FormatFloat(f, 'f', prec, 64)
}
