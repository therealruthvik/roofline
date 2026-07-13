package cmd

import (
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/spf13/cobra"

	"roofline/internal/hardware"
	"roofline/internal/model"
	"roofline/internal/output"
	"roofline/internal/roofline"
)

var (
	sweepGPU       string
	sweepModel     string
	sweepPrecision string
	sweepBatches   string
	sweepSeqLen    int
	sweepOut       string
)

var sweepCmd = &cobra.Command{
	Use:   "sweep",
	Short: "Sweep batch sizes and write predicted throughput/latency to CSV",
	RunE: func(cmd *cobra.Command, args []string) error {
		gpu, err := hardware.Lookup(sweepGPU)
		if err != nil {
			return err
		}
		cfg, err := model.Lookup(sweepModel)
		if err != nil {
			return err
		}
		batches, err := parseBatchList(sweepBatches)
		if err != nil {
			return err
		}

		rows := make([]output.SweepRow, 0, len(batches))
		for _, b := range batches {
			result, err := roofline.Compute(gpu, cfg, model.Precision(sweepPrecision), b, sweepSeqLen)
			if err != nil {
				return err
			}
			rows = append(rows, output.SweepRow{
				BatchSize:             b,
				ArithmeticIntensity:   result.ArithmeticIntensity,
				BoundType:             string(result.Bound),
				PredictedTokensPerSec: result.PredictedTokensPerSec,
				PredictedLatencyMs:    result.PredictedLatencyMs,
			})
		}

		f, err := os.Create(sweepOut)
		if err != nil {
			return fmt.Errorf("creating output file: %w", err)
		}
		defer f.Close()

		if err := output.WriteSweepCSV(f, rows); err != nil {
			return fmt.Errorf("writing csv: %w", err)
		}

		fmt.Fprintf(os.Stdout, "wrote %d rows to %s\n", len(rows), sweepOut)
		return nil
	},
}

func parseBatchList(s string) ([]int, error) {
	parts := strings.Split(s, ",")
	batches := make([]int, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		b, err := strconv.Atoi(p)
		if err != nil {
			return nil, fmt.Errorf("invalid batch size %q: %w", p, err)
		}
		batches = append(batches, b)
	}
	if len(batches) == 0 {
		return nil, fmt.Errorf("no batch sizes provided")
	}
	return batches, nil
}

func init() {
	rootCmd.AddCommand(sweepCmd)
	sweepCmd.Flags().StringVar(&sweepGPU, "gpu", "", "GPU preset name (required)")
	sweepCmd.Flags().StringVar(&sweepModel, "model", "", "model preset name (required)")
	sweepCmd.Flags().StringVar(&sweepPrecision, "precision", "fp16", "weight precision: fp16, bf16, int8, int4")
	sweepCmd.Flags().StringVar(&sweepBatches, "batch", "1,4,8,16,32,64", "comma-separated batch sizes")
	sweepCmd.Flags().IntVar(&sweepSeqLen, "seqlen", 2048, "sequence length (KV-cache depth)")
	sweepCmd.Flags().StringVar(&sweepOut, "out", "sweep.csv", "output CSV path")
	sweepCmd.MarkFlagRequired("gpu")
	sweepCmd.MarkFlagRequired("model")
}
