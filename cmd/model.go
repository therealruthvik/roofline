package cmd

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"

	"roofline/internal/hardware"
	"roofline/internal/model"
	"roofline/internal/output"
	"roofline/internal/roofline"
)

var (
	modelGPU       string
	modelName      string
	modelPrecision string
	modelBatch     int
	modelSeqLen    int
)

var modelCmd = &cobra.Command{
	Use:   "model",
	Short: "Predict throughput/latency for a single GPU + model configuration",
	RunE: func(cmd *cobra.Command, args []string) error {
		gpu, err := hardware.Lookup(modelGPU)
		if err != nil {
			return err
		}
		cfg, err := model.Lookup(modelName)
		if err != nil {
			return err
		}
		result, err := roofline.Compute(gpu, cfg, model.Precision(modelPrecision), modelBatch, modelSeqLen)
		if err != nil {
			return err
		}

		header := []string{"metric", "value"}
		rows := [][]string{
			{"gpu", gpu.Name},
			{"model", cfg.Name},
			{"precision", modelPrecision},
			{"batch_size", fmt.Sprintf("%d", modelBatch)},
			{"seq_len", fmt.Sprintf("%d", modelSeqLen)},
			{"arithmetic_intensity", output.FormatFloat(result.ArithmeticIntensity, 4) + " FLOPs/byte"},
			{"machine_balance", output.FormatFloat(result.MachineBalance, 4) + " FLOPs/byte"},
			{"bound", string(result.Bound)},
			{"predicted_tokens_per_sec", output.FormatFloat(result.PredictedTokensPerSec, 2)},
			{"predicted_latency_ms_per_token", output.FormatFloat(result.PredictedLatencyMs, 4)},
		}
		return output.Table(os.Stdout, header, rows)
	},
}

func init() {
	rootCmd.AddCommand(modelCmd)
	modelCmd.Flags().StringVar(&modelGPU, "gpu", "", "GPU preset name (required)")
	modelCmd.Flags().StringVar(&modelName, "model", "", "model preset name (required)")
	modelCmd.Flags().StringVar(&modelPrecision, "precision", "fp16", "weight precision: fp16, bf16, int8, int4")
	modelCmd.Flags().IntVar(&modelBatch, "batch", 1, "batch size")
	modelCmd.Flags().IntVar(&modelSeqLen, "seqlen", 2048, "sequence length (KV-cache depth)")
	modelCmd.MarkFlagRequired("gpu")
	modelCmd.MarkFlagRequired("model")
}
