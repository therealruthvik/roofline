package cmd

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"

	"roofline/internal/output"
	"roofline/internal/validate"
)

var (
	validatePredictions string
	validateActual      string
	validateGPU         string
	validateModel       string
)

var validateCmd = &cobra.Command{
	Use:   "validate",
	Short: "Compare roofline predictions against real benchmark measurements",
	RunE: func(cmd *cobra.Command, args []string) error {
		predicted, err := validate.LoadPredictions(validatePredictions)
		if err != nil {
			return err
		}
		actual, err := validate.LoadActual(validateActual)
		if err != nil {
			return err
		}

		joined := validate.Join(validateGPU, validateModel, predicted, actual)
		if len(joined) == 0 {
			return fmt.Errorf("no matching rows for gpu=%q model=%q between %s and %s", validateGPU, validateModel, validatePredictions, validateActual)
		}

		header := []string{"batch_size", "bound_type", "predicted_tok/s", "measured_tok/s", "pct_error", "flag"}
		rows := make([][]string, 0, len(joined))
		breakdowns := 0
		for _, j := range joined {
			flag := ""
			if j.ModelBreakdown {
				flag = "MODEL BREAKDOWN"
				breakdowns++
			}
			rows = append(rows, []string{
				fmt.Sprintf("%d", j.BatchSize),
				j.BoundType,
				output.FormatFloat(j.PredictedTokensPerSec, 2),
				output.FormatFloat(j.MeasuredTokensPerSec, 2),
				fmt.Sprintf("%+.2f%%", j.PercentError),
				flag,
			})
		}

		if err := output.Table(os.Stdout, header, rows); err != nil {
			return err
		}
		fmt.Fprintf(os.Stdout, "\n%d/%d rows exceed %.0f%% error (model breakdown)\n", breakdowns, len(joined), validate.BreakdownThresholdPct)
		return nil
	},
}

func init() {
	rootCmd.AddCommand(validateCmd)
	validateCmd.Flags().StringVar(&validatePredictions, "predictions", "", "path to predictions CSV from `roofline sweep` (required)")
	validateCmd.Flags().StringVar(&validateActual, "actual", "", "path to real benchmark JSON (required)")
	validateCmd.Flags().StringVar(&validateGPU, "gpu", "", "GPU name to match in actual data (required)")
	validateCmd.Flags().StringVar(&validateModel, "model", "", "model name to match in actual data (required)")
	validateCmd.MarkFlagRequired("predictions")
	validateCmd.MarkFlagRequired("actual")
	validateCmd.MarkFlagRequired("gpu")
	validateCmd.MarkFlagRequired("model")
}
