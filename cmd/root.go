// Package cmd holds the cobra command definitions for the roofline CLI.
package cmd

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
)

var rootCmd = &cobra.Command{
	Use:   "roofline",
	Short: "GPU roofline performance model for LLM inference",
	Long: `roofline predicts LLM inference throughput/latency from GPU hardware
specs and model config using roofline analysis, and validates those
predictions against real benchmark data.`,
}

// Execute runs the root command.
func Execute() {
	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
