package cmd

import (
	"fmt"
	"math"

	"github.com/spf13/cobra"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"

	"roofline/internal/hardware"
	"roofline/internal/validate"
)

var (
	plotSweep string
	plotGPU   string
	plotOut   string
)

var plotCmd = &cobra.Command{
	Use:   "plot",
	Short: "Render a roofline plot (AI vs achievable FLOPS) from a sweep CSV",
	RunE: func(cmd *cobra.Command, args []string) error {
		gpu, err := hardware.Lookup(plotGPU)
		if err != nil {
			return err
		}
		rows, err := validate.LoadPredictions(plotSweep)
		if err != nil {
			return err
		}
		if len(rows) == 0 {
			return fmt.Errorf("sweep CSV %s has no rows", plotSweep)
		}

		p := plot.New()
		p.Title.Text = fmt.Sprintf("Roofline: %s", gpu.Name)
		p.X.Label.Text = "Arithmetic Intensity (FLOPs/byte)"
		p.Y.Label.Text = "Achievable Performance (FLOPs/sec)"
		p.X.Scale = plot.LogScale{}
		p.Y.Scale = plot.LogScale{}
		p.X.Tick.Marker = plot.LogTicks{}
		p.Y.Tick.Marker = plot.LogTicks{}

		balance := gpu.MachineBalance()
		aiMin, aiMax := rows[0].ArithmeticIntensity, rows[0].ArithmeticIntensity
		for _, r := range rows {
			aiMin = math.Min(aiMin, r.ArithmeticIntensity)
			aiMax = math.Max(aiMax, r.ArithmeticIntensity)
		}
		aiMin = math.Min(aiMin, balance) / 4
		aiMax = math.Max(aiMax, balance) * 4

		// Memory-bound diagonal: y = AI * peak_bandwidth, up to machine balance.
		memLine := plotter.XYs{
			{X: aiMin, Y: aiMin * gpu.PeakBandwidth},
			{X: balance, Y: balance * gpu.PeakBandwidth},
		}
		// Compute-bound ceiling: y = peak_FLOPS, from machine balance onward.
		computeLine := plotter.XYs{
			{X: balance, Y: gpu.PeakFLOPS},
			{X: aiMax, Y: gpu.PeakFLOPS},
		}

		points := make(plotter.XYs, len(rows))
		labels := make([]string, len(rows))
		for i, r := range rows {
			achieved := math.Min(gpu.PeakFLOPS, r.ArithmeticIntensity*gpu.PeakBandwidth)
			points[i] = plotter.XY{X: r.ArithmeticIntensity, Y: achieved}
			labels[i] = fmt.Sprintf("b=%d", r.BatchSize)
		}

		if err := plotutil.AddLines(p, "memory-bound ceiling", memLine, "compute-bound ceiling", computeLine); err != nil {
			return fmt.Errorf("adding roofline lines: %w", err)
		}
		scatter, err := plotter.NewScatter(points)
		if err != nil {
			return fmt.Errorf("creating scatter points: %w", err)
		}
		scatter.GlyphStyle.Radius = vg.Points(4)
		p.Add(scatter)
		p.Legend.Add("sweep batch sizes", scatter)

		labelData, err := plotter.NewLabels(plotter.XYLabels{XYs: points, Labels: labels})
		if err != nil {
			return fmt.Errorf("creating point labels: %w", err)
		}
		p.Add(labelData)

		p.Legend.Top = true
		if err := p.Save(8*vg.Inch, 6*vg.Inch, plotOut); err != nil {
			return fmt.Errorf("saving plot: %w", err)
		}
		fmt.Printf("wrote roofline plot to %s\n", plotOut)
		return nil
	},
}

func init() {
	rootCmd.AddCommand(plotCmd)
	plotCmd.Flags().StringVar(&plotSweep, "sweep", "", "path to sweep CSV (required)")
	plotCmd.Flags().StringVar(&plotGPU, "gpu", "", "GPU preset name used for the sweep (required)")
	plotCmd.Flags().StringVar(&plotOut, "out", "roofline.png", "output image path")
	plotCmd.MarkFlagRequired("sweep")
	plotCmd.MarkFlagRequired("gpu")
}
