package validate

import (
	"math"
	"testing"
)

func TestJoin_ComputesPercentErrorAndSortsWorstFirst(t *testing.T) {
	predicted := []Predicted{
		{BatchSize: 1, PredictedTokensPerSec: 100},
		{BatchSize: 4, PredictedTokensPerSec: 90},
		{BatchSize: 8, PredictedTokensPerSec: 80},
	}
	actual := []Actual{
		{GPU: "A10G", Model: "mistral-7b", BatchSize: 1, MeasuredTokensPerSec: 95},  // ~5.3% error
		{GPU: "A10G", Model: "mistral-7b", BatchSize: 4, MeasuredTokensPerSec: 50},  // 80% error -> breakdown
		{GPU: "A10G", Model: "mistral-7b", BatchSize: 8, MeasuredTokensPerSec: 100}, // -20% error
		{GPU: "A10G", Model: "other-model", BatchSize: 1, MeasuredTokensPerSec: 1},  // wrong model, must be excluded
	}

	got := Join("A10G", "mistral-7b", predicted, actual)

	if len(got) != 3 {
		t.Fatalf("expected 3 joined rows, got %d", len(got))
	}
	// Worst error (batch=4, 80%) must come first.
	if got[0].BatchSize != 4 {
		t.Errorf("expected worst-error row first (batch=4), got batch=%d", got[0].BatchSize)
	}
	if !got[0].ModelBreakdown {
		t.Errorf("expected batch=4 row to be flagged as model breakdown (80%% error)")
	}
	if got[0].MeasuredTokensPerSec != 50 {
		t.Errorf("expected measured=50, got %v", got[0].MeasuredTokensPerSec)
	}

	// batch=1 should be last (smallest error).
	last := got[len(got)-1]
	if last.BatchSize != 1 {
		t.Errorf("expected smallest-error row last (batch=1), got batch=%d", last.BatchSize)
	}
	if last.ModelBreakdown {
		t.Errorf("batch=1 row (~5%% error) should not be flagged as breakdown")
	}

	wantPct := (100.0 - 95.0) / 95.0 * 100
	if math.Abs(last.PercentError-wantPct) > 1e-9 {
		t.Errorf("percent error = %v, want %v", last.PercentError, wantPct)
	}
}

func TestJoin_SkipsBatchSizesWithNoActualMatch(t *testing.T) {
	predicted := []Predicted{
		{BatchSize: 1, PredictedTokensPerSec: 100},
		{BatchSize: 16, PredictedTokensPerSec: 10},
	}
	actual := []Actual{
		{GPU: "T4", Model: "llama-3.1-8b", BatchSize: 1, MeasuredTokensPerSec: 100},
	}

	got := Join("T4", "llama-3.1-8b", predicted, actual)
	if len(got) != 1 {
		t.Fatalf("expected 1 joined row (batch=16 has no actual data), got %d", len(got))
	}
	if got[0].BatchSize != 1 {
		t.Errorf("expected surviving row to be batch=1, got %d", got[0].BatchSize)
	}
}

func TestBreakdownThreshold(t *testing.T) {
	predicted := []Predicted{{BatchSize: 1, PredictedTokensPerSec: 125}}
	actual := []Actual{{GPU: "A10G", Model: "mistral-7b", BatchSize: 1, MeasuredTokensPerSec: 100}} // exactly 25% error

	got := Join("A10G", "mistral-7b", predicted, actual)
	if len(got) != 1 {
		t.Fatalf("expected 1 row, got %d", len(got))
	}
	if got[0].ModelBreakdown {
		t.Errorf("exactly 25%% error should not be flagged (threshold is exceeds 25%%, not >=)")
	}
}
