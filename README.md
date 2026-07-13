# roofline

A GPU roofline performance model for LLM inference. Predicts decode-step
throughput and latency from GPU hardware specs + model config, and
validates those predictions against real benchmark measurements.

## Install / build

```
go build -o roofline .
```

## The math

Per decode step (generating one new token per sequence in the batch):

- `FLOPs/token ≈ 2 × num_params` — standard decode-step FLOPs approximation, independent of batch size (batch scales total compute and total output tokens by the same factor, so they cancel).
- `Bytes/token = weight_bytes + kv_cache_bytes`, where:
  - `weight_bytes = num_params × bytes_per_param` (2 for fp16/bf16, 1 for int8, 0.5 for int4) — the full weight set read once per step.
  - `kv_cache_bytes = batch_size × num_layers × hidden_dim × 2 × bytes_per_param × seq_len` — the K and V cache read, which grows with sequence length and batch size.
- `Arithmetic Intensity (AI) = FLOPs/token ÷ Bytes/token`
- `Machine balance = peak_FLOPS ÷ peak_mem_bandwidth`
- If `AI < machine balance`: **memory-bound**, `throughput = peak_mem_bandwidth ÷ bytes_per_token`
- Else: **compute-bound**, `throughput = peak_FLOPS ÷ flops_per_token`
- `latency_per_token = 1 ÷ throughput`

Important simplification, called out explicitly: `weight_bytes` is **not**
divided by batch size. In real inference engines, one weight read is
amortized across every sequence in the batch (that's the entire point of
batching — better weight reuse per byte moved). This model charges the
full weight-read cost per decode step regardless of batch size, so its
predictions get worse specifically as batch size grows. This is
intentional-by-omission (a first-order roofline model, not a simulator)
and is the main driver of the divergence documented below.

## GPU presets

| GPU | Peak FP16 FLOPS | Peak Bandwidth | VRAM |
|---|---|---|---|
| A10G | 125 TFLOPS | 600 GB/s | 24 GB |
| T4 | 65 TFLOPS | 320 GB/s | 16 GB |
| A100-40GB | 312 TFLOPS | 1555 GB/s | 40 GB |
| A100-80GB | 312 TFLOPS | 2039 GB/s | 80 GB |
| H100-SXM | 990 TFLOPS | 3350 GB/s | 80 GB |

## Model presets

| Model | Layers | Hidden dim | Params |
|---|---|---|---|
| mistral-7b | 32 | 4096 | 7.24B |
| llama-3.1-8b | 32 | 4096 | 8.03B |

## Commands

### `model` — single-point prediction

```
roofline model --gpu A10G --model mistral-7b --precision fp16 --batch 1 --seqlen 2048
```

Prints arithmetic intensity, machine balance, bound classification,
predicted tokens/sec, and predicted latency/token as a table.

### `sweep` — batch-size sweep to CSV

```
roofline sweep --gpu A10G --model mistral-7b --batch 1,4,8,16,32,64 --seqlen 2048 --out sweep.csv
```

Writes `sweep.csv` with columns: `batch_size, arithmetic_intensity,
bound_type, predicted_tokens_per_sec, predicted_latency_ms`.

### `validate` — compare predictions to real measurements

```
roofline validate --predictions sweep.csv --actual benchmark.json --gpu A10G --model mistral-7b
```

`benchmark.json` is either a top-level array, or an object with a
`results`/`benchmarks` key, of records shaped like:

```json
{"gpu": "A10G", "model": "mistral-7b", "batch_size": 8, "measured_tokens_per_sec": 201.3}
```

Rows are joined on `gpu + model + batch_size`, percent error is computed
as `(predicted - measured) / measured × 100`, printed worst-error-first,
and any row where `|error| > 25%` is flagged `MODEL BREAKDOWN`.

### `plot` — roofline chart

```
roofline plot --sweep sweep.csv --gpu A10G --out roofline.png
```

Renders arithmetic intensity (log x) vs. achievable FLOPs/sec (log y),
with the memory-bound diagonal, the compute-bound ceiling, and each swept
batch size plotted as a point.

## Getting real benchmark data

`roofline validate` needs a JSON file of real measurements. If you don't
have one, run vLLM's own throughput benchmark and convert its output:

```
python benchmarks/benchmark_throughput.py \
  --model mistralai/Mistral-7B-v0.1 \
  --input-len 128 --output-len 128 --num-prompts 8 \
  > A10G_mistral-7b_batch8.log
```

Repeat per (gpu, model, batch_size), naming each log
`<gpu>_<model>_batch<N>.log`, then convert them all in one shot:

```
python3 scripts/convert_vllm_results.py *.log -o benchmark.json
```

`scripts/convert_vllm_results.py` parses vLLM's `Throughput: ...
requests/s, ... total tokens/s, ... output tokens/s` summary line and
emits the JSON schema `validate` expects. A10G maps to AWS `g5.xlarge`,
T4 maps to `g4dn.xlarge`, if you need to rent the hardware.

## Model vs Reality

**Status: illustrative, not yet validated against real hardware.** The
table below comes from `testdata/synthetic_example_a10g_mistral7b.json`,
a fixture with plausible-but-fabricated vLLM-style numbers, used only to
exercise the `validate` command end-to-end. Swap in real benchmark JSON
(same schema) and rerun the command below to replace this table with
actual measurements:

```
roofline sweep --gpu A10G --model mistral-7b --batch 1,4,8,16,32,64 --seqlen 2048 --out sweep.csv
roofline validate --predictions sweep.csv --actual <your-real-benchmark.json> --gpu A10G --model mistral-7b
```

Current output against the synthetic fixture:

| batch_size | bound_type | predicted_tok/s | measured_tok/s | pct_error | flag |
|---|---|---|---|---|---|
| 64 | memory-bound | 7.21 | 468.90 | -98.46% | MODEL BREAKDOWN |
| 32 | memory-bound | 12.29 | 421.50 | -97.09% | MODEL BREAKDOWN |
| 16 | memory-bound | 18.95 | 312.70 | -93.94% | MODEL BREAKDOWN |
| 8 | memory-bound | 26.01 | 201.30 | -87.08% | MODEL BREAKDOWN |
| 4 | memory-bound | 31.96 | 118.60 | -73.05% | MODEL BREAKDOWN |
| 1 | memory-bound | 38.58 | 34.20 | +12.80% | |

### Why the model diverges (and why this is expected)

The batch=1 row is the only one that should be trusted as directionally
meaningful right now (+12.8% error, no flag) — this is the one regime
where the model's core assumption (one weight read charged per decode
step, no batching amortization) matches reality, because there's no
batching happening. The batch≥4 rows are expected to blow up, for
reasons the model doesn't (and isn't trying to) capture:

1. **Weight-read amortization.** This is the dominant effect. Real
   inference engines load each weight tile once per decode step and
   reuse it across every sequence in the batch via batched GEMMs. This
   model charges `weight_bytes` fully on every step regardless of batch
   size (see "Important simplification" above), so its memory-bound
   throughput estimate barely grows with batch — while real throughput
   grows roughly linearly until compute or scheduling limits kick in.
   This alone accounts for most of the gap at batch≥8.

2. **KV-cache layout and attention kernel efficiency.** The KV-cache
   byte estimate here is a flat `batch × layers × hidden × 2 × seq_len ×
   bytes_per_elem` read. Real kernels (paged attention, FlashAttention-
   style fused kernels) achieve much better effective bandwidth
   utilization on this access pattern than a naive read would, and
   paged KV-cache allocation (vLLM's core contribution) avoids the
   fragmentation and over-allocation a naive contiguous KV buffer would
   incur — so real KV traffic is often lower per token than the linear
   estimate implies at higher batch/seq_len.

3. **Continuous batching / scheduling overlap.** vLLM overlaps
   prefill and decode across requests and keeps the GPU fed via
   continuous (iteration-level) batching, hiding memory-bound stalls
   that a naive per-step model assumes are fully serialized. The
   roofline model here assumes every decode step pays its full
   memory-bound latency in isolation; the real scheduler pipelines
   this work.

4. **Batch size is a controlled input for vLLM's scheduler, not a hard
   multiplier on isolated per-sequence cost.** As the sweep goes from
   batch=1 to batch=64, real measured throughput grows ~14×, while
   predicted throughput here actually *falls* (because KV-cache bytes
   scale linearly with batch while weight bytes don't shrink). The
   model's decode-step abstraction is closer to "cost of stepping one
   isolated sequence forward" than "cost of stepping a scheduled batch
   forward" — the latter is what real serving systems optimize for.

**Practical takeaway:** this roofline model is useful for order-of-
magnitude, batch=1 latency-bound estimates (e.g. "is this workload
memory- or compute-bound on this GPU, roughly") but should not be trusted
for absolute batched-throughput predictions without adding a weight-
amortization term. A natural follow-up would be to divide `weight_bytes`
by `batch_size` in `internal/model.BytesPerToken` and re-validate — that
single change should close most of the gap above batch=1.

## Project structure

```
cmd/                  cobra command definitions (model, sweep, validate, plot)
internal/hardware/    GPU spec table + lookup
internal/model/       model config presets, FLOPs/bytes calculators
internal/roofline/    AI calculation, bound classification, throughput prediction
internal/validate/    load real benchmark JSON, join + compute error
internal/output/      table formatting, CSV writer
testdata/             benchmark JSON fixtures
```

## Testing

```
go test ./...
```

`internal/roofline` and `internal/validate` have table-driven tests
covering both memory-bound and compute-bound classification, throughput
math, and the join/error/breakdown-flagging logic.
