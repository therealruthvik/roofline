#!/usr/bin/env python3
"""Convert vLLM benchmark_throughput.py stdout logs into roofline's
benchmark JSON schema.

Usage:
    1. Run vLLM's benchmark_throughput.py once per (gpu, model, batch_size)
       combination, redirecting stdout to a log file named:
           <gpu>_<model>_batch<N>.log
       e.g. A10G_mistral-7b_batch8.log

       python benchmarks/benchmark_throughput.py \\
         --model mistralai/Mistral-7B-v0.1 \\
         --input-len 128 --output-len 128 --num-prompts 8 \\
         > A10G_mistral-7b_batch8.log

    2. Run this script over all the logs:
           python scripts/convert_vllm_results.py logs/*.log -o benchmark.json

The script looks for vLLM's summary line, which (across versions) reads
something like:
    Throughput: 1.23 requests/s, 456.78 total tokens/s, 201.30 output tokens/s
It extracts the "output tokens/s" figure as measured_tokens_per_sec (falls
back to "total tokens/s" if output-tokens figure isn't present).
"""
import argparse
import json
import re
import sys
from pathlib import Path

THROUGHPUT_RE = re.compile(
    r"Throughput:\s*([\d.]+)\s*requests/s,\s*([\d.]+)\s*total tokens/s"
    r"(?:,\s*([\d.]+)\s*output tokens/s)?",
    re.IGNORECASE,
)

FILENAME_RE = re.compile(r"^(?P<gpu>[^_]+)_(?P<model>.+)_batch(?P<batch>\d+)$")


def parse_log(path: Path) -> dict:
    stem = path.stem
    m = FILENAME_RE.match(stem)
    if not m:
        raise ValueError(
            f"{path.name}: filename must match <gpu>_<model>_batch<N>.log, "
            f"e.g. A10G_mistral-7b_batch8.log"
        )

    text = path.read_text()
    tm = THROUGHPUT_RE.search(text)
    if not tm:
        raise ValueError(f"{path.name}: no 'Throughput: ...' line found in log")

    total_tps = float(tm.group(2))
    output_tps = float(tm.group(3)) if tm.group(3) else total_tps

    return {
        "gpu": m.group("gpu"),
        "model": m.group("model"),
        "batch_size": int(m.group("batch")),
        "measured_tokens_per_sec": output_tps,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", help="vLLM benchmark_throughput.py stdout log files")
    ap.add_argument("-o", "--out", default="benchmark.json", help="output JSON path (default: benchmark.json)")
    args = ap.parse_args()

    records = []
    for log_arg in args.logs:
        path = Path(log_arg)
        try:
            records.append(parse_log(path))
        except ValueError as e:
            print(f"skipping: {e}", file=sys.stderr)

    if not records:
        print("no valid records parsed, nothing written", file=sys.stderr)
        return 1

    records.sort(key=lambda r: (r["gpu"], r["model"], r["batch_size"]))
    Path(args.out).write_text(json.dumps(records, indent=2) + "\n")
    print(f"wrote {len(records)} records to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
