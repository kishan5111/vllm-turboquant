#!/usr/bin/env python3
"""
Summarize benchmark results: FP8 vs TurboQuant side-by-side.
Reads results/fp8_baseline.json and results/tq_baseline.json.
Outputs results/summary.csv and prints a table.
"""
import csv
import json
import os
import sys
from pathlib import Path


def load_json(path: str) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def main():
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")

    fp8_data = load_json(results_dir / "fp8_baseline.json")
    tq_data  = load_json(results_dir / "turboquant-4bit_baseline.json")

    if not fp8_data and not tq_data:
        print("No result files found in", results_dir)
        return

    # Build lookup: workload -> results dict
    def index(data):
        if not data:
            return {}
        return {r["workload"]: r for r in data.get("results", [])}

    fp8 = index(fp8_data)
    tq  = index(tq_data)

    all_workloads = sorted(set(fp8) | set(tq))

    rows = []
    header = [
        "workload",
        "fp8_tok_per_s", "tq_tok_per_s", "speedup_vs_fp8",
        "fp8_req_per_s", "tq_req_per_s",
        "fp8_p99_lat_s", "tq_p99_lat_s",
        "fp8_vram_gb",   "tq_vram_gb",   "vram_savings_pct",
        "fp8_kv_bytes",  "tq_kv_bytes",  "compression_ratio",
    ]

    print()
    print(f"{'Workload':<22} {'FP8 tok/s':>10} {'TQ tok/s':>10} {'Speedup':>8} "
          f"{'FP8 VRAM':>9} {'TQ VRAM':>9} {'VRAM save':>10}")
    print("-" * 90)

    for wl in all_workloads:
        f = fp8.get(wl, {})
        t = tq.get(wl, {})

        f_tok = f.get("output_tok_per_s", 0)
        t_tok = t.get("output_tok_per_s", 0)
        speedup = round(t_tok / f_tok, 3) if f_tok > 0 else 0

        f_vram = f.get("peak_vram_gb", 0)
        t_vram = t.get("peak_vram_gb", 0)
        vram_save = round((1 - t_vram / f_vram) * 100, 1) if f_vram > 0 else 0

        f_kv = f.get("effective_kv_bytes_per_token", 0)
        t_kv = t.get("effective_kv_bytes_per_token", 0)
        comp_ratio = round(f_kv / t_kv, 3) if t_kv > 0 else 0

        print(f"{wl:<22} {f_tok:>10.1f} {t_tok:>10.1f} {speedup:>7.3f}x "
              f"{f_vram:>8.1f}G {t_vram:>8.1f}G {vram_save:>8.1f}%")

        rows.append({
            "workload": wl,
            "fp8_tok_per_s": f_tok, "tq_tok_per_s": t_tok, "speedup_vs_fp8": speedup,
            "fp8_req_per_s": f.get("request_per_s", 0), "tq_req_per_s": t.get("request_per_s", 0),
            "fp8_p99_lat_s": f.get("p99_latency_s", 0), "tq_p99_lat_s": t.get("p99_latency_s", 0),
            "fp8_vram_gb": f_vram, "tq_vram_gb": t_vram, "vram_savings_pct": vram_save,
            "fp8_kv_bytes": f_kv, "tq_kv_bytes": t_kv, "compression_ratio": comp_ratio,
        })

    out_path = results_dir / "summary.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    print()
    print(f"Summary saved → {out_path}")


if __name__ == "__main__":
    main()
