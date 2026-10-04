#!/usr/bin/env python3
"""Create README-ready timing and speedup plots from benchmark CSV data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator


ORDER = ("CPU serial", "CPU/OpenMP", "MPI", "CUDA FP64", "CUDA mixed")
COLORS = {"CPU serial": "#577590", "CPU/OpenMP": "#277da1", "MPI": "#f8961e",
          "CUDA FP64": "#43aa8b", "CUDA mixed": "#d1495b"}
MARKERS = {"CPU serial": "D", "CPU/OpenMP": "o", "MPI": "s",
           "CUDA FP64": "^", "CUDA mixed": "v"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("benchmarks/results.csv"))
    parser.add_argument("--metadata", type=Path, default=Path("benchmarks/system.json"))
    parser.add_argument("--output-prefix", type=Path,
                        default=Path("benchmarks/backend_scaling"))
    args = parser.parse_args()
    with args.input.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        parser.error("the input file has no benchmark rows")
    metadata = json.loads(args.metadata.read_text())

    samples: dict[str, dict[int, list[float]]] = {}
    for row in rows:
        samples.setdefault(row["backend"], {}).setdefault(int(row["resolution"]), []).append(
            float(row["seconds_per_step"]))
    labels = {
        "CPU serial": "CPU serial (1 core)",
        "CPU/OpenMP": f"CPU/OpenMP ({metadata['cpu_openmp_threads']} threads)",
        "MPI": f"MPI ({metadata['mpi_ranks']} ranks)",
        "CUDA FP64": "CUDA FP64",
        "CUDA mixed": "CUDA mixed (FP64 state, FP32 FFT)",
    }
    figure, (timing_axis, speedup_axis) = plt.subplots(1, 2, figsize=(10.8, 4.3))
    medians: dict[str, dict[int, float]] = {}
    for backend in ORDER:
        if backend not in samples:
            continue
        resolutions = sorted(samples[backend])
        values = [median(samples[backend][resolution]) for resolution in resolutions]
        medians[backend] = dict(zip(resolutions, values))
        lower = [value - min(samples[backend][resolution])
                 for resolution, value in zip(resolutions, values)]
        upper = [max(samples[backend][resolution]) - value
                 for resolution, value in zip(resolutions, values)]
        timing_axis.errorbar(resolutions, values, yerr=[lower, upper], label=labels[backend],
                             color=COLORS[backend], marker=MARKERS[backend], linewidth=2,
                             capsize=3)
    timing_axis.set(xscale="log", yscale="log", xlabel="Grid resolution, N (for N x N)",
                    ylabel="Time per ETD4 step (s)", title="Complete timestep time")
    timing_axis.grid(True, which="both", alpha=0.25)
    timing_axis.legend(frameon=False, fontsize=8.5)

    baseline = medians.get("CPU serial", {})
    for backend in ORDER:
        if backend not in medians:
            continue
        resolutions = sorted(set(baseline) & set(medians[backend]))
        speedups = [baseline[n] / medians[backend][n] for n in resolutions]
        speedup_axis.plot(resolutions, speedups, label=labels[backend], color=COLORS[backend],
                          marker=MARKERS[backend], linewidth=2)
    speedup_axis.axhline(1.0, color="0.45", linewidth=1, linestyle="--")
    speedup_axis.set(xscale="log", yscale="log", xlabel="Grid resolution, N (for N x N)",
                     ylabel="Speedup over one CPU core", title="Backend speedup")
    speedup_axis.grid(True, which="both", alpha=0.25)

    all_resolutions = sorted({n for values in samples.values() for n in values})
    for axis in (timing_axis, speedup_axis):
        axis.set_xscale("log", base=2)
        axis.set_xticks(all_resolutions, labels=[f"{n:,}" for n in all_resolutions])
        axis.xaxis.set_minor_locator(NullLocator())
    figure.suptitle(f"2D Navier--Stokes ETD4 benchmark\n{metadata['cpu_model']} · "
                    f"{metadata['gpu']}", fontsize=11)
    figure.tight_layout()
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "png"):
        output = args.output_prefix.with_suffix(f".{extension}")
        figure.savefig(output, dpi=180, bbox_inches="tight")
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
