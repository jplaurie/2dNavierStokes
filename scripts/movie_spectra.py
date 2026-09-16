#!/usr/bin/env python3
"""Create an MP4 or GIF of energy and enstrophy spectra."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from ns2d_plotting import (
    available_frames,
    read_csv,
    repository_root,
    rows_for_frame,
    select_frames,
    use_plot_style,
    write_animation,
)


def parse_arguments() -> argparse.Namespace:
    root = repository_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "output/spectra.csv")
    parser.add_argument("--output", type=Path, default=root / "figures/spectra.mp4")
    parser.add_argument("--frames", nargs="*", type=int)
    parser.add_argument("--start", type=int)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--segment-mean", action="store_true")
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--dpi", type=int, default=140)
    parser.add_argument("--codec", choices=["h264", "h265"], default="h264")
    parser.add_argument("--no-tex", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    use_plot_style(not args.no_tex, font_size=14)
    table = read_csv(args.input)
    frames = select_frames(
        available_frames(table),
        args.frames if args.frames else None,
        start=args.start,
        stop=args.stop,
        stride=args.stride,
    )
    energy_column = "segment_mean_energy_spectrum" if args.segment_mean else "energy_spectrum"
    enstrophy_column = (
        "segment_mean_enstrophy_spectrum" if args.segment_mean else "enstrophy_spectrum"
    )

    def values(frame: int):
        rows = rows_for_frame(table, frame)
        k = np.asarray(rows["wavenumber"])
        energy = np.asarray(rows[energy_column])
        enstrophy = np.asarray(rows[enstrophy_column])
        return k, energy, enstrophy, float(rows["time"][0])

    datasets = [values(frame) for frame in frames]
    positive_k = np.concatenate([entry[0][entry[0] > 0.0] for entry in datasets])
    positive_energy = np.concatenate(
        [entry[1][np.isfinite(entry[1]) & (entry[1] > 0.0)] for entry in datasets]
    )
    positive_enstrophy = np.concatenate(
        [entry[2][np.isfinite(entry[2]) & (entry[2] > 0.0)] for entry in datasets]
    )
    if not positive_k.size or not positive_energy.size or not positive_enstrophy.size:
        raise ValueError("the selected spectra have no finite positive data to animate")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    (energy_line,) = axes[0].plot([], [], color="C0")
    (enstrophy_line,) = axes[1].plot([], [], color="C1")
    for axis, ylabel, data in zip(
        axes, [r"$E(k)$", r"$Z(k)$"], [positive_energy, positive_enstrophy]
    ):
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(positive_k.min(), positive_k.max())
        lower, upper = data.min(), data.max()
        if lower == upper:
            lower, upper = 0.9 * lower, 1.1 * upper
        axis.set_ylim(lower, upper)
        axis.set_xlabel(r"$k$")
        axis.set_ylabel(ylabel)
        axis.grid(True, which="both", alpha=0.2)
    title = fig.suptitle("")

    def update(index: int):
        k, energy, enstrophy, time = datasets[index]
        energy_mask = np.isfinite(k) & np.isfinite(energy) & (k > 0.0) & (energy > 0.0)
        enstrophy_mask = (
            np.isfinite(k) & np.isfinite(enstrophy) & (k > 0.0) & (enstrophy > 0.0)
        )
        energy_line.set_data(k[energy_mask], energy[energy_mask])
        enstrophy_line.set_data(k[enstrophy_mask], enstrophy[enstrophy_mask])
        title.set_text(rf"frame {frames[index]}, $t={time:.6g}$")
        return energy_line, enstrophy_line, title

    movie = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=1000.0 / args.fps, blit=False
    )
    destination = write_animation(movie, args.output, args.fps, args.dpi, args.codec)
    plt.close(fig)
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
