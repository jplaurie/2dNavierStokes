#!/usr/bin/env python3
"""Create an MP4 or GIF of energy and enstrophy fluxes."""

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
    parser.add_argument("--input", type=Path, default=root / "output/fluxes.csv")
    parser.add_argument("--output", type=Path, default=root / "figures/fluxes.mp4")
    parser.add_argument("--frames", nargs="*", type=int)
    parser.add_argument("--start", type=int)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--segment-mean", action="store_true")
    parser.add_argument("--x-scale", choices=["linear", "log"], default="log")
    parser.add_argument("--y-scale", choices=["linear", "symlog"], default="symlog")
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
    energy_column = "segment_mean_energy_flux" if args.segment_mean else "energy_flux"
    enstrophy_column = (
        "segment_mean_enstrophy_flux" if args.segment_mean else "enstrophy_flux"
    )

    def values(frame: int):
        rows = rows_for_frame(table, frame)
        k = np.asarray(rows["wavenumber"])
        energy = np.asarray(rows[energy_column])
        enstrophy = np.asarray(rows[enstrophy_column])
        mask = np.isfinite(k) & np.isfinite(energy) & np.isfinite(enstrophy)
        if args.x_scale == "log":
            mask &= k > 0.0
        return k[mask], energy[mask], enstrophy[mask], float(rows["time"][0])

    datasets = [values(frame) for frame in frames]
    if any(not entry[0].size for entry in datasets):
        raise ValueError("a selected flux frame contains no finite plottable data")
    k_all = np.concatenate([entry[0] for entry in datasets])
    energy_all = np.concatenate([entry[1] for entry in datasets])
    enstrophy_all = np.concatenate([entry[2] for entry in datasets])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    (energy_line,) = axes[0].plot([], [], color="C0")
    (enstrophy_line,) = axes[1].plot([], [], color="C1")
    for axis, ylabel, data in zip(
        axes, [r"$\Pi_E(k)$", r"$\Pi_Z(k)$"], [energy_all, enstrophy_all]
    ):
        axis.axhline(0.0, color="0.25", linewidth=0.8)
        axis.set_xscale(args.x_scale)
        axis.set_yscale(args.y_scale)
        x_min, x_max = float(k_all.min()), float(k_all.max())
        if x_min == x_max:
            x_min, x_max = x_min - 0.5, x_max + 0.5
        axis.set_xlim(x_min, x_max)
        limit = max(1.05 * float(np.max(np.abs(data))), np.finfo(float).eps)
        axis.set_ylim(-limit, limit)
        axis.set_xlabel(r"$k$")
        axis.set_ylabel(ylabel)
        axis.grid(True, which="both", alpha=0.2)
    title = fig.suptitle("")

    def update(index: int):
        k, energy, enstrophy, time = datasets[index]
        energy_line.set_data(k, energy)
        enstrophy_line.set_data(k, enstrophy)
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
