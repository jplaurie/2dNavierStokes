#!/usr/bin/env python3
"""Create an MP4 or GIF of vorticity and/or recovered velocity snapshots."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from ns2d_plotting import (
    discover_vorticity,
    domain_lengths,
    read_csv,
    read_parameters,
    read_vorticity,
    repository_root,
    select_frames,
    use_plot_style,
    velocity_from_vorticity,
    write_animation,
)


QUANTITIES = {
    "vorticity": (r"vorticity $\omega$", "RdBu_r"),
    "u": (r"velocity $u$", "RdBu_r"),
    "v": (r"velocity $v$", "RdBu_r"),
}


def parse_arguments() -> argparse.Namespace:
    root = repository_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument(
        "--parameters", type=Path, default=root / "output/resolved_parameters.txt"
    )
    parser.add_argument(
        "--diagnostics", type=Path, default=root / "output/diagnostics.csv"
    )
    parser.add_argument("--output", type=Path, default=root / "figures/vorticity.mp4")
    parser.add_argument("--quantity", choices=[*QUANTITIES, "all"], default="vorticity")
    parser.add_argument("--frames", nargs="*", type=int)
    parser.add_argument("--start", type=int)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--dpi", type=int, default=140)
    parser.add_argument("--codec", choices=["h264", "h265"], default="h264")
    parser.add_argument("--vmin", type=float, help="override scale for one quantity")
    parser.add_argument("--vmax", type=float, help="override scale for one quantity")
    parser.add_argument(
        "--color-scale",
        choices=["global", "first", "dynamic"],
        default="global",
        help="derive fixed limits from all frames, the first frame, or each frame",
    )
    parser.add_argument("--interpolation", default="bilinear")
    parser.add_argument("--no-tex", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.quantity == "all" and (args.vmin is not None or args.vmax is not None):
        raise ValueError("vmin and vmax can only be used with one quantity")
    use_plot_style(not args.no_tex, font_size=14)
    files = discover_vorticity(args.data_dir)
    frames = select_frames(
        files,
        args.frames if args.frames else None,
        start=args.start,
        stop=args.stop,
        stride=args.stride,
    )
    parameters = read_parameters(args.parameters) if args.parameters.exists() else {}
    lx, ly = domain_lengths(parameters)
    extent = (0.0, lx, 0.0, ly)
    times: dict[int, float] = {}
    if args.diagnostics.exists():
        diagnostics = read_csv(args.diagnostics)
        times = {int(row["frame"]): float(row["time"]) for row in diagnostics}

    names = list(QUANTITIES) if args.quantity == "all" else [args.quantity]

    def fields(frame: int) -> dict[str, np.ndarray]:
        omega = read_vorticity(files[frame])
        result = {"vorticity": omega}
        if names == ["vorticity"]:
            return result
        u, v = velocity_from_vorticity(omega, lx, ly)
        result.update({"u": u, "v": v})
        return result

    first = fields(frames[0])
    limits = {
        name: max(float(np.max(np.abs(first[name]))), np.finfo(float).eps)
        for name in names
    }
    if args.color_scale == "global":
        for frame in frames[1:]:
            current = fields(frame)
            for name in names:
                limits[name] = max(limits[name], float(np.max(np.abs(current[name]))))

    if len(names) == 1:
        fig, axis = plt.subplots(figsize=(7, 6))
        axes = [axis]
    else:
        fig, axes_array = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True)
        axes = list(axes_array)
    images = []
    for axis, name in zip(axes, names):
        limit = limits[name]
        lower = -limit if args.vmin is None else args.vmin
        upper = limit if args.vmax is None else args.vmax
        image = axis.imshow(
            first[name],
            origin="lower",
            extent=extent,
            interpolation=args.interpolation,
            cmap=QUANTITIES[name][1],
            vmin=lower,
            vmax=upper,
        )
        axis.set_title(QUANTITIES[name][0])
        axis.set_xlabel(r"$x$")
        axis.set_ylabel(r"$y$")
        axis.set_aspect("equal")
        fig.colorbar(image, ax=axis)
        images.append(image)
    title = fig.suptitle("")

    def update(index: int):
        frame = frames[index]
        current = fields(frame)
        for image, name in zip(images, names):
            image.set_data(current[name])
            if args.color_scale == "dynamic":
                limit = max(float(np.max(np.abs(current[name]))), np.finfo(float).eps)
                lower = -limit if args.vmin is None else args.vmin
                upper = limit if args.vmax is None else args.vmax
                image.set_clim(lower, upper)
        time_text = rf", $t={times[frame]:.6g}$" if frame in times else ""
        title.set_text(rf"frame {frame}{time_text}")
        return [*images, title]

    movie = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=1000.0 / args.fps, blit=False
    )
    destination = write_animation(movie, args.output, args.fps, args.dpi, args.codec)
    plt.close(fig)
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
