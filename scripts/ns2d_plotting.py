"""Shared plotting helpers for the 2D Navier--Stokes solver."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib as mpl
import matplotlib.animation as mpl_animation
import matplotlib.pyplot as plt
import numpy as np


FRAME_PATTERN = re.compile(r"vorticity_(\d+)\.dat$")


def repository_root() -> Path:
    """Return the repository containing this module."""
    return Path(__file__).resolve().parent.parent


def use_plot_style(use_tex: bool = True, font_size: float = 16.0) -> None:
    """Apply a publication-oriented serif/LaTeX Matplotlib style."""
    mpl.rcParams.update(
        {
            "text.usetex": use_tex,
            "font.family": "serif",
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size,
            "legend.fontsize": 0.78 * font_size,
            "xtick.labelsize": 0.85 * font_size,
            "ytick.labelsize": 0.85 * font_size,
            "lines.linewidth": 2.0,
            "savefig.bbox": "tight",
            "savefig.format": "pdf",
            "figure.constrained_layout.use": True,
        }
    )


def read_parameters(path: str | Path) -> dict[str, str]:
    """Read either a user parameter file or ``resolved_parameters.txt``."""
    parameters: dict[str, str] = {}
    with Path(path).open(encoding="utf-8") as stream:
        for raw_line in stream:
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            key, *values = line.replace("=", " ").split()
            parameters[key] = " ".join(values)
    return parameters


def parameter_float(parameters: dict[str, str], key: str, default: float) -> float:
    value = parameters.get(key, "")
    return float(value) if value else default


def domain_lengths(parameters: dict[str, str]) -> tuple[float, float]:
    """Return the physical domain lengths recorded by the solver."""
    aspect_ratio = parameter_float(parameters, "aspectRatio", 1.0)
    lx = parameter_float(parameters, "domainLengthX", 2.0 * np.pi * aspect_ratio)
    ly = parameter_float(parameters, "domainLengthY", 2.0 * np.pi)
    return lx, ly


def discover_vorticity(data_directory: str | Path) -> dict[int, Path]:
    """Map saved frame numbers to vorticity snapshot paths."""
    files: dict[int, Path] = {}
    for path in sorted(Path(data_directory).glob("vorticity_*.dat")):
        match = FRAME_PATTERN.search(path.name)
        if match:
            files[int(match.group(1))] = path
    if not files:
        raise FileNotFoundError(
            f"no vorticity_XXXXXXXX.dat files found in {data_directory}"
        )
    return files


def read_vorticity(path: str | Path) -> np.ndarray:
    """Load a solver snapshot as a real array with shape ``(ny, nx)``."""
    values = np.loadtxt(path, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"{path} is not a rectangular two-dimensional snapshot")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path} contains a non-finite value")
    return values


def velocity_from_vorticity(
    vorticity: np.ndarray, lx: float, ly: float
) -> tuple[np.ndarray, np.ndarray]:
    r"""Recover velocity using ``omega=Delta psi`` and ``u=(-psi_y,psi_x)``."""
    if vorticity.ndim != 2:
        raise ValueError("vorticity must be a two-dimensional array")
    ny, nx = vorticity.shape
    if nx < 2 or ny < 2 or lx <= 0.0 or ly <= 0.0:
        raise ValueError("grid dimensions and domain lengths must be positive")
    omega_hat = np.fft.rfft2(vorticity)
    kx = 2.0 * np.pi * np.fft.rfftfreq(nx, d=lx / nx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=ly / ny)
    kx_grid = kx[np.newaxis, :]
    ky_grid = ky[:, np.newaxis]
    k_squared = kx_grid**2 + ky_grid**2
    inverse_k_squared = np.zeros_like(k_squared)
    np.divide(1.0, k_squared, out=inverse_k_squared, where=k_squared > 0.0)
    u_hat = 1j * ky_grid * inverse_k_squared * omega_hat
    v_hat = -1j * kx_grid * inverse_k_squared * omega_hat
    u = np.fft.irfft2(u_hat, s=(ny, nx))
    v = np.fft.irfft2(v_hat, s=(ny, nx))
    return u, v


def read_csv(path: str | Path) -> np.ndarray:
    """Load one of the solver CSV files as a named NumPy array."""
    table = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    if table.size == 0:
        raise ValueError(f"{path} has a header but no data rows")
    return np.atleast_1d(table)


def available_frames(table: np.ndarray) -> list[int]:
    """Return sorted unique frame numbers from a named CSV table."""
    return sorted(int(frame) for frame in np.unique(table["frame"]))


def select_frames(
    available: Iterable[int],
    requested: Sequence[int] | None = None,
    *,
    start: int | None = None,
    stop: int | None = None,
    stride: int = 1,
) -> list[int]:
    """Select frames, accepting negative indices such as ``[-1]`` for last."""
    choices = sorted(set(int(frame) for frame in available))
    if not choices:
        raise ValueError("there are no available frames")
    if requested is not None:
        selected: list[int] = []
        for frame in requested:
            if frame < 0:
                try:
                    resolved = choices[frame]
                except IndexError as error:
                    raise ValueError(
                        f"negative frame index {frame} exceeds the available range"
                    ) from error
            else:
                resolved = int(frame)
            if resolved not in choices:
                raise ValueError(f"frame {resolved} is not available")
            if resolved not in selected:
                selected.append(resolved)
        if not selected:
            raise ValueError("the frame selection is empty")
        return selected
    if stride < 1:
        raise ValueError("stride must be at least one")
    selected = [
        frame
        for frame in choices
        if (start is None or frame >= start) and (stop is None or frame <= stop)
    ][::stride]
    if not selected:
        raise ValueError("the frame selection is empty")
    return selected


def rows_for_frame(table: np.ndarray, frame: int) -> np.ndarray:
    """Return the rows for one frame, ordered by wavenumber when present."""
    rows = np.atleast_1d(table[table["frame"] == frame])
    if rows.size == 0:
        raise ValueError(f"frame {frame} is absent from the table")
    if "wavenumber" in (rows.dtype.names or ()):
        rows = rows[np.argsort(rows["wavenumber"])]
    return rows


def frame_time(table: np.ndarray, frame: int) -> float:
    """Return the saved time corresponding to a frame."""
    return float(rows_for_frame(table, frame)["time"][0])


def curves_for_frames(
    table: np.ndarray,
    frames: Sequence[int],
    columns: Sequence[str],
    mode: str,
) -> tuple[np.ndarray, list[dict[str, np.ndarray | float | str]]]:
    """Extract individual curves or their pointwise frame average."""
    if mode not in {"single", "multiple", "average"}:
        raise ValueError("mode must be 'single', 'multiple', or 'average'")
    if mode == "single" and len(frames) != 1:
        raise ValueError("single mode requires exactly one selected frame")
    rows = [rows_for_frame(table, frame) for frame in frames]
    k = np.asarray(rows[0]["wavenumber"], dtype=float)
    for entry in rows[1:]:
        if not np.allclose(entry["wavenumber"], k, rtol=1e-12, atol=1e-14):
            raise ValueError("selected frames do not use the same wavenumber bins")
    if mode == "average":
        curve: dict[str, np.ndarray | float | str] = {
            column: np.mean([entry[column] for entry in rows], axis=0)
            for column in columns
        }
        curve["time"] = float(np.mean([entry["time"][0] for entry in rows]))
        curve["label"] = (
            rf"average, $t\in[{rows[0]['time'][0]:.4g},{rows[-1]['time'][0]:.4g}]$"
        )
        return k, [curve]
    curves: list[dict[str, np.ndarray | float | str]] = []
    for frame, entry in zip(frames, rows):
        curve = {column: np.asarray(entry[column]) for column in columns}
        curve["time"] = float(entry["time"][0])
        curve["label"] = rf"$t={entry['time'][0]:.4g}$ (frame {frame})"
        curves.append(curve)
    return k, curves


def positive_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mask non-positive or non-finite values before a logarithmic plot."""
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    return x[mask], y[mask]


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average with unchanged length and NaNs at the edges."""
    array = np.asarray(values, dtype=float)
    if window <= 1:
        return array.copy()
    if window > array.size:
        raise ValueError("rolling-average window exceeds the selected data length")
    kernel = np.ones(window) / window
    result = np.full(array.shape, np.nan)
    valid = np.convolve(array, kernel, mode="valid")
    left = (window - 1) // 2
    result[left : left + valid.size] = valid
    return result


def filter_time_series(
    table: np.ndarray,
    time_range: tuple[float | None, float | None] | None = None,
    frame_range: tuple[int | None, int | None] | None = None,
) -> np.ndarray:
    """Select an inclusive time and/or frame interval from diagnostics."""
    mask = np.ones(table.size, dtype=bool)
    if time_range is not None:
        lower, upper = time_range
        if lower is not None:
            mask &= table["time"] >= lower
        if upper is not None:
            mask &= table["time"] <= upper
    if frame_range is not None:
        lower, upper = frame_range
        if lower is not None:
            mask &= table["frame"] >= lower
        if upper is not None:
            mask &= table["frame"] <= upper
    selected = table[mask]
    if selected.size == 0:
        raise ValueError("the requested diagnostics interval is empty")
    return selected


def save_figure(fig: plt.Figure, path: str | Path, dpi: int = 200) -> Path:
    """Create the destination directory and save a tightly cropped figure."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=dpi)
    return destination


def write_animation(
    movie_animation,
    path: str | Path,
    fps: float,
    dpi: int,
    codec: str = "h264",
) -> Path:
    """Save a Matplotlib animation with ffmpeg (MP4) or Pillow (GIF)."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = destination.suffix.lower()
    if suffix == ".gif":
        movie_animation.save(destination, writer="pillow", fps=fps, dpi=dpi)
    elif suffix == ".mp4":
        codecs = {"h264": "libx264", "h265": "libx265"}
        if codec not in codecs:
            raise ValueError("codec must be 'h264' or 'h265'")
        writer = mpl_animation.FFMpegWriter(
            fps=fps,
            codec=codecs[codec],
            extra_args=["-pix_fmt", "yuv420p"],
        )
        movie_animation.save(destination, writer=writer, dpi=dpi)
    else:
        raise ValueError("movie output must end in .mp4 or .gif")
    return destination
