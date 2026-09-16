# Plotting and movies

These notebooks and scripts read the current solver output format directly.
The notebooks write publication-ready PDF figures, while the command-line
movie generators produce MP4 or GIF animations. The physical-field tools read
the saved vorticity and recover velocity spectrally from

```math
\omega=\Delta\psi,\qquad u=-\partial_y\psi,\qquad v=\partial_x\psi.
```

No additional velocity output from the solver is required.

## Requirements

All tools require Python 3, NumPy, and Matplotlib; the notebooks additionally
require Jupyter. PDF plots use external LaTeX by default. Set `USE_TEX=False`
in a notebook when LaTeX is unavailable. MP4 output also requires `ffmpeg`;
GIF output uses Matplotlib's Pillow writer.

Paths default to `data/`, `output/`, and `figures/`. The quick-start example
uses `data/quickstart_modern/` and `output/quickstart_modern/`, so set those
paths explicitly in the notebook configuration cells or movie arguments.

## Plotting notebooks

- `vorticity.ipynb` writes separate vorticity, horizontal-velocity, and
  vertical-velocity PDFs.
- `spectra.ipynb` writes separate energy and enstrophy spectrum PDFs.
- `fluxes.ipynb` writes a two-panel energy/enstrophy flux PDF.
- `diagnostics.ipynb` writes one energy/enstrophy PDF and one dissipation-rate
  PDF.

Edit the clearly marked **Configuration** cell near the top of each notebook.
The physical-field, spectrum, and flux notebooks support `MODE='single'`,
`'multiple'`, or `'average'`. `FRAMES=None` uses the range controlled by
`FRAME_START`, `FRAME_STOP`, and `FRAME_STRIDE`; negative entries such as
`FRAMES=[-1]` select from the end. Spectrum and flux notebooks can select the
solver's running segment means with `USE_SEGMENT_MEAN=True`.

Launch Jupyter from the repository root or from `scripts/`:

```bash
jupyter notebook scripts/vorticity.ipynb
```

For quick-start output, change the input paths in the configuration cell to
`ROOT / 'data/quickstart_modern'` and
`ROOT / 'output/quickstart_modern/...'`. The diagnostics notebook supports
inclusive `TIME_RANGE` and `FRAME_RANGE` selections, an optional rolling
average, and domain-area normalization.

## Movies

The movie scripts retain their command-line interfaces. Run any of them with
`--help` for all options. They accept explicit frames or an inclusive range
and support H.264 and H.265/HEVC output:

```bash
python scripts/movie_vorticity.py \
    --data-dir data/quickstart_modern \
    --parameters output/quickstart_modern/resolved_parameters.txt \
    --diagnostics output/quickstart_modern/diagnostics.csv \
    --quantity all --output figures/fields.mp4 --no-tex

python scripts/movie_spectra.py \
    --input output/quickstart_modern/spectra.csv \
    --start 1 --stop 100 --stride 2 \
    --output figures/spectra.mp4 --no-tex

python scripts/movie_fluxes.py \
    --input output/quickstart_modern/fluxes.csv \
    --segment-mean --codec h265 --output figures/fluxes.mp4 --no-tex
```

Omit `--frames`, `--start`, and `--stop` to animate every available frame.
The physical-field movie accepts `--quantity vorticity`, `u`, `v`, or `all`.
It uses one global color scale by default; `--color-scale first` avoids the
all-frame pre-scan and `--color-scale dynamic` rescales every frame.  Set the
output suffix to `.gif` to use the Pillow writer; `--codec` is then ignored.

## File summary

| Script | Output |
| --- | --- |
| `vorticity.ipynb` | Separate vorticity, `u`, and `v` PDFs |
| `spectra.ipynb` | Separate energy and enstrophy spectrum PDFs |
| `fluxes.ipynb` | Two-panel energy/enstrophy flux PDF |
| `diagnostics.ipynb` | Quantity and dissipation-rate PDFs |
| `movie_vorticity.py` | Vorticity and/or velocity MP4/GIF |
| `movie_spectra.py` | Energy/enstrophy spectrum MP4/GIF |
| `movie_fluxes.py` | Energy/enstrophy flux MP4/GIF |

`ns2d_plotting.py` contains the shared readers, frame selection, velocity
reconstruction, plotting style, and animation writer.
