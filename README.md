# 2D Navier–Stokes pseudo-spectral solver

A C++20 solver for the doubly periodic two-dimensional vorticity equation. It
uses a dealiased pseudo-spectral method, supports deterministic and stochastic
forcing, and writes restartable simulations with CSV diagnostics.

Three executables share the same model, parameter format, integration methods,
and output format:

| Executable | Backend | Use case |
| --- | --- | --- |
| `navier_stokes_cpu` | FFTW, with optional OpenMP | Standard shared-memory runs |
| `navier_stokes_mpi` | FFTW-MPI, with optional OpenMP | Distributed-memory runs |
| `navier_stokes_cuda` | CUDA and cuFFT | NVIDIA GPU runs |

The current solver is self-contained; the original source trees are kept in
[`archived/`](archived/) for reference.

## Equation, forcing, and dissipation

The solver evolves the scalar vorticity $\omega(\boldsymbol{x},t)$ on the
doubly periodic domain $L_x=2\pi A_r$, $L_y=2\pi$, where $A_r$ is
`aspectRatio`. The complete equation represented by the parameter file is

```math
\partial_t\omega
+\boldsymbol{u}\cdot\nabla\omega
+\beta\,\partial_x\psi
=-\nu(-\Delta)^p\omega
-\alpha(-\Delta)^q\omega
+F(\boldsymbol{x},t),
```

with the streamfunction and incompressible velocity defined by

```math
\omega=\Delta\psi,
\qquad
\boldsymbol{u}=(-\partial_y\psi,\,\partial_x\psi),
\qquad
\nabla\cdot\boldsymbol{u}=0.
```

The beta-plane term is present only when `betaPlane true`; its coefficient is
`beta`. In Fourier space, the linear part is

```math
\partial_t\widehat\omega_{\boldsymbol{k}}\big|_{\mathrm{linear}}
=\left[
  i\beta\frac{k_x}{|\boldsymbol{k}|^2}
  -\nu|\boldsymbol{k}|^{2p}
  -\alpha|\boldsymbol{k}|^{2q}
 \right]\widehat\omega_{\boldsymbol{k}},
\qquad \boldsymbol{k}\ne\boldsymbol{0}.
```

Here $(\nu,p)$ are `viscosity` and `viscosityOrder`, and
$(\alpha,q)$ are `linearDrag` and `dragOrder`. Thus the same implementation
covers ordinary or hyperviscosity through $p$, and linear or scale-selective
drag through $q$. The zero vorticity mode is removed.

With $F=\nu=\alpha=0$, nonlinear advection and the beta-plane term conserve
kinetic energy and enstrophy:

```math
E=\frac12\int_\Omega|\boldsymbol{u}|^2\,d^2x
=\frac12\sum_{\boldsymbol{k}\ne0}
  \frac{|\widehat\omega_{\boldsymbol{k}}|^2}{|\boldsymbol{k}|^2},
\qquad
Z=\frac12\int_\Omega\omega^2\,d^2x
=\frac12\sum_{\boldsymbol{k}}|\widehat\omega_{\boldsymbol{k}}|^2,
```

up to the Fourier-normalization/domain-area convention used by the diagnostic
files.

### Forcing profiles

For `annulus` and `exponential`, the forcing is real-valued Gaussian
white-in-time noise. In spectral notation,

```math
d\widehat\omega_{\boldsymbol{k}}\big|_{\mathrm{force}}
=f(|\boldsymbol{k}|)\,dW_{\boldsymbol{k}},
\qquad
\mathbb{E}[dW_{\boldsymbol{k}}]=0,
\qquad
\mathbb{E}[dW_{\boldsymbol{k}}dW_{\boldsymbol{k}'}^*]
=\delta_{\boldsymbol{k}\boldsymbol{k}'}\,dt,
\qquad
dW_{-\boldsymbol{k}}=dW_{\boldsymbol{k}}^*,
```

with envelopes

```math
\begin{aligned}
f_{\mathrm{annulus}}(k)
  &=A\,\mathbf{1}_{\{\lvert k-k_f\rvert<\Delta k\}},\\
f_{\mathrm{exponential}}(k)
  &=A\left(\frac{k}{k_f}\right)^s
    \exp\!\left[-\left(\frac{k}{k_f}\right)^s\right].
\end{aligned}
```

The symbols $A,k_f,\Delta k,s$ map to `forcingAmplitude`,
`forcingWavenumber`, `forcingWidth`, and `forcingShapeOrder`. A positive
`targetEnergyInjectionRate` rescales the stochastic spectrum to the requested
coefficient

```math
\varepsilon=\frac12\sum_{\boldsymbol{k}\ne0}
\frac{|f(|\boldsymbol{k}|)|^2}{|\boldsymbol{k}|^2},
```

using the solver's real-transform multiplicities. `singleMode` is
deterministic and acts on the two independent stored modes corresponding to
$(m,m)$ and $(m,-m)$, with signs chosen to preserve a real vorticity
field and $m=\texttt{forcingWavenumber}$.

### Parameter-symbol map and discretization

| Symbol | Parameter key | Meaning |
| --- | --- | --- |
| $N_x,N_y$ | `nx`, `ny` | Physical-grid dimensions |
| $A_r$ | `aspectRatio` | Domain aspect ratio $L_x/L_y$ |
| $\Delta t$ | `timeStep` | Fixed timestep |
| $\beta$ | `beta` | Beta-plane coefficient; gated by `betaPlane` |
| $\nu,p$ | `viscosity`, `viscosityOrder` | Small-scale damping coefficient and power |
| $\alpha,q$ | `linearDrag`, `dragOrder` | Large-scale damping coefficient and power |

Advection is evaluated on a fully padded 3/2-rule grid. The zero mode and the
even-grid Nyquist lines are removed. Available fixed-step integrators are
ETDRK2 (`etd2`), ETDRK3 (`etd3`), ETDRK4-B (`etd4`), and second-order
integrating-factor Runge–Kutta (`rk2`). Linear damping and beta-plane
propagation are integrated analytically.

## Requirements

The CPU build requires:

- CMake 3.20+
- a C++20 compiler
- FFTW3 development headers and library

OpenMP and FFTW's threads library are optional. Python 3 enables the complete
regression suite. MPI runs also need MPI and FFTW-MPI; CUDA runs need the NVIDIA
CUDA Toolkit and cuFFT, plus an NVIDIA GPU at run time.

On Arch Linux, the relevant packages are typically `fftw`, `openmpi`,
`fftw-openmpi`, and `cuda`.

## Build and test

Start with the portable CPU build:

```bash
cmake -S . -B build/cpu -DCMAKE_BUILD_TYPE=Release \
  -DNS2D_MPI=OFF -DNS2D_CUDA=OFF
cmake --build build/cpu -j
ctest --test-dir build/cpu --output-on-failure
```

To build every backend supported by the local toolchain, leave the optional
backends enabled:

```bash
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/release -j
ctest --test-dir build/release --output-on-failure
```

CMake omits the MPI executable when MPI or FFTW-MPI is unavailable, and omits
the CUDA executable when no CUDA compiler is found. Useful options are:

```text
-DNS2D_OPENMP=OFF
-DNS2D_MPI=OFF
-DNS2D_CUDA=OFF
-DNS2D_CUDA_ARCHITECTURES=<CUDA architecture>
-DNS2D_BACKEND_TESTS=ON
```

`NS2D_BACKEND_TESTS` adds MPI and CUDA comparisons when those executables can
run locally. The standard suite checks numerical kernels, outputs, restart
recovery, and parameter handling. Convenience targets are `make cpu`,
`make mpi`, `make cuda`, and `make test`; set `BUILD_DIR` if needed.

## Quick start

[`examples/quickstart.params`](examples/quickstart.params) is a small,
repeatable run. From the repository root, run:

```bash
./build/cpu/navier_stokes_cpu examples/quickstart.params
```

It writes snapshots and checkpoints to `data/quickstart_modern/`, and CSV
diagnostics to `output/quickstart_modern/`. Remove those directories or change
the paths before rerunning a fresh quick-start case.

The root [`params.txt`](params.txt) is a production-scale example. Copy it and
adjust the grid, time step, output directories, and run length before use; its
default step count is intentionally very large.

All executables take an optional parameter-file path. Without one, they read
`params.txt` in the current directory:

```bash
./build/release/navier_stokes_cpu run.params

OMP_NUM_THREADS=4 \
  mpirun -n 2 ./build/release/navier_stokes_mpi run.params

./build/release/navier_stokes_cuda run.params
```

Set `threadCount` in the parameter file to select host OpenMP threads per
process; `0` uses the OpenMP runtime default. CPU and MPI FFTs use the same
count when FFTW threads support is available. For MPI, plan for
`ranks × threadCount` CPU cores. Only rank zero writes files.

## Parameter files

Each nonempty line is `key value` or `key = value`; `#` begins a comment.
Keys are case-sensitive. Invalid keys, values, or extra fields stop the run.
Paths are interpreted relative to the directory from which the executable is
launched.

| Key | Purpose |
| --- | --- |
| `nx`, `ny` | Physical-grid dimensions. Both must be multiples of four and at least four. |
| `aspectRatio` | Positive `Lx / (2*pi)` domain aspect ratio. |
| `timeStep` | Positive fixed time step. |
| `numberOfSteps` | Additional steps to perform in this invocation. |
| `outputIntervalSteps` | Save every this many steps; the final step is always saved. |
| `integrator` | One of `etd2`, `etd3`, `etd4`, or `rk2`. |
| `betaPlane`, `beta` | Enable and set the beta-plane coefficient. |
| `viscosity`, `viscosityOrder` | Viscosity coefficient and spectral power. |
| `linearDrag`, `dragOrder` | Drag coefficient and spectral power. |
| `forcingEnabled` | Enable or disable forcing. |
| `forcingProfile` | `annulus`, `exponential`, or `singleMode`. |
| `forcingWavenumber` | Central physical wavenumber, or integer mode index for `singleMode`. |
| `forcingWidth` | Half-width of `annulus` forcing. |
| `forcingAmplitude` | Forcing amplitude before stochastic normalization. |
| `forcingShapeOrder` | Positive exponent for `exponential` forcing. |
| `targetEnergyInjectionRate` | Positive value normalizes stochastic forcing; `0` leaves its amplitude unchanged. |
| `randomSeed` | Reproducible 64-bit seed; `0` chooses and records a time-based seed. |
| `writeModeDiagnostics` | Write selected Fourier modes to `modes.csv`. |
| `threadCount` | Host threads per process; `0` uses the OpenMP default. |
| `overwriteOutput` | Allow replacement of an existing frame. It does not disable automatic restart. |
| `initialConditionFile` | Optional whitespace-delimited `ny` × `nx` vorticity matrix. |
| `dataDirectory` | Snapshots, checkpoints, and restart metadata. |
| `outputDirectory` | CSV diagnostics and resolved run configuration. |

Booleans accept `true`/`false` or `1`/`0`. The solver writes the validated
configuration, selected backend, and actual random seed to
`outputDirectory/resolved_parameters.txt`.

`annulus` and `exponential` are Gaussian white-in-time spectral forcing. A
positive `targetEnergyInjectionRate` normalizes their amplitude. `singleMode`
is deterministic forcing at the `(m, +m)` and `(m, -m)` modes, where `m` is
`forcingWavenumber`; it does not accept `targetEnergyInjectionRate`.

## Output and restart

Each fresh run saves frame zero, then saves at the requested cadence and at the
final step. The initial frame has no diagnostics row because diagnostics are
computed after an integration step.

| Location | Contents |
| --- | --- |
| `dataDirectory/vorticity_NNNNNNNN.dat` | Whitespace-delimited physical vorticity matrix (`ny` rows by `nx` columns). |
| `dataDirectory/checkpoint_NNNNNNNN.bin` | Binary normalized spectral state for exact restart on compatible machines. |
| `dataDirectory/restart_state.txt` | Latest committed time, frame, grid identity, and random-generator state. |
| `outputDirectory/diagnostics.csv` | Time, frame, energy, enstrophy, and damping rates. |
| `outputDirectory/spectra.csv` | Energy and enstrophy spectra by radial shell. |
| `outputDirectory/fluxes.csv` | Energy and enstrophy fluxes by radial shell. |
| `outputDirectory/modes.csv` | Optional selected complex Fourier modes. |
| `outputDirectory/forcing_summary.csv` | Forcing type, forced-mode count, and injection coefficients. |
| `outputDirectory/forcing_spectrum.csv` | Spectral forcing amplitude for each stored mode. |
| `outputDirectory/segments/` | Per-invocation parameter and forcing records. |

If `restart_state.txt` exists, the solver resumes automatically from its
checkpoint. `numberOfSteps` then means additional steps, CSV files are
appended, and frame numbering continues. Matching `nx`, `ny`, and
`aspectRatio` are required; other physical settings can be changed between
invocations. A split stochastic run reproduces an uninterrupted run on the
same backend, hardware, FFT library, compiler, and thread/rank configuration.

Output frames are journaled and committed atomically. On the next run, an
interrupted frame is rolled back automatically. For a new simulation, use new
`dataDirectory` and `outputDirectory` paths. `overwriteOutput` permits file
replacement, but does not turn a detected restart into a fresh run.

## Plotting and movies

The [`scripts/`](scripts/) directory contains Jupyter plotting notebooks for
vorticity, the separately reconstructed `u` and `v` velocity fields, spectra,
fluxes, and time diagnostics. It also contains command-line MP4/GIF movie
generators for the physical fields, spectra, and fluxes. The notebooks support
single frames, multiple frames, and frame averages and write publication-ready
PDF figures. Movie output supports H.264 and H.265/HEVC through ffmpeg. See
[`scripts/README.md`](scripts/README.md) for dependencies, configuration, and
examples.

## Code structure

```text
src/
  main.cpp                    shared executable entry point
  parameters.cpp/.hpp         parse, validate, and record run settings
  spectral.cpp/.hpp           Fourier indexing and reality constraints
  fftw_utils.cpp/.hpp         base-grid FFTW transforms for I/O
  solver.cpp/.hpp             forcing, linear operator, and time stepping
  integrator.hpp              shared CPU/CUDA integration formulas
  output.cpp/.hpp             diagnostics, snapshots, and checkpoints
  output_transaction.cpp      atomic output recovery and run history
  backend.hpp                 common nonlinear-backend interface
  backend_cpu.cpp             FFTW/OpenMP advection backend
  backend_mpi.cpp             FFTW-MPI/OpenMP advection backend
  backend_cuda.cu             CUDA/cuFFT backend and GPU time stepping
examples/
  quickstart.params           small reproducible example
tests/
  numerics.cpp                numerical unit tests
  regression.py               output, restart, and backend regression tests
scripts/
  ns2d_plotting.py            shared readers, styling, and velocity recovery
  *.ipynb                     physical and diagnostic PDF plotting notebooks
  movie_*.py                  MP4/GIF field, spectrum, and flux movies
```

The `Solver` owns the shared simulation state and delegates only nonlinear
advection to the selected backend. This keeps the CPU, MPI, and CUDA programs
on the same parameter, forcing, integration, diagnostic, and restart paths.
