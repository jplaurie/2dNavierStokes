# Backend benchmarks

This benchmark times complete ETD4-B timesteps on square grids with 3/2-rule padding. Each step
contains four nonlinear right-hand-side evaluations; each evaluation performs four inverse FFTs,
one physical-space Jacobian product, and one forward FFT.

The timed interval starts after construction, FFT planning, allocation, coefficient/state upload,
and warm-up. Synchronization is included at both timing boundaries. Diagnostics, output, and the
final GPU download are excluded.

Build and run all backends, then create the README plot:

```bash
cmake -S . -B build/bench -DCMAKE_BUILD_TYPE=Release
cmake --build build/bench --parallel
python3 benchmarks/run_benchmarks.py --overwrite
python3 benchmarks/plot_benchmarks.py
```

By default, CPU/OpenMP and MPI use one worker per detected physical core. MPI uses single-threaded
ranks, and `CPU serial` is linked to code compiled without OpenMP or threaded FFTW. The mixed CUDA
backend retains the spectral state, integrator stages, coefficients, forcing, and noise in FP64;
only the padded FFT fields and physical-space Jacobian product use FP32.

The plot's timing panel reports absolute ETD4 step time. Its speedup panel is normalized by the
single-core `CPU serial` backend, which is the conventional baseline for total parallel speedup.

Raw trials are stored in `results.csv`; exact machine, compiler, run settings, and timing exclusions
are in `system.json`. Override durations, resolutions, backends, CPU threads, or MPI ranks through
`run_benchmarks.py --help`. The runner uses only the Python standard library; plot generation also
requires Matplotlib.
