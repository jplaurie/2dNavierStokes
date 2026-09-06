#!/usr/bin/env python3
"""End-to-end numerical and recovery checks; Python standard library only."""

import argparse
import cmath
import csv
import math
from pathlib import Path
import struct
import subprocess
import tempfile


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def parameters(folder, **changes):
    folder.mkdir(parents=True, exist_ok=True)
    p = dict(nx=16, ny=16, aspectRatio=1, timeStep=0.01,
             numberOfSteps=1, outputIntervalSteps=100000,
             integrator="etd4", forcingEnabled="false", viscosity=0.1,
             viscosityOrder=1, linearDrag=0, betaPlane="false",
             randomSeed=12345, threadCount=1,
             dataDirectory=folder / "data", outputDirectory=folder / "output")
    p.update(changes)
    (folder / "run.params").write_text("".join(f"{k} {v}\n" for k, v in p.items()))
    return folder


def run(command, folder, error=None):
    result = subprocess.run([*command, str(folder / "run.params")],
                            capture_output=True, text=True, timeout=120)
    if error is not None:
        require(result.returncode != 0 and error in result.stderr,
                f"Expected {error!r}:\n{result.stdout}\n{result.stderr}")
    else:
        require(result.returncode == 0, result.stdout + result.stderr)
    return result


def checkpoint(folder, frame=1):
    data = (folder / "data" / f"checkpoint_{frame:08d}.bin").read_bytes()
    nx, ny, count = struct.unpack("=QQQ", data[8:32])
    require(data[:8] == b"NS2DCP2\0" and count == ny * (nx // 2 + 1), "Checkpoint header")
    values = [complex(r, i) for r, i in struct.iter_unpack("=dd", data[32:])]
    require(all(math.isfinite(z.real) and math.isfinite(z.imag) for z in values), "Nonfinite checkpoint")
    return values


def field(path, nx, ny, function):
    path.write_text("".join(" ".join(f"{function(2*math.pi*x/nx, 2*math.pi*y/ny):.17g}"
                                    for x in range(nx)) + "\n" for y in range(ny)))
    return path


def difference(a, b):
    require(len(a) == len(b), "Field sizes differ")
    return max(abs(x - y) for x, y in zip(a, b))


def numerics(root, cpu):
    wave = field(root / "wave.dat", 16, 16, lambda x, y: math.cos(x+y))
    multi = field(root / "multi.dat", 16, 16,
                  lambda x, y: math.sin(x) + .5*math.cos(2*y) +
                  .3*math.sin(x+y) + .2*math.cos(2*x-y))
    reference = parameters(root / "reference", timeStep=.4/2048,
                           numberOfSteps=2048, initialConditionFile=multi,
                           betaPlane="true", beta=3)
    run(cpu, reference)
    expected = checkpoint(reference)
    for method, order in [("etd2", 2), ("etd3", 3), ("etd4", 4), ("rk2", 2)]:
        # A large beta*h deliberately lies outside the old explicit wave limit.
        for aspect in [1, 2]:
            folder = parameters(root / f"wave_{method}_{aspect}", integrator=method,
                                timeStep=.25, numberOfSteps=4, initialConditionFile=wave,
                                aspectRatio=aspect, betaPlane="true", beta=40)
            run(cpu, folder)
            k2 = 1 + 1/aspect**2
            exact = .5*cmath.exp(-.1*k2 + 40j/aspect/k2)
            require(abs(checkpoint(folder)[10] - exact) < 3e-13,
                    f"{method}: analytic beta wave propagation")
        if method != "rk2":
            folder = parameters(root / f"constant_force_{method}", integrator=method,
                                timeStep=.2, numberOfSteps=5, forcingEnabled="true",
                                forcingProfile="singleMode", forcingWavenumber=1,
                                forcingAmplitude=.3, betaPlane="true", beta=7)
            run(cpu, folder)
            linear = complex(-.2, 3.5)
            exact = -.3*(cmath.exp(linear)-1)/linear
            require(abs(checkpoint(folder)[10] - exact) < 3e-13, f"{method}: constant forcing with complex L")
        errors = []
        for steps in [8, 16, 32]:
            folder = parameters(root / f"convergence_{method}_{steps}", integrator=method,
                                timeStep=.4/steps, numberOfSteps=steps,
                                initialConditionFile=multi, betaPlane="true", beta=3)
            run(cpu, folder)
            errors.append(difference(checkpoint(folder), expected))
        observed = math.log2(errors[-2] / errors[-1])
        require(order-.35 < observed < order+.35, f"{method}: convergence order {observed}, errors {errors}")
        print(f"{method}: nonlinear convergence order {observed:.3f}")
        # Exact variance for an isolated, equally damped ring: N(w)=0.
        for damping in [0, 10]:
            folder = parameters(root / f"noise_{method}_{damping}", integrator=method,
                                timeStep=.1, viscosity=0, linearDrag=damping,
                                forcingEnabled="true", forcingWavenumber=1,
                                forcingWidth=.01, forcingAmplitude=1)
            run(cpu, folder)
        a = checkpoint(root / f"noise_{method}_0")
        b = checkpoint(root / f"noise_{method}_10")
        ratio = sum(abs(z)**2 for z in b) / sum(abs(z)**2 for z in a)
        require(abs(ratio - (-math.expm1(-2)/2)) < 2e-14, f"{method}: stochastic covariance {ratio}")
    # Positive stochastic wavenumbers need not fit an integer; used to trigger UB.
    folder = parameters(root / "large_stochastic_wave", forcingEnabled="true",
                        forcingWavenumber=1e300, forcingWidth=1.1e300)
    run(cpu, folder)
    # Lawson RK2 is evaluated without inverse exponentials, even for stiff damping.
    folder = parameters(root / "stiff_rk2", integrator="rk2", viscosity=0,
                        linearDrag=10000, timeStep=1, initialConditionFile=wave)
    run(cpu, folder)
    require(max(abs(z) for z in checkpoint(folder)) == 0, "Stiff RK2 decay")
    folder = parameters(root / "overflowing_step", viscosity=0, linearDrag=10,
                        timeStep=1e308, integrator="etd4")
    run(cpu, folder, error="time integration coefficients contains a non-finite coefficient")


def output(root, cpu):
    initial = field(root / "small.dat", 4, 4, lambda x, y: math.sin(y))
    folder = parameters(root / "small", nx=4, ny=4, initialConditionFile=initial,
                        writeModeDiagnostics="true")
    run(cpu, folder)
    with (folder / "output/modes.csv").open() as stream:
        row = next(csv.DictReader(stream))
    require(row["omega_0_3_real"] == "" and row["omega_2_1_real"] == "", "Unavailable modes must be blank")
    require(abs(float(row["omega_0_1_imag"]) + .5*math.exp(-.001)) < 1e-13, "Available mode changed")

    settings = dict(numberOfSteps=1, outputIntervalSteps=1, forcingEnabled="true",
                    forcingWavenumber=2, forcingWidth=.6, writeModeDiagnostics="true")
    folder = parameters(root / "recovery", **settings)
    run(cpu, folder)
    committed = {name: (folder / "output" / name).read_bytes()
                 for name in ["diagnostics.csv", "spectra.csv", "fluxes.csv", "modes.csv"]}
    # Force a real failure after CSVs, snapshot and binary checkpoint are written,
    # but before the atomic metadata rename that commits the frame.
    (folder / "data/restart_state.tmp").mkdir()
    run(cpu, folder, error="cannot write restart metadata")
    journal = (folder / "data/output_transaction.txt").read_bytes()
    require((folder / "data/checkpoint_00000002.bin").exists(), "Fault did not reach checkpoint write")
    result = run(cpu, folder)
    require("recovered interrupted output frame 2" in result.stdout, "Interrupted frame not recovered")
    for name, prefix in committed.items():
        content = (folder / "output" / name).read_bytes()
        require(content.startswith(prefix), f"Committed {name} changed during recovery")
        with (folder / "output" / name).open() as stream:
            rows = list(csv.DictReader(stream))
        frames = [int(row["frame"]) for row in rows]
        require(frames.count(1) == frames.count(2), f"Duplicated or incomplete {name}")
    full = parameters(root / "full", **(settings | {"numberOfSteps": 2}))
    run(cpu, full)
    require(checkpoint(folder, 2) == checkpoint(full, 2), "Recovery changed stochastic trajectory")

    # A crash immediately after metadata commit must keep the committed frame.
    (folder / "data/output_transaction.txt").write_bytes(journal)
    saved = (folder / "data/checkpoint_00000002.bin").read_bytes()
    run(cpu, folder)
    require((folder / "data/checkpoint_00000002.bin").read_bytes() == saved, "Committed frame was rolled back")

    # Future collisions are rejected before appending diagnostics for that frame.
    collision = parameters(root / "collision", **settings)
    run(cpu, collision)
    (collision / "data/vorticity_00000003.dat").write_text("existing data\n")
    parameters(collision, **(settings | {"numberOfSteps": 2}))
    run(cpu, collision, error="refusing to overwrite output frame file")
    with (collision / "output/diagnostics.csv").open() as stream:
        require([row["frame"] for row in csv.DictReader(stream)] == ["1", "2"], "Collision appended orphan diagnostics")
    parameters(collision, **(settings | {"overwriteOutput": "true", "linearDrag": .3}))
    run(cpu, collision)
    history = sorted((collision / "output/segments").glob("segment_*/resolved_parameters.txt"))
    require(len(history) == 3, "Missing invocation history")
    require("linearDrag 0\n" in history[0].read_text(), "Old parameters lost")
    require("linearDrag 0.29999999999999999\n" in history[-1].read_text(), "New parameters not recorded")
    require(f"outputDirectory {collision / 'output'}\n" in history[-1].read_text(), "Archived parameter path changed")
    require(not (collision / "data/output_transaction.txt").exists(), "Successful frame left a journal")
    fresh = parameters(root / "fresh_recovery", **settings)
    (fresh / "data").mkdir()
    (fresh / "data/restart_state.tmp").mkdir()
    run(cpu, fresh, error="cannot write restart metadata")
    require("recovered interrupted output frame 0" in run(cpu, fresh).stdout, "Initial frame not recovered")
    require(checkpoint(fresh) == checkpoint(full), "Fresh recovery changed initial trajectory")

    # Old checkpoint states remain readable, with an explicit numerical-version notice.
    metadata = fresh / "data/restart_state.txt"
    metadata.write_text(metadata.read_text().replace("ns2d_restart_v3", "ns2d_restart_v2")
                        .replace("numericsVersion 3\n", ""))
    require("loading an older checkpoint" in run(cpu, fresh).stdout, "Missing migration notice")
    print("Output recovery, collision safety, mode availability and parameter histories passed")


def backend(root, cpu, candidate):
    initial = field(root / "initial.dat", 32, 48,
                    lambda x, y: math.sin(x) + .4*math.cos(2*y) + .2*math.sin(3*x-y))
    for method in ["etd2", "etd3", "etd4", "rk2"]:
        for profile in ["disabled", "annulus", "exponential", "singleMode"]:
            settings = dict(nx=32, ny=48, aspectRatio=2, integrator=method,
                            timeStep=.005, numberOfSteps=4, outputIntervalSteps=2,
                            initialConditionFile=initial, betaPlane="true", beta=7,
                            threadCount=2, forcingEnabled=str(profile != "disabled").lower(),
                            forcingProfile=profile if profile != "disabled" else "annulus",
                            forcingWavenumber=3, forcingWidth=.6, forcingAmplitude=.02)
            reference = parameters(root / f"cpu_{method}_{profile}", **settings)
            comparison = parameters(root / f"candidate_{method}_{profile}", **settings)
            run(cpu, reference)
            run(candidate, comparison)
            require(difference(checkpoint(reference, 2), checkpoint(comparison, 2)) < 2e-12,
                    f"Backend mismatch: {method}, {profile}")
            if profile == "annulus":
                split = parameters(root / f"split_{method}", **(settings | {"numberOfSteps": 2}))
                run(candidate, split)
                run(candidate, split)
                require(checkpoint(split, 2) == checkpoint(comparison, 2), f"Backend restart mismatch: {method}")
    print("All integrators and forcing profiles agree across backends; stochastic restarts match")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["numerics", "output", "backend"])
    parser.add_argument("--cpu", required=True)
    parser.add_argument("--candidate")
    parser.add_argument("--mpiexec")
    parser.add_argument("--mpi-numproc-flag", default="-n")
    args = parser.parse_args()
    cpu = [args.cpu]
    candidate = [args.candidate]
    if args.mpiexec:
        candidate = [args.mpiexec, args.mpi_numproc_flag, "2", *candidate]
    with tempfile.TemporaryDirectory(prefix="ns2d-regression-") as temporary:
        root = Path(temporary)
        if args.mode == "numerics":
            numerics(root, cpu)
        elif args.mode == "output":
            output(root, cpu)
        else:
            backend(root, cpu, candidate)


if __name__ == "__main__":
    main()
