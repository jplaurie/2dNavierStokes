#!/usr/bin/env python3
"""Exercise HDF5 snapshots, export formats, and HDF5 initial conditions."""

import argparse
import math
import subprocess
import sys
import tempfile
from pathlib import Path


def run(command):
    result = subprocess.run([str(value) for value in command], text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"command failed: {' '.join(map(str, command))}\n"
                           f"{result.stdout}\n{result.stderr}")


def require_failure(command, expected):
    result = subprocess.run([str(value) for value in command], text=True, capture_output=True)
    if result.returncode == 0 or expected not in result.stderr:
        raise RuntimeError(f"command did not fail as expected: {' '.join(map(str, command))}\n"
                           f"{result.stdout}\n{result.stderr}")


def numbers(path):
    return [float(value) for value in path.read_text().split()]


def check_python_reader(hdf5_field, text_field):
    """Exercise the optional plotting reader when its Python stack is installed."""
    try:
        import h5py  # noqa: F401
        import matplotlib  # noqa: F401
        import numpy as np
    except ImportError:
        print("Python HDF5 plotting check skipped (h5py/NumPy/Matplotlib unavailable)")
        return

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    from ns2d_plotting import (  # pylint: disable=import-outside-toplevel
        discover_vorticity,
        read_vorticity,
        read_vorticity_metadata,
    )

    values = read_vorticity(hdf5_field)
    expected = np.asarray(numbers(text_field)).reshape(8, 12)
    assert values.shape == (8, 12)
    assert np.max(np.abs(values - expected)) < 1e-12
    metadata = read_vorticity_metadata(hdf5_field)
    assert metadata["nx"] == 12 and metadata["ny"] == 8
    assert abs(metadata["length_x"] - 2.0 * math.pi * 1.3) < 1e-12
    assert discover_vorticity(hdf5_field.parent)[0] == text_field


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", required=True)
    parser.add_argument("--exporter", required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="ns2d-hdf5-") as temporary:
        root = Path(temporary)
        initial = root / "initial.dat"
        with initial.open("w") as stream:
            for y in range(8):
                row = [f"{math.sin(2*math.pi*x/12) + .2*math.cos(2*math.pi*y/8):.17g}"
                       for x in range(12)]
                stream.write(" ".join(row) + "\n")
        params = root / "run.params"
        params.write_text(
            "nx 12\nny 8\naspectRatio 1.3\ntimeStep 0.001\nnumberOfSteps 1\n"
            "outputIntervalSteps 1\nintegrator etd4\nviscosity 0.1\nviscosityOrder 1\n"
            "linearDrag 0\nforcingEnabled false\nthreadCount 1\n"
            "fieldOutputFormat both\nhdf5CompressionLevel 1\n"
            f"fftwPlanning measure\nfftwWisdomFile {root / 'fftw.wisdom'}\n"
            f"initialConditionFile {initial}\ndataDirectory {root / 'data'}\n"
            f"outputDirectory {root / 'output'}\n"
        )
        run([args.cpu, params])
        assert (root / "fftw.wisdom").exists()
        for frame in (0, 1):
            stem = f"vorticity_{frame:08d}"
            text_field = root / "data" / f"{stem}.dat"
            hdf5_field = root / "data" / f"{stem}.h5"
            exported = root / f"{stem}_exported.dat"
            assert text_field.exists() and hdf5_field.exists()
            run([args.exporter, hdf5_field, exported])
            expected, actual = numbers(text_field), numbers(exported)
            assert len(expected) == len(actual) == 12 * 8
            assert max(abs(a - b) for a, b in zip(expected, actual)) < 1e-12
            require_failure([args.exporter, hdf5_field, exported, "--format"],
                            "usage: ns2d_hdf5_export")
            if frame == 0:
                check_python_reader(hdf5_field, text_field)

        table = root / "gnuplot.dat"
        run([args.exporter, root / "data/vorticity_00000001.h5", table,
             "--format", "gnuplot"])
        rows = [line.split() for line in table.read_text().splitlines()
                if line and not line.startswith("#")]
        assert len(rows) == 12 * 8 and all(len(row) == 3 for row in rows)
        run([args.cpu, params])
        assert (root / "data/vorticity_00000002.h5").exists()

        restart_root = root / "from_hdf5"
        restart_params = root / "from_hdf5.params"
        restart_params.write_text(
            "nx 12\nny 8\naspectRatio 1.3\ntimeStep 0.001\nnumberOfSteps 1\n"
            "outputIntervalSteps 1\nintegrator etd2\nviscosity 0.1\nviscosityOrder 1\n"
            "linearDrag 0\nforcingEnabled false\nthreadCount 1\nfieldOutputFormat hdf5\n"
            f"initialConditionFile {root / 'data/vorticity_00000001.h5'}\n"
            f"dataDirectory {restart_root / 'data'}\n"
            f"outputDirectory {restart_root / 'output'}\n"
        )
        run([args.cpu, restart_params])
        assert (restart_root / "data/vorticity_00000000.h5").exists()
        assert not (restart_root / "data/vorticity_00000000.dat").exists()
        print("HDF5 snapshots, exports, compression, and initial-condition input passed")


if __name__ == "__main__":
    main()
