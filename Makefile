BUILD_DIR ?= build/release
BUILD_TYPE ?= Release
JOBS ?= 4

.PHONY: all cpu cpu-serial mpi cuda cuda-mixed benchmark-backends test clean configure

all: configure
	cmake --build $(BUILD_DIR) -j$(JOBS)

configure:
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)

cpu mpi cuda: configure
	cmake --build $(BUILD_DIR) --target navier_stokes_$@ -j$(JOBS)

cpu-serial: configure
	cmake --build $(BUILD_DIR) --target navier_stokes_cpu_serial -j$(JOBS)

cuda-mixed: configure
	cmake --build $(BUILD_DIR) --target navier_stokes_cuda_mixed -j$(JOBS)

benchmark-backends: configure
	cmake --build $(BUILD_DIR) --target ns2d_benchmark_cpu_serial ns2d_benchmark_cpu \
		ns2d_benchmark_mpi ns2d_benchmark_cuda ns2d_benchmark_cuda_mixed -j$(JOBS)

test: all
	ctest --test-dir $(BUILD_DIR) --output-on-failure

clean:
	cmake -E remove_directory build
