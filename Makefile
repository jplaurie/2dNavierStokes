BUILD_DIR ?= build/release
BUILD_TYPE ?= Release
JOBS ?= 4

.PHONY: all cpu mpi cuda test clean configure

all: configure
	cmake --build $(BUILD_DIR) -j$(JOBS)

configure:
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)

cpu mpi cuda: configure
	cmake --build $(BUILD_DIR) --target navier_stokes_$@ -j$(JOBS)

test: all
	ctest --test-dir $(BUILD_DIR) --output-on-failure

clean:
	cmake -E remove_directory build
