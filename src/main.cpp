#include "backend.hpp"
#include "parameters.hpp"
#include "solver.hpp"

#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>

#ifdef NS2D_HDF5_FAILURE_TEST
#include <hdf5.h>
#endif

int main(int argc, char **argv) {
    bool backendReady = false;
    try {
#ifdef NS2D_HDF5_FAILURE_TEST
        // Only the fault-injection test build disables this final cleanup retry.
        // HDF5 1.10 can crash in its atexit handler after persistent write errors.
        // Normal RAII cleanup and the production executable remain unchanged.
        if (H5dont_atexit() < 0)
            throw std::runtime_error("cannot disable HDF5 atexit in failure fixture");
#endif
        backendInitialize(argc, argv);
        backendReady = true;
        {
            const std::filesystem::path parameterFile = argc > 1 ? argv[1] : "params.txt";
            Parameters parameters = readParameters(parameterFile);
            auto backend = makeBackend(parameters);
            Solver solver(std::move(parameters), std::move(backend));
            solver.run();
        } // Destroy every FFT plan before cleaning up its FFT runtime.
        backendSaveWisdom();
        backendFinalize();
        return 0;
    } catch (const std::exception &error) {
        if (!backendReady || backendIsRoot())
            std::cerr << "error: " << error.what() << '\n';
        if (backendReady) {
            backendAbort(1);
            backendFinalize();
        }
        return 1;
    }
}
