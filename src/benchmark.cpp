#include "backend.hpp"
#include "parameters.hpp"
#include "solver.hpp"

#include <charconv>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
template <class Integer> Integer parseInteger(const char *text, const char *name) {
    Integer value{};
    const std::string_view input(text);
    const auto [end, error] = std::from_chars(input.data(), input.data() + input.size(), value);
    if (error != std::errc{} || end != input.data() + input.size())
        throw std::runtime_error(std::string("invalid ") + name + ": " + text);
    return value;
}
} // namespace

int main(int argc, char **argv) {
    bool backendReady = false;
    try {
        backendInitialize(argc, argv);
        backendReady = true;
        if (argc < 3 || argc > 5)
            throw std::runtime_error(
                "usage: benchmark <resolution> <measured-steps> [warmup-steps] [threads]");

        Parameters parameters;
        parameters.nx = parameters.ny = parseInteger<std::size_t>(argv[1], "resolution");
        const std::uint64_t measuredSteps =
            parseInteger<std::uint64_t>(argv[2], "measured step count");
        const std::uint64_t warmupSteps =
            argc >= 4 ? parseInteger<std::uint64_t>(argv[3], "warmup step count") : 2;
        parameters.threadCount = argc >= 5 ? parseInteger<int>(argv[4], "thread count") : 0;
        parameters.integrator = Integrator::etd4;
        parameters.timeStep = 1.0e-4;
        parameters.numberOfSteps = measuredSteps;
        parameters.outputIntervalSteps = measuredSteps;
        parameters.forcingEnabled = false;
        parameters.betaPlane = false;
        parameters.viscosity = 1.0e-6;
        parameters.viscosityOrder = 1.0;
        parameters.linearDrag = 0.0;
        parameters.randomSeed = 1;
        validateParameters(parameters);

        double elapsed = 0.0;
        {
            auto backend = makeBackend(parameters);
            Solver solver(parameters, std::move(backend));
            elapsed = solver.benchmark(warmupSteps, measuredSteps);
        } // Destroy FFT plans before finalizing their runtimes.
        if (backendIsRoot()) {
            std::cout << std::setprecision(17) << "BENCHMARK," << backendName() << ','
                      << parameters.nx << ',' << parameters.ny << ',' << measuredSteps << ','
                      << elapsed << ',' << elapsed / static_cast<double>(measuredSteps) << '\n';
        }
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
