#pragma once

#include "integrator.hpp"
#include "parameters.hpp"

#include <complex>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

using Complex = std::complex<double>;
using SpectralField = std::vector<Complex>;

class NonlinearBackend {
  public:
    enum class NoiseLayout { fullField, forcedModes };

    virtual ~NonlinearBackend() = default;
    virtual void evaluate(const SpectralField &vorticity, SpectralField &result) = 0;
    [[nodiscard]] virtual NoiseLayout noiseLayout() const = 0;
    virtual void initializeTimeStepping(const IntegrationCoefficients &coefficients,
                                        const std::vector<double> &deterministicForcing,
                                        const std::vector<std::size_t> &noiseIndices,
                                        const SpectralField &state) = 0;
    virtual void advanceTimeStep(const SpectralField &noise) = 0;
    // Distributed backends collect onto rank zero and leave the destination
    // empty on other ranks.
    virtual void downloadState(SpectralField &state) = 0;
    virtual void evaluateCurrent(SpectralField &nonlinearTerm) = 0;
};

void backendInitialize(int &argc, char **&argv);
void backendFinalize();
void backendAbort(int exitCode);
void backendBarrier();
[[nodiscard]] bool backendIsRoot();
[[nodiscard]] const char *backendName();
std::uint64_t backendSynchronizeSeed(std::uint64_t seed);
std::unique_ptr<NonlinearBackend> makeBackend(const Parameters &parameters);
