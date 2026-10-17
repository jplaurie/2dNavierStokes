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
    // rootForcedModes supplies compact increments on rank zero only; the
    // distributed backend sends each rank the modes it owns.
    enum class NoiseLayout { fullField, forcedModes, rootForcedModes };

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
// Save plans before cleanup so an I/O error can be reported without throwing
// from finalization or retrying the failed save in an exception handler.
void backendSaveWisdom();
void backendFinalize() noexcept;
void backendAbort(int exitCode);
void backendBarrier();
[[nodiscard]] bool backendIsRoot();
[[nodiscard]] const char *backendName();
std::uint64_t backendSynchronizeSeed(std::uint64_t seed);
std::unique_ptr<NonlinearBackend> makeBackend(const Parameters &parameters);
