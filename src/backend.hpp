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
  virtual ~NonlinearBackend() = default;
  virtual void evaluate(const SpectralField &vorticity,
                        SpectralField &result) = 0;
  [[nodiscard]] virtual bool deviceTimeStepping() const { return false; }
  virtual void initializeTimeStepping(const IntegrationCoefficients &,
                                      const std::vector<double> &,
                                      const SpectralField &) {
    throw std::logic_error("backend has no device time integrator");
  }
  virtual void advance(const SpectralField &) {
    throw std::logic_error("backend has no device time integrator");
  }
  virtual void downloadState(SpectralField &) {
    throw std::logic_error("backend has no device state");
  }
};

void backendInitialize(int &argc, char **&argv);
void backendFinalize();
void backendAbort(int exitCode);
void backendBarrier();
[[nodiscard]] bool backendIsRoot();
[[nodiscard]] const char *backendName();
std::uint64_t backendSynchronizeSeed(std::uint64_t seed);
std::unique_ptr<NonlinearBackend> makeBackend(const Parameters &parameters);
