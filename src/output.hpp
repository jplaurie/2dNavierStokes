#pragma once

#include "backend.hpp"
#include "fftw_utils.hpp"

#include <cstdint>
#include <string>
#include <vector>

struct RestartState {
  double time = 0.0;
  std::uint64_t frame = 0;
  std::uint64_t randomSeed = 0;
  SpectralField vorticity;
  std::string randomEngineState;
  std::string randomDistributionState;
  bool restarting = false;
};

struct DiagnosticsAverages {
  std::vector<double> energySpectrum;
  std::vector<double> enstrophySpectrum;
  std::vector<double> energyFlux;
  std::vector<double> enstrophyFlux;
  std::size_t count = 0;
};

// The metadata rename commits a frame. The journal permits rollback of CSV
// appends and uncommitted files after an exception or process interruption.
bool recoverOutputTransaction(const Parameters &parameters);
void beginOutputTransaction(const Parameters &parameters, std::uint64_t frame);
void finishOutputTransaction(const Parameters &parameters);
void writeRunRecords(const Parameters &parameters, const std::string &backend,
                     double time, std::uint64_t frame,
                     const std::vector<double> &amplitude,
                     std::size_t forcedModes, double energyInjectionCoefficient,
                     double enstrophyInjectionCoefficient);

RestartState readRestart(const Parameters &parameters, BaseTransform &transform,
                         bool isRoot);
void prepareOutputFiles(const Parameters &parameters, bool restarting,
                        std::uint64_t restartFrame);
void writeVorticity(const Parameters &parameters, BaseTransform &transform,
                    const SpectralField &vorticity, std::uint64_t frame);
double writeDiagnostics(const Parameters &parameters, double time,
                        std::uint64_t frame, const SpectralField &vorticity,
                        const SpectralField &nonlinear,
                        DiagnosticsAverages &averages);
void writeRestart(const Parameters &parameters, double time,
                  std::uint64_t frame, const SpectralField &vorticity,
                  const std::string &randomEngineState,
                  const std::string &randomDistributionState);
void writeForcingFiles(const Parameters &parameters,
                       const std::vector<double> &amplitude,
                       std::size_t forcedModes,
                       double energyInjectionCoefficient,
                       double enstrophyInjectionCoefficient);
