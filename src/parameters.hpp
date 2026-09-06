#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

enum class Integrator { etd2, etd3, etd4, integratingFactorRk2 };
enum class ForcingProfile { annulus, exponential, singleMode };

struct Parameters {
  std::size_t nx = 1024;
  std::size_t ny = 1024;
  double aspectRatio = 1.0;
  double timeStep = 1.0e-4;
  std::uint64_t numberOfSteps = 9'999'999'999ULL;
  std::uint64_t outputIntervalSteps = 100;

  Integrator integrator = Integrator::etd4;
  bool betaPlane = false;
  double beta = 36.0;
  double viscosity = 1.0e-41;
  double viscosityOrder = 8.0;
  double linearDrag = 1.0e-4;
  double dragOrder = 0.0;

  bool forcingEnabled = true;
  ForcingProfile forcingProfile = ForcingProfile::annulus;
  double forcingWavenumber = 256.0;
  double forcingWidth = 1.0;
  double forcingAmplitude = 0.2;
  double forcingShapeOrder = 4.0;
  double targetEnergyInjectionRate = 0.0;
  std::uint64_t randomSeed = 0;

  bool writeModeDiagnostics = false;
  int threadCount = 0;
  bool overwriteOutput = false;
  std::filesystem::path initialConditionFile;
  std::filesystem::path dataDirectory = "data";
  std::filesystem::path outputDirectory = "output";

  [[nodiscard]] std::size_t nxf() const { return nx / 2 + 1; }
  [[nodiscard]] std::size_t mx() const { return 3 * nx / 2; }
  [[nodiscard]] std::size_t my() const { return 3 * ny / 2; }
  [[nodiscard]] std::size_t mxf() const { return mx() / 2 + 1; }
  [[nodiscard]] double lx() const;
  [[nodiscard]] double ly() const;
  [[nodiscard]] std::size_t spectrumBins() const;
  [[nodiscard]] bool usesEtd() const {
    return integrator != Integrator::integratingFactorRk2;
  }
};

[[nodiscard]] const char *integratorName(Integrator integrator);
[[nodiscard]] const char *forcingProfileName(ForcingProfile profile);
Parameters readParameters(const std::filesystem::path &path);
void validateParameters(const Parameters &parameters);
void writeParameterRecord(const Parameters &parameters,
                          const std::string &backend,
                          const std::filesystem::path &recordDirectory = {});
