#pragma once

#include "backend.hpp"
#include "fftw_utils.hpp"
#include "output.hpp"

#include <array>
#include <memory>
#include <random>
#include <vector>

class Solver {
public:
  Solver(Parameters parameters, std::unique_ptr<NonlinearBackend> backend);
  void run();

private:
  void buildLinearOperator();
  void buildIntegrationCoefficients();
  void buildForcing();
  void generateNoise(SpectralField &noise);
  void rightHandSide(const SpectralField &input, SpectralField &output);
  void step(SpectralField &vorticity);

  Parameters p_;
  std::unique_ptr<NonlinearBackend> backend_;
  BaseTransform baseTransform_;
  SpectralField linear_;
  IntegrationCoefficients coefficients_;
  SpectralField noise_, n1_, n2_, n3_, n4_, stageA_, stageB_, stageC_;
  SpectralField diagnosticNonlinear_;
  std::vector<double> forcingAmplitude_, noiseScale_;
  std::vector<std::size_t> forcedIndices_;
  std::size_t forcedModeCount_ = 0;
  double energyInjectionCoefficient_ = 0.0;
  double enstrophyInjectionCoefficient_ = 0.0;
  std::mt19937_64 random_;
  std::normal_distribution<double> normal_{0.0, 1.0};
};
