#pragma once

#include "backend.hpp"
#include "fftw_utils.hpp"
#include "output.hpp"

#include <array>
#include <memory>
#include <random>
#include <utility>
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
    RestartState prepareRun();
    void validateRunBounds(const RestartState &state) const;
    void restoreRandomState(const RestartState &state);
    void initializeDeviceTimeStepping(const SpectralField &vorticity);
    void writeOutputFrame(const RestartState &state, DiagnosticsAverages &averages);
    void writeState(const RestartState &state);

    Parameters parameters_;
    std::unique_ptr<NonlinearBackend> backend_;
    BaseTransform baseTransform_;
    SpectralField linearOperator_;
    IntegrationCoefficients coefficients_;
    SpectralField noise_, nonlinearAtStart_, nonlinearAtStageA_, nonlinearAtStageB_,
        nonlinearAtStageC_, stageA_, stageB_, stageC_;
    SpectralField diagnosticNonlinearTerm_;
    std::vector<double> forcingAmplitude_, stochasticNoiseScale_;
    std::vector<std::size_t> forcedIndices_;
    std::vector<std::pair<std::size_t, std::size_t>> compactNoiseRealityPairs_;
    std::size_t forcedModeCount_ = 0;
    bool compactDeviceNoise_ = false;
    double energyInjectionCoefficient_ = 0.0;
    double enstrophyInjectionCoefficient_ = 0.0;
    std::mt19937_64 random_;
    std::normal_distribution<double> normal_{0.0, 1.0};
};
