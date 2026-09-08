#include "solver.hpp"
#include "spectral.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
[[maybe_unused]] constexpr std::size_t parallelThreshold = 16384;

SpectralField field(const Parameters &p, bool needed = true) {
  return needed ? SpectralField(p.spectralSize()) : SpectralField{};
}

void requireFinite(const SpectralField &values, const char *description) {
  const bool valid = std::all_of(values.begin(), values.end(), [](Complex x) {
    return std::isfinite(x.real()) && std::isfinite(x.imag());
  });
  if (!valid)
    throw std::runtime_error(std::string(description) +
                             " contains a non-finite coefficient");
}

template <class Operation>
void forEachIndex(std::size_t count, Operation operation) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (count >= parallelThreshold)
#endif
  for (std::ptrdiff_t rawIndex = 0;
       rawIndex < static_cast<std::ptrdiff_t>(count); ++rawIndex)
    operation(static_cast<std::size_t>(rawIndex));
}

std::uint64_t resolveRandomSeed(std::uint64_t configuredSeed) {
  if (configuredSeed == 0)
    configuredSeed = static_cast<std::uint64_t>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());
  return backendSynchronizeSeed(configuredSeed);
}
} // namespace

Solver::Solver(Parameters p, std::unique_ptr<NonlinearBackend> backend)
    : p_(std::move(p)), backend_(std::move(backend)), baseTransform_(p_),
      linear_(field(p_)),
      noise_(field(p_, p_.usesStochasticForcing())),
      n1_(field(p_, !backend_->deviceTimeStepping())),
      n2_(field(p_, !backend_->deviceTimeStepping())),
      n3_(field(p_, !backend_->deviceTimeStepping() && p_.usesStageB())),
      n4_(field(p_, !backend_->deviceTimeStepping() && p_.usesStageC())),
      stageA_(field(p_, !backend_->deviceTimeStepping())),
      stageB_(field(p_, !backend_->deviceTimeStepping() && p_.usesStageB())),
      stageC_(field(p_, !backend_->deviceTimeStepping() && p_.usesStageC())),
      diagnosticNonlinear_(field(p_)), forcingAmplitude_(p_.spectralSize()),
      noiseScale_(noise_.size()),
      random_(p_.randomSeed = resolveRandomSeed(p_.randomSeed)) {
#ifdef _OPENMP
  if (p_.threadCount > 0)
    omp_set_num_threads(p_.threadCount);
#endif
  std::filesystem::create_directories(p_.dataDirectory);
  std::filesystem::create_directories(p_.outputDirectory);
  buildLinearOperator();
  buildIntegrationCoefficients();
  if (p_.forcingEnabled)
    buildForcing();
}

void Solver::buildLinearOperator() {
  forEachIndex(linear_.size(), [&](std::size_t index) {
    const std::size_t y = index / p_.nxf();
    const std::size_t x = index % p_.nxf();
    const double ky = waveNumberY(p_, y);
    const double kx = waveNumberX(p_, x);
    const double k2 = kx * kx + ky * ky;
    double value = 0.0;
    if (k2 > 0.0) {
      if (p_.viscosity > 0.0)
        value -= p_.viscosity * std::pow(k2, p_.viscosityOrder);
      if (p_.linearDrag > 0.0)
        value -= p_.linearDrag * std::pow(k2, p_.dragOrder);
    }
    const double frequency = p_.betaPlane && k2 > 0.0 ? p_.beta * kx / k2 : 0.0;
    linear_[index] = Complex(value, frequency);
  });
  requireFinite(linear_,
                "linear operator; reduce dissipation coefficients or orders");
}

void Solver::buildIntegrationCoefficients() {
  // Taylor series near zero avoids cancellation; recurrence is stable away
  // from zero. Complex arguments include the analytically integrated beta term.
  const auto phi = [](Complex z, int order) {
    double factorial = 1.0;
    for (int k = 2; k <= order; ++k)
      factorial *= k;
    if (std::abs(z) < 2.0) {
      Complex term = 1.0 / factorial;
      Complex sum = term;
      for (int k = 1; k < 64; ++k) {
        term *= z / static_cast<double>(order + k);
        sum += term;
        if (std::abs(term) < 1e-17 * std::abs(sum))
          break;
      }
      return sum;
    }
    Complex value = (std::exp(z) - 1.0) / z;
    double previousFactorial = 1.0;
    for (int k = 2; k <= order; ++k) {
      value = (value - 1.0 / previousFactorial) / z;
      previousFactorial *= k;
    }
    return value;
  };
  auto &c = coefficients_;
  c.e1 = field(p_);
  if (p_.usesEtd()) {
    c.q1 = field(p_);
    c.f1 = field(p_);
    if (p_.integrator != Integrator::etd2) {
      c.e2 = field(p_);
      c.q2 = field(p_);
      c.f2 = field(p_);
      c.f3 = field(p_);
    }
    if (p_.integrator == Integrator::etd4) {
      c.q3 = field(p_);
      c.q4 = field(p_);
      c.q5 = field(p_);
    }
  }
  const double h = p_.timeStep;
  forEachIndex(linear_.size(), [&](std::size_t i) {
    const Complex z = h * linear_[i];
    if (!std::isfinite(z.real()) || !std::isfinite(z.imag())) {
      c.e1[i] = Complex(std::numeric_limits<double>::quiet_NaN(), 0.0);
      return; // Report outside the OpenMP region, without throwing in a worker.
    }
    c.e1[i] = std::exp(z);
    if (p_.usesEtd()) {
      const Complex p1 = phi(z, 1), p2 = phi(z, 2);
      if (p_.integrator == Integrator::etd2) {
        c.q1[i] = h * p1;
        c.f1[i] = h * p2;
      } else {
        const Complex p3 = phi(z, 3), halfP1 = phi(0.5 * z, 1);
        c.e2[i] = std::exp(0.5 * z);
        c.q1[i] = (0.5 * h) * halfP1;
        c.q2[i] = h * p1;
        c.f1[i] = h * (p1 - 3.0 * p2 + 4.0 * p3);
        c.f2[i] = h * (p2 - 2.0 * p3);
        c.f3[i] = h * (-p2 + 4.0 * p3);
        if (p_.integrator == Integrator::etd4) {
          const Complex halfP2 = phi(0.5 * z, 2);
          c.q2[i] = h * (0.5 * halfP1 - halfP2);
          c.q3[i] = h * halfP2;
          c.q4[i] = h * (p1 - 2.0 * p2);
          c.q5[i] = (2.0 * h) * p2;
        }
      }
    }
    if (!noiseScale_.empty()) {
      const double damping = -linear_[i].real();
      const double exponent = -2.0 * (damping * h);
      // Exact covariance of the linear stochastic convolution. Circular
      // complex Gaussian noise is invariant under the beta-plane rotation.
      noiseScale_[i] = damping == 0.0 || exponent == 0.0
                           ? std::sqrt(h)
                           : std::sqrt(-std::expm1(exponent)) *
                                 (std::sqrt(0.5) / std::sqrt(damping));
    }
  });
  for (const auto *values : c.fields())
    requireFinite(*values, "time integration coefficients");
}

void Solver::buildForcing() {
  const long singleMode = p_.forcingProfile == ForcingProfile::singleMode
                              ? static_cast<long>(p_.forcingWavenumber)
                              : 0;
  const double maximumLog = std::log(std::numeric_limits<double>::max());
  for (std::size_t y = 0; y < p_.ny; ++y) {
    const long kyIndex = signedWave(y, p_.ny);
    for (std::size_t x = 0; x < p_.nxf(); ++x) {
      const double kx = waveNumberX(p_, x);
      const double ky = waveNumberY(p_, y);
      const double k = std::hypot(kx, ky);
      double amplitude = 0.0;
      if (p_.forcingProfile == ForcingProfile::annulus && k > 0.0 &&
          std::abs(k - p_.forcingWavenumber) < p_.forcingWidth)
        amplitude = p_.forcingAmplitude;
      else if (p_.forcingProfile == ForcingProfile::exponential && k > 0.0) {
        const double logRatio =
            p_.forcingShapeOrder * std::log(k / p_.forcingWavenumber);
        if (std::isfinite(logRatio) && logRatio <= maximumLog) {
          const double ratio = std::exp(logRatio);
          amplitude = p_.forcingAmplitude * std::exp(logRatio - ratio);
        }
      } else if (p_.forcingProfile == ForcingProfile::singleMode &&
                 static_cast<long>(x) == singleMode &&
                 std::abs(kyIndex) == singleMode)
        amplitude = kyIndex >= 0 ? -p_.forcingAmplitude : p_.forcingAmplitude;
      // Derivatives on even-grid Nyquist lines have no unique sign, so those
      // lines are deliberately excluded from both the state and the forcing.
      if (x == p_.nx / 2 || y == p_.ny / 2)
        amplitude = 0.0;
      forcingAmplitude_[spectralIndex(x, y, p_.nxf())] = amplitude;
      if (amplitude != 0.0 && k > 0.0) {
        forcedIndices_.push_back(spectralIndex(x, y, p_.nxf()));
        const double halfFactor = (x == 0 || x == p_.nx / 2) ? 0.5 : 1.0;
        enstrophyInjectionCoefficient_ += halfFactor * amplitude * amplitude;
        energyInjectionCoefficient_ +=
            halfFactor * amplitude * amplitude / (k * k);
        forcedModeCount_ += static_cast<std::size_t>(2.0 * halfFactor);
      }
    }
  }
  if (forcedModeCount_ == 0)
    throw std::runtime_error(
        "forcingEnabled is true, but the selected profile contains no modes");
  if (!std::isfinite(energyInjectionCoefficient_) ||
      !std::isfinite(enstrophyInjectionCoefficient_))
    throw std::runtime_error(
        "forcing amplitude produces non-finite injection coefficients");
  if (p_.targetEnergyInjectionRate > 0.0) {
    if (energyInjectionCoefficient_ == 0.0)
      throw std::runtime_error("cannot normalize an empty forcing spectrum");
    const double scale = std::sqrt(p_.targetEnergyInjectionRate) /
                         std::sqrt(energyInjectionCoefficient_);
    if (!(scale > 0.0) || !std::isfinite(scale))
      throw std::runtime_error(
          "targetEnergyInjectionRate cannot be represented with this forcing "
          "spectrum");
    forEachIndex(forcingAmplitude_.size(),
                 [&](std::size_t index) { forcingAmplitude_[index] *= scale; });
    enstrophyInjectionCoefficient_ *= scale * scale;
    if (!std::isfinite(enstrophyInjectionCoefficient_))
      throw std::runtime_error(
          "normalized forcing produces a non-finite enstrophy coefficient");
    energyInjectionCoefficient_ = p_.targetEnergyInjectionRate;
  }
  if (backendIsRoot()) {
    std::cout << "number of forced modes = " << forcedModeCount_;
    if (p_.forcingProfile != ForcingProfile::singleMode)
      std::cout << "\nstochastic energy-injection coefficient = "
                << energyInjectionCoefficient_
                << "\nstochastic enstrophy-injection coefficient = "
                << enstrophyInjectionCoefficient_;
    std::cout << '\n';
  }
}

void Solver::generateNoise(SpectralField &noise) {
  std::fill(noise.begin(), noise.end(), Complex{});
  const double scale = std::sqrt(0.5);
  for (const std::size_t i : forcedIndices_) {
    const double amplitude = forcingAmplitude_[i];
    const double real = normal_(random_);
    const double imaginary = normal_(random_);
    noise[i] = (amplitude * scale * noiseScale_[i]) * Complex(real, imaginary);
  }
  enforceRealityConstraints(noise, p_);
}

void Solver::rightHandSide(const SpectralField &input, SpectralField &output) {
  backend_->evaluate(input, output);
  if (p_.forcingEnabled && p_.forcingProfile == ForcingProfile::singleMode)
    forEachIndex(output.size(),
                 [&](std::size_t i) { output[i] += forcingAmplitude_[i]; });
}

void Solver::step(SpectralField &w) {
  if (!noise_.empty())
    generateNoise(noise_);
  if (backend_->deviceTimeStepping()) {
    backend_->advance(noise_);
    return;
  }
  const auto c = coefficients_.pointers();
  rightHandSide(w, n1_);
  forEachIndex(w.size(), [&](std::size_t i) {
    stageA_[i] =
        integrationStageA(p_.integrator, p_.timeStep, i, c, w[i], n1_[i]);
  });
  rightHandSide(stageA_, n2_);
  if (!n3_.empty()) {
    forEachIndex(w.size(), [&](std::size_t i) {
      stageB_[i] = integrationStageB(p_.integrator, i, c, w[i], n1_[i], n2_[i]);
    });
    rightHandSide(stageB_, n3_);
  }
  if (!n4_.empty()) {
    forEachIndex(w.size(), [&](std::size_t i) {
      stageC_[i] = integrationStageC(i, c, w[i], n1_[i], n3_[i]);
    });
    rightHandSide(stageC_, n4_);
  }
  forEachIndex(w.size(), [&](std::size_t i) {
    w[i] = integrationFinish(p_.integrator, p_.timeStep, i, c, w[i], stageA_[i],
                             n1_[i], n2_[i], n3_.empty() ? Complex{} : n3_[i],
                             n4_.empty() ? Complex{} : n4_[i]);
    if (!noise_.empty())
      w[i] += noise_[i];
  });
  enforceRealityConstraints(w, p_);
}

void Solver::writeState(const RestartState &state) {
  writeVorticity(p_, baseTransform_, state.vorticity, state.frame);
  std::ostringstream randomState, distributionState;
  randomState << random_;
  distributionState << normal_;
  writeRestart(p_, state.time, state.frame, state.vorticity, randomState.str(),
               distributionState.str());
}

void Solver::run() {
  const bool recoveredFresh = backendIsRoot() && recoverOutputTransaction(p_);
  backendBarrier();
  RestartState state = readRestart(p_, baseTransform_, backendIsRoot());
  const long double finalTime =
      static_cast<long double>(state.time) +
      static_cast<long double>(p_.timeStep) * p_.numberOfSteps;
  if (finalTime > static_cast<long double>(std::numeric_limits<double>::max()))
    throw std::runtime_error("requested run would overflow simulation time");
  const std::uint64_t scheduledOutputs =
      p_.numberOfSteps / p_.outputIntervalSteps +
      (p_.numberOfSteps % p_.outputIntervalSteps != 0 ? 1 : 0);
  if (state.frame >
      std::numeric_limits<std::uint64_t>::max() - scheduledOutputs)
    throw std::runtime_error("requested run would overflow output frame count");
  if (!state.randomEngineState.empty()) {
    std::istringstream savedRandomState(state.randomEngineState);
    if (!(savedRandomState >> random_))
      throw std::runtime_error("cannot restore random-generator state");
    p_.randomSeed = state.randomSeed;
  }
  if (!state.randomDistributionState.empty()) {
    std::istringstream savedDistributionState(state.randomDistributionState);
    if (!(savedDistributionState >> normal_))
      throw std::runtime_error("cannot restore normal-distribution state");
  } else if (state.restarting) {
    normal_.reset();
  }
  if (backendIsRoot()) {
    prepareOutputFiles(p_, state.restarting || recoveredFresh, state.frame);
    writeRunRecords(p_, backendName(), state.time, state.frame,
                    forcingAmplitude_, forcedModeCount_,
                    energyInjectionCoefficient_,
                    enstrophyInjectionCoefficient_);
    if (!state.restarting) {
      beginOutputTransaction(p_, state.frame);
      writeState(state);
      finishOutputTransaction(p_);
    }
    std::cout << "backend = " << backendName() << "\nnx = " << p_.nx
              << " ny = " << p_.ny << " timeStep = " << p_.timeStep
              << " numberOfSteps = " << p_.numberOfSteps
              << " outputIntervalSteps = " << p_.outputIntervalSteps
              << "\nrandom seed = " << p_.randomSeed << '\n';
  }
  if (backend_->deviceTimeStepping()) {
    const std::vector<double> noForcing;
    backend_->initializeTimeStepping(
        coefficients_,
        p_.forcingEnabled && p_.forcingProfile == ForcingProfile::singleMode
            ? forcingAmplitude_
            : noForcing,
        state.vorticity);
    coefficients_ = {};
  }
  DiagnosticsAverages averages;
  const auto start = std::chrono::steady_clock::now();
  for (std::uint64_t stepIndex = 0; stepIndex < p_.numberOfSteps; ++stepIndex) {
    const std::uint64_t stepNumber = stepIndex + 1;
    step(state.vorticity);
    state.time += p_.timeStep;
    if (stepNumber % p_.outputIntervalSteps == 0 ||
        stepNumber == p_.numberOfSteps) {
      ++state.frame;
      backend_->downloadStateAndEvaluate(state.vorticity,
                                         diagnosticNonlinear_); // MPI collective
      if (backendIsRoot()) {
        beginOutputTransaction(p_, state.frame);
        const double energy =
            writeDiagnostics(p_, state.time, state.frame, state.vorticity,
                             diagnosticNonlinear_, averages);
        writeState(state);
        finishOutputTransaction(p_);
        std::cout << "time = " << state.time << " file = " << state.frame
                  << " Energy = " << energy << '\n';
      }
    }
  }
  if (backendIsRoot()) {
    const std::chrono::duration<double> elapsed =
        std::chrono::steady_clock::now() - start;
    std::cout << "time taken for code is = " << elapsed.count() << '\n';
  }
}
