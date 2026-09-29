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

SpectralField makeSpectralField(const Parameters &parameters, bool needed = true) {
    return needed ? SpectralField(parameters.spectralSize()) : SpectralField{};
}

void requireFinite(const SpectralField &values, const char *description) {
    const bool valid = std::all_of(values.begin(), values.end(), [](Complex x) {
        return std::isfinite(x.real()) && std::isfinite(x.imag());
    });
    if (!valid)
        throw std::runtime_error(std::string(description) + " contains a non-finite coefficient");
}

template <class Operation> void forEachIndex(std::size_t count, Operation operation) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (count >= parallelThreshold)
#endif
    for (std::ptrdiff_t rawIndex = 0; rawIndex < static_cast<std::ptrdiff_t>(count); ++rawIndex)
        operation(static_cast<std::size_t>(rawIndex));
}

std::uint64_t resolveRandomSeed(std::uint64_t configuredSeed) {
    if (configuredSeed == 0)
        configuredSeed = static_cast<std::uint64_t>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count());
    return backendSynchronizeSeed(configuredSeed);
}
} // namespace

Solver::Solver(Parameters parameters, std::unique_ptr<NonlinearBackend> backend)
    : parameters_(std::move(parameters)), backend_(std::move(backend)), baseTransform_(parameters_),
      linearOperator_(makeSpectralField(parameters_)),
      noise_(makeSpectralField(parameters_, parameters_.usesStochasticForcing())),
      nonlinearAtStart_(makeSpectralField(parameters_, !backend_->supportsDeviceTimeStepping())),
      nonlinearAtStageA_(makeSpectralField(parameters_, !backend_->supportsDeviceTimeStepping())),
      nonlinearAtStageB_(makeSpectralField(parameters_, !backend_->supportsDeviceTimeStepping() &&
                                                            parameters_.usesStageB())),
      nonlinearAtStageC_(makeSpectralField(parameters_, !backend_->supportsDeviceTimeStepping() &&
                                                            parameters_.usesStageC())),
      stageA_(makeSpectralField(parameters_, !backend_->supportsDeviceTimeStepping())),
      stageB_(makeSpectralField(parameters_, !backend_->supportsDeviceTimeStepping() &&
                                                 parameters_.usesStageB())),
      stageC_(makeSpectralField(parameters_, !backend_->supportsDeviceTimeStepping() &&
                                                 parameters_.usesStageC())),
      diagnosticNonlinearTerm_(makeSpectralField(parameters_)),
      forcingAmplitude_(parameters_.spectralSize()), stochasticNoiseScale_(noise_.size()),
      random_(parameters_.randomSeed = resolveRandomSeed(parameters_.randomSeed)) {
#ifdef _OPENMP
    if (parameters_.threadCount > 0)
        omp_set_num_threads(parameters_.threadCount);
#endif
    std::filesystem::create_directories(parameters_.dataDirectory);
    std::filesystem::create_directories(parameters_.outputDirectory);
    buildLinearOperator();
    buildIntegrationCoefficients();
    if (parameters_.forcingEnabled) {
        buildForcing();
        // A narrow stochastic spectrum should not require a full spectral makeSpectralField to
        // cross the PCIe bus every step. Keep the dense representation for broad
        // spectra, where a separate scatter kernel would save little transfer.
        if (parameters_.usesStochasticForcing() && backend_->supportsDeviceTimeStepping() &&
            forcedIndices_.size() < noise_.size() / 2) {
            compactDeviceNoise_ = true;
            SpectralField compactNoise(forcedIndices_.size());
            noise_.swap(compactNoise);
            for (std::size_t destination = 0; destination < forcedIndices_.size(); ++destination) {
                const std::size_t index = forcedIndices_[destination];
                const std::size_t x = index % parameters_.nxf();
                const std::size_t y = index / parameters_.nxf();
                if (x != 0 || y <= parameters_.ny / 2)
                    continue;
                const std::size_t partnerIndex = (parameters_.ny - y) * parameters_.nxf();
                const auto partner =
                    std::lower_bound(forcedIndices_.begin(), forcedIndices_.end(), partnerIndex);
                if (partner == forcedIndices_.end() || *partner != partnerIndex)
                    throw std::logic_error("compact stochastic mode lacks its conjugate");
                compactNoiseRealityPairs_.emplace_back(
                    destination, static_cast<std::size_t>(partner - forcedIndices_.begin()));
            }
        }
    }
}

void Solver::buildLinearOperator() {
    forEachIndex(linearOperator_.size(), [&](std::size_t index) {
        const std::size_t y = index / parameters_.nxf();
        const std::size_t x = index % parameters_.nxf();
        const double ky = waveNumberY(parameters_, y);
        const double kx = waveNumberX(parameters_, x);
        const double k2 = kx * kx + ky * ky;
        double value = 0.0;
        if (k2 > 0.0) {
            if (parameters_.viscosity > 0.0)
                value -= parameters_.viscosity * std::pow(k2, parameters_.viscosityOrder);
            if (parameters_.linearDrag > 0.0)
                value -= parameters_.linearDrag * std::pow(k2, parameters_.dragOrder);
        }
        const double frequency =
            parameters_.betaPlane && k2 > 0.0 ? parameters_.beta * kx / k2 : 0.0;
        linearOperator_[index] = Complex(value, frequency);
    });
    requireFinite(linearOperator_, "linear operator; reduce dissipation coefficients or orders");
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
    auto &coefficients = coefficients_;
    coefficients.e1 = makeSpectralField(parameters_);
    if (parameters_.usesEtd()) {
        coefficients.q1 = makeSpectralField(parameters_);
        coefficients.f1 = makeSpectralField(parameters_);
        if (parameters_.integrator != Integrator::etd2) {
            coefficients.e2 = makeSpectralField(parameters_);
            coefficients.q2 = makeSpectralField(parameters_);
            coefficients.f2 = makeSpectralField(parameters_);
            coefficients.f3 = makeSpectralField(parameters_);
        }
        if (parameters_.integrator == Integrator::etd4) {
            coefficients.q3 = makeSpectralField(parameters_);
            coefficients.q4 = makeSpectralField(parameters_);
            coefficients.q5 = makeSpectralField(parameters_);
        }
    }
    const double h = parameters_.timeStep;
    forEachIndex(linearOperator_.size(), [&](std::size_t i) {
        const Complex z = h * linearOperator_[i];
        if (!std::isfinite(z.real()) || !std::isfinite(z.imag())) {
            coefficients.e1[i] = Complex(std::numeric_limits<double>::quiet_NaN(), 0.0);
            return; // Report outside the OpenMP region, without throwing in a worker.
        }
        coefficients.e1[i] = std::exp(z);
        if (parameters_.usesEtd()) {
            const Complex p1 = phi(z, 1), p2 = phi(z, 2);
            if (parameters_.integrator == Integrator::etd2) {
                coefficients.q1[i] = h * p1;
                coefficients.f1[i] = h * p2;
            } else {
                const Complex p3 = phi(z, 3), halfP1 = phi(0.5 * z, 1);
                coefficients.e2[i] = std::exp(0.5 * z);
                coefficients.q1[i] = (0.5 * h) * halfP1;
                coefficients.q2[i] = h * p1;
                coefficients.f1[i] = h * (p1 - 3.0 * p2 + 4.0 * p3);
                coefficients.f2[i] = h * (p2 - 2.0 * p3);
                coefficients.f3[i] = h * (-p2 + 4.0 * p3);
                if (parameters_.integrator == Integrator::etd4) {
                    const Complex halfP2 = phi(0.5 * z, 2);
                    coefficients.q2[i] = h * (0.5 * halfP1 - halfP2);
                    coefficients.q3[i] = h * halfP2;
                    coefficients.q4[i] = h * (p1 - 2.0 * p2);
                    coefficients.q5[i] = (2.0 * h) * p2;
                }
            }
        }
        if (!stochasticNoiseScale_.empty()) {
            const double damping = -linearOperator_[i].real();
            const double exponent = -2.0 * (damping * h);
            // Exact covariance of the linear stochastic convolution. Circular
            // complex Gaussian noise is invariant under the beta-plane rotation.
            stochasticNoiseScale_[i] =
                damping == 0.0 || exponent == 0.0
                    ? std::sqrt(h)
                    : std::sqrt(-std::expm1(exponent)) * (std::sqrt(0.5) / std::sqrt(damping));
        }
    });
    for (const auto *values : coefficients.fields())
        requireFinite(*values, "time integration coefficients");
}

void Solver::buildForcing() {
    const long singleMode = parameters_.forcingProfile == ForcingProfile::singleMode
                                ? static_cast<long>(parameters_.forcingWavenumber)
                                : 0;
    const double maximumLog = std::log(std::numeric_limits<double>::max());
    for (std::size_t y = 0; y < parameters_.ny; ++y) {
        const long kyIndex = signedWave(y, parameters_.ny);
        for (std::size_t x = 0; x < parameters_.nxf(); ++x) {
            const double kx = waveNumberX(parameters_, x);
            const double ky = waveNumberY(parameters_, y);
            const double k = std::hypot(kx, ky);
            double amplitude = 0.0;
            if (parameters_.forcingProfile == ForcingProfile::annulus && k > 0.0 &&
                std::abs(k - parameters_.forcingWavenumber) < parameters_.forcingWidth)
                amplitude = parameters_.forcingAmplitude;
            else if (parameters_.forcingProfile == ForcingProfile::exponential && k > 0.0) {
                const double logRatio =
                    parameters_.forcingShapeOrder * std::log(k / parameters_.forcingWavenumber);
                if (std::isfinite(logRatio) && logRatio <= maximumLog) {
                    const double ratio = std::exp(logRatio);
                    amplitude = parameters_.forcingAmplitude * std::exp(logRatio - ratio);
                }
            } else if (parameters_.forcingProfile == ForcingProfile::singleMode &&
                       static_cast<long>(x) == singleMode && std::abs(kyIndex) == singleMode)
                amplitude =
                    kyIndex >= 0 ? -parameters_.forcingAmplitude : parameters_.forcingAmplitude;
            // Derivatives on even-grid Nyquist lines have no unique sign, so those
            // lines are deliberately excluded from both the state and the forcing.
            if (x == parameters_.nx / 2 || y == parameters_.ny / 2)
                amplitude = 0.0;
            forcingAmplitude_[spectralIndex(x, y, parameters_.nxf())] = amplitude;
            if (amplitude != 0.0 && k > 0.0) {
                forcedIndices_.push_back(spectralIndex(x, y, parameters_.nxf()));
                const double halfFactor = (x == 0 || x == parameters_.nx / 2) ? 0.5 : 1.0;
                enstrophyInjectionCoefficient_ += halfFactor * amplitude * amplitude;
                energyInjectionCoefficient_ += halfFactor * amplitude * amplitude / (k * k);
                forcedModeCount_ += static_cast<std::size_t>(2.0 * halfFactor);
            }
        }
    }
    if (forcedModeCount_ == 0)
        throw std::runtime_error(
            "forcingEnabled is true, but the selected profile contains no modes");
    if (!std::isfinite(energyInjectionCoefficient_) ||
        !std::isfinite(enstrophyInjectionCoefficient_))
        throw std::runtime_error("forcing amplitude produces non-finite injection coefficients");
    if (parameters_.targetEnergyInjectionRate > 0.0) {
        if (energyInjectionCoefficient_ == 0.0)
            throw std::runtime_error("cannot normalize an empty forcing spectrum");
        const double scale = std::sqrt(parameters_.targetEnergyInjectionRate) /
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
        energyInjectionCoefficient_ = parameters_.targetEnergyInjectionRate;
    }
    if (backendIsRoot()) {
        std::cout << "number of forced modes = " << forcedModeCount_;
        if (parameters_.forcingProfile != ForcingProfile::singleMode)
            std::cout << "\nstochastic energy-injection coefficient = "
                      << energyInjectionCoefficient_
                      << "\nstochastic enstrophy-injection coefficient = "
                      << enstrophyInjectionCoefficient_;
        std::cout << '\n';
    }
}

void Solver::generateNoise(SpectralField &noise) {
    if (!compactDeviceNoise_)
        std::fill(noise.begin(), noise.end(), Complex{});
    const double scale = std::sqrt(0.5);
    for (std::size_t position = 0; position < forcedIndices_.size(); ++position) {
        const std::size_t i = forcedIndices_[position];
        const double amplitude = forcingAmplitude_[i];
        const double real = normal_(random_);
        const double imaginary = normal_(random_);
        noise[compactDeviceNoise_ ? position : i] =
            (amplitude * scale * stochasticNoiseScale_[i]) * Complex(real, imaginary);
    }
    if (compactDeviceNoise_) {
        for (const auto [destination, source] : compactNoiseRealityPairs_)
            noise[destination] = std::conj(noise[source]);
    } else {
        enforceRealityConstraints(noise, parameters_);
    }
}

void Solver::rightHandSide(const SpectralField &input, SpectralField &output) {
    backend_->evaluate(input, output);
    if (parameters_.forcingEnabled && parameters_.forcingProfile == ForcingProfile::singleMode)
        forEachIndex(output.size(), [&](std::size_t i) { output[i] += forcingAmplitude_[i]; });
}

void Solver::step(SpectralField &vorticity) {
    if (!noise_.empty())
        generateNoise(noise_);
    if (backend_->supportsDeviceTimeStepping()) {
        backend_->advanceDeviceState(noise_);
        return;
    }
    const auto coefficients = coefficients_.pointers();
    rightHandSide(vorticity, nonlinearAtStart_);
    forEachIndex(vorticity.size(), [&](std::size_t i) {
        stageA_[i] = integrationStageA(parameters_.integrator, parameters_.timeStep, i,
                                       coefficients, vorticity[i], nonlinearAtStart_[i]);
    });
    rightHandSide(stageA_, nonlinearAtStageA_);
    if (!nonlinearAtStageB_.empty()) {
        forEachIndex(vorticity.size(), [&](std::size_t i) {
            stageB_[i] = integrationStageB(parameters_.integrator, i, coefficients, vorticity[i],
                                           nonlinearAtStart_[i], nonlinearAtStageA_[i]);
        });
        rightHandSide(stageB_, nonlinearAtStageB_);
    }
    if (!nonlinearAtStageC_.empty()) {
        forEachIndex(vorticity.size(), [&](std::size_t i) {
            stageC_[i] = integrationStageC(i, coefficients, vorticity[i], nonlinearAtStart_[i],
                                           nonlinearAtStageB_[i]);
        });
        rightHandSide(stageC_, nonlinearAtStageC_);
    }
    forEachIndex(vorticity.size(), [&](std::size_t i) {
        vorticity[i] =
            integrationFinish(parameters_.integrator, parameters_.timeStep, i, coefficients,
                              vorticity[i], stageA_[i], nonlinearAtStart_[i], nonlinearAtStageA_[i],
                              nonlinearAtStageB_.empty() ? Complex{} : nonlinearAtStageB_[i],
                              nonlinearAtStageC_.empty() ? Complex{} : nonlinearAtStageC_[i]);
        if (!noise_.empty())
            vorticity[i] += noise_[i];
    });
    enforceRealityConstraints(vorticity, parameters_);
}

void Solver::writeState(const RestartState &state) {
    writeVorticity(parameters_, baseTransform_, state.vorticity, state.frame);
    std::ostringstream randomState, distributionState;
    randomState << random_;
    distributionState << normal_;
    writeRestart(parameters_, state.time, state.frame, state.vorticity, randomState.str(),
                 distributionState.str());
}

void Solver::validateRunBounds(const RestartState &state) const {
    // Compute these bounds before entering the time loop so counters and the
    // floating-point simulation time cannot silently wrap during a long run.
    const long double finalTime =
        static_cast<long double>(state.time) +
        static_cast<long double>(parameters_.timeStep) * parameters_.numberOfSteps;
    if (finalTime > static_cast<long double>(std::numeric_limits<double>::max()))
        throw std::runtime_error("requested run would overflow simulation time");

    const std::uint64_t scheduledOutputs =
        parameters_.numberOfSteps / parameters_.outputIntervalSteps +
        (parameters_.numberOfSteps % parameters_.outputIntervalSteps != 0 ? 1 : 0);
    if (state.frame > std::numeric_limits<std::uint64_t>::max() - scheduledOutputs)
        throw std::runtime_error("requested run would overflow output frame count");
}

void Solver::restoreRandomState(const RestartState &state) {
    if (!state.randomEngineState.empty()) {
        std::istringstream savedRandomState(state.randomEngineState);
        if (!(savedRandomState >> random_))
            throw std::runtime_error("cannot restore random-generator state");
        parameters_.randomSeed = state.randomSeed;
    }
    if (!state.randomDistributionState.empty()) {
        std::istringstream savedDistributionState(state.randomDistributionState);
        if (!(savedDistributionState >> normal_))
            throw std::runtime_error("cannot restore normal-distribution state");
    } else if (state.restarting) {
        normal_.reset();
    }
}

void Solver::initializeDeviceTimeStepping(const SpectralField &vorticity) {
    if (!backend_->supportsDeviceTimeStepping())
        return;

    const std::vector<double> noForcing;
    const std::vector<std::size_t> noStochasticIndices;
    backend_->initializeDeviceState(
        coefficients_,
        parameters_.forcingEnabled && parameters_.forcingProfile == ForcingProfile::singleMode
            ? forcingAmplitude_
            : noForcing,
        compactDeviceNoise_ ? forcedIndices_ : noStochasticIndices, vorticity);
    coefficients_ = {}; // The device now owns the integration coefficients.
}

RestartState Solver::prepareRun() {
    const bool recoveredFresh = backendIsRoot() && recoverOutputTransaction(parameters_);
    backendBarrier();
    RestartState state = readRestart(parameters_, baseTransform_, backendIsRoot());
    validateRunBounds(state);
    restoreRandomState(state);

    if (backendIsRoot()) {
        prepareOutputFiles(parameters_, state.restarting || recoveredFresh, state.frame);
        writeRunRecords(parameters_, backendName(), state.time, state.frame, forcingAmplitude_,
                        forcedModeCount_, energyInjectionCoefficient_,
                        enstrophyInjectionCoefficient_);
        if (!state.restarting) {
            beginOutputTransaction(parameters_, state.frame);
            writeState(state);
            finishOutputTransaction(parameters_);
        }
        std::cout << "backend = " << backendName() << "\nnx = " << parameters_.nx
                  << " ny = " << parameters_.ny << " timeStep = " << parameters_.timeStep
                  << " numberOfSteps = " << parameters_.numberOfSteps
                  << " outputIntervalSteps = " << parameters_.outputIntervalSteps
                  << "\nrandom seed = " << parameters_.randomSeed << '\n';
    }
    initializeDeviceTimeStepping(state.vorticity);
    return state;
}

void Solver::writeOutputFrame(const RestartState &state, DiagnosticsAverages &averages) {
    beginOutputTransaction(parameters_, state.frame);
    const double energy = writeDiagnostics(parameters_, state.time, state.frame, state.vorticity,
                                           diagnosticNonlinearTerm_, averages);
    writeState(state);
    finishOutputTransaction(parameters_);
    std::cout << "time = " << state.time << " file = " << state.frame << " Energy = " << energy
              << '\n';
}

void Solver::run() {
    RestartState state = prepareRun();
    DiagnosticsAverages averages;
    const auto start = std::chrono::steady_clock::now();
    for (std::uint64_t stepIndex = 0; stepIndex < parameters_.numberOfSteps; ++stepIndex) {
        const std::uint64_t stepNumber = stepIndex + 1;
        step(state.vorticity);
        state.time += parameters_.timeStep;
        if (stepNumber % parameters_.outputIntervalSteps == 0 ||
            stepNumber == parameters_.numberOfSteps) {
            ++state.frame;
            backend_->downloadStateAndEvaluate(state.vorticity,
                                               diagnosticNonlinearTerm_); // MPI collective
            if (backendIsRoot())
                writeOutputFrame(state, averages);
        }
    }
    if (backendIsRoot()) {
        const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
        std::cout << "time taken for code is = " << elapsed.count() << '\n';
    }
}
