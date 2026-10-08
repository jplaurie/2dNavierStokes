#pragma once

#include "backend.hpp"
#include "parallel.hpp"

#include <array>
#include <stdexcept>

struct HostIntegrationWorkspace {
    std::array<SpectralField, 4> nonlinearStages;
    std::array<SpectralField, 3> stageStates;

    void initialize(std::size_t count, std::size_t nonlinearStageCount) {
        for (std::size_t i = 0; i < nonlinearStageCount; ++i)
            nonlinearStages[i].resize(count);
        for (std::size_t i = 1; i < nonlinearStageCount; ++i)
            stageStates[i - 1].resize(count);
    }
};

template <class RightHandSide, class EnforceConstraints>
void advanceHostTimeStep(const Parameters &parameters, const IntegrationCoefficients &coefficients,
                         SpectralField &state, const SpectralField &noise,
                         HostIntegrationWorkspace &workspace, RightHandSide rightHandSide,
                         EnforceConstraints enforceConstraints) {
    if (!noise.empty() && noise.size() != state.size())
        throw std::runtime_error("invalid host stochastic-noise field");
    const auto coefficientPointers = coefficients.pointers();
    SpectralField &n1 = workspace.nonlinearStages[0];
    SpectralField &n2 = workspace.nonlinearStages[1];
    SpectralField &n3 = workspace.nonlinearStages[2];
    SpectralField &n4 = workspace.nonlinearStages[3];
    SpectralField &a = workspace.stageStates[0];
    SpectralField &b = workspace.stageStates[1];
    SpectralField &c = workspace.stageStates[2];

    rightHandSide(state, n1);
    forEachIndex(state.size(), [&](std::size_t i) {
        a[i] = integrationStageA(parameters.integrator, parameters.timeStep, i, coefficientPointers,
                                 state[i], n1[i]);
    });
    rightHandSide(a, n2);
    if (!n3.empty()) {
        forEachIndex(state.size(), [&](std::size_t i) {
            b[i] = integrationStageB(parameters.integrator, i, coefficientPointers, state[i], n1[i],
                                     n2[i]);
        });
        rightHandSide(b, n3);
    }
    if (!n4.empty()) {
        forEachIndex(state.size(), [&](std::size_t i) {
            c[i] = integrationStageC(i, coefficientPointers, state[i], n1[i], n3[i]);
        });
        rightHandSide(c, n4);
    }
    forEachIndex(state.size(), [&](std::size_t i) {
        state[i] = integrationFinish(
            parameters.integrator, parameters.timeStep, i, coefficientPointers, state[i], a[i],
            n1[i], n2[i], n3.empty() ? Complex{} : n3[i], n4.empty() ? Complex{} : n4[i]);
        if (!noise.empty())
            state[i] += noise[i];
    });
    enforceConstraints(state);
}
