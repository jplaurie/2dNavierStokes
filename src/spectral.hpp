#pragma once

#include "backend.hpp"

#include <array>
#include <cstddef>

constexpr double nsPi = 3.141592653589793238462643383279502884;

// The four physical-space fields used to evaluate J(omega, psi). Giving the
// components names keeps their ordering explicit across the CPU and MPI
// backends instead of relying on unexplained array indices.
enum class NonlinearComponent : std::size_t {
    vorticityX,
    streamfunctionY,
    vorticityY,
    streamfunctionX,
};

constexpr std::array nonlinearComponents{
    NonlinearComponent::vorticityX, NonlinearComponent::streamfunctionY,
    NonlinearComponent::vorticityY, NonlinearComponent::streamfunctionX};

constexpr std::size_t componentIndex(NonlinearComponent component) {
    return static_cast<std::size_t>(component);
}

inline std::size_t spectralIndex(std::size_t x, std::size_t y, std::size_t nxf) {
    return y * nxf + x;
}

inline long signedWave(std::size_t y, std::size_t ny) {
    return y < ny / 2 ? static_cast<long>(y) : static_cast<long>(y) - static_cast<long>(ny);
}

inline std::size_t paddedWaveRow(std::size_t y, std::size_t ny, std::size_t my) {
    const long wave = signedWave(y, ny);
    return wave >= 0 ? static_cast<std::size_t>(wave)
                     : static_cast<std::size_t>(static_cast<long>(my) + wave);
}

inline double waveNumberX(const Parameters &parameters, std::size_t x) {
    return 2.0 * nsPi * static_cast<double>(x) / parameters.lx();
}

inline double waveNumberY(const Parameters &parameters, std::size_t y) {
    return 2.0 * nsPi * static_cast<double>(signedWave(y, parameters.ny)) / parameters.ly();
}

inline Complex nonlinearFactor(NonlinearComponent component, double kx, double ky) {
    const double k2 = kx * kx + ky * ky;
    switch (component) {
    case NonlinearComponent::vorticityX:
        return Complex(0.0, kx);
    case NonlinearComponent::streamfunctionY:
        return Complex(0.0, k2 == 0.0 ? 0.0 : -ky / k2);
    case NonlinearComponent::vorticityY:
        return Complex(0.0, ky);
    case NonlinearComponent::streamfunctionX:
        return Complex(0.0, k2 == 0.0 ? 0.0 : -kx / k2);
    }
    return {};
}

void enforceRealityConstraints(SpectralField &field, const Parameters &parameters);
