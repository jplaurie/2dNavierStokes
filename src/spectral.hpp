#pragma once

#include "backend.hpp"

#include <cstddef>

constexpr double nsPi = 3.141592653589793238462643383279502884;

inline std::size_t spectralIndex(std::size_t x, std::size_t y,
                                 std::size_t nxf) {
  return y * nxf + x;
}

inline long signedWave(std::size_t y, std::size_t ny) {
  return y < ny / 2 ? static_cast<long>(y)
                    : static_cast<long>(y) - static_cast<long>(ny);
}

inline std::size_t paddedWaveRow(std::size_t y, std::size_t ny,
                                 std::size_t my) {
  const long wave = signedWave(y, ny);
  return wave >= 0 ? static_cast<std::size_t>(wave)
                   : static_cast<std::size_t>(static_cast<long>(my) + wave);
}

inline double waveNumberX(const Parameters &p, std::size_t x) {
  return 2.0 * nsPi * static_cast<double>(x) / p.lx();
}

inline double waveNumberY(const Parameters &p, std::size_t y) {
  return 2.0 * nsPi * static_cast<double>(signedWave(y, p.ny)) / p.ly();
}

inline Complex nonlinearFactor(std::size_t component, double kx, double ky) {
  const double k2 = kx * kx + ky * ky;
  const double factors[] = {kx, k2 == 0.0 ? 0.0 : -ky / k2, ky,
                            k2 == 0.0 ? 0.0 : -kx / k2};
  return Complex(0.0, factors[component]);
}

void enforceRealityConstraints(SpectralField &field,
                               const Parameters &parameters);
