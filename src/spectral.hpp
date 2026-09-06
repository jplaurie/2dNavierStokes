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

void enforceRealityConstraints(SpectralField &field,
                               const Parameters &parameters);
