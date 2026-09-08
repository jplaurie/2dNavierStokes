#include "spectral.hpp"

#include <complex>
#include <stdexcept>

void enforceRealityConstraints(SpectralField &a, const Parameters &p) {
  if (a.size() != p.spectralSize())
    throw std::runtime_error("invalid spectral field size");
  const auto at = [&](std::size_t x, std::size_t y) -> Complex & {
    return a[spectralIndex(x, y, p.nxf())];
  };
  at(0, 0) = 0.0;
  // Even-grid Nyquist derivatives have no unique sign. Removing both Nyquist
  // lines keeps all padded-grid derivatives real and avoids double-counting
  // the x Nyquist mode after 3/2 embedding.
  for (std::size_t y = 0; y < p.ny; ++y)
    at(p.nx / 2, y) = 0.0;
  for (std::size_t x = 0; x < p.nxf(); ++x)
    at(x, p.ny / 2) = 0.0;
  for (std::size_t y = 1; y < p.ny / 2; ++y)
    at(0, p.ny - y) = std::conj(at(0, y));
}
