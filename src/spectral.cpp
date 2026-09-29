#include "spectral.hpp"

#include <complex>
#include <stdexcept>

void enforceRealityConstraints(SpectralField &a, const Parameters &parameters) {
    if (a.size() != parameters.spectralSize())
        throw std::runtime_error("invalid spectral field size");
    const auto at = [&](std::size_t x, std::size_t y) -> Complex & {
        return a[spectralIndex(x, y, parameters.nxf())];
    };
    at(0, 0) = 0.0;
    // Even-grid Nyquist derivatives have no unique sign. Removing both Nyquist
    // lines keeps all padded-grid derivatives real and avoids double-counting
    // the x Nyquist mode after 3/2 embedding.
    for (std::size_t y = 0; y < parameters.ny; ++y)
        at(parameters.nx / 2, y) = 0.0;
    for (std::size_t x = 0; x < parameters.nxf(); ++x)
        at(x, parameters.ny / 2) = 0.0;
    for (std::size_t y = 1; y < parameters.ny / 2; ++y)
        at(0, parameters.ny - y) = std::conj(at(0, y));
}
