#include "fftw_utils.hpp"

#include <algorithm>

BaseTransform::BaseTransform(const Parameters &p)
    : p_(p), planningReal_(p.nx * p.ny), planningComplex_(p.spectralSize()) {
  forward_.reset(fftw_plan_dft_r2c_2d(
      static_cast<int>(p.ny), static_cast<int>(p.nx), planningReal_.data(),
      fftwData(planningComplex_), FFTW_ESTIMATE));
  inverse_.reset(fftw_plan_dft_c2r_2d(
      static_cast<int>(p.ny), static_cast<int>(p.nx),
      fftwData(planningComplex_), planningReal_.data(), FFTW_ESTIMATE));
  if (!forward_ || !inverse_)
    throw std::runtime_error("FFTW could not create base-grid plans");
}

void BaseTransform::forward(const std::vector<double> &real,
                            SpectralField &spectral) {
  if (real.size() != p_.nx * p_.ny)
    throw std::runtime_error("invalid real field size");
  planningReal_.assign(real.begin(), real.end());
  forward_.execute();
  spectral.assign(planningComplex_.begin(), planningComplex_.end());
  const double scale = 1.0 / static_cast<double>(p_.nx * p_.ny);
  for (Complex &value : spectral)
    value *= scale;
}

void BaseTransform::inverse(const SpectralField &spectral,
                            std::vector<double> &real) {
  if (spectral.size() != p_.spectralSize())
    throw std::runtime_error("invalid spectral field size");
  planningComplex_.assign(spectral.begin(),
                          spectral.end()); // FFTW may overwrite c2r input.
  inverse_.execute();
  real.assign(planningReal_.begin(), planningReal_.end());
}
