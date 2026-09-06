#include "backend.hpp"
#include "fftw_utils.hpp"
#include "spectral.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
Complex factor(std::size_t component, double kx, double ky) {
  const double k2 = kx * kx + ky * ky;
  const Complex imaginary(0.0, 1.0);
  if (component == 0)
    return imaginary * kx;
  if (component == 1)
    return k2 == 0.0 ? Complex{} : -imaginary * ky / k2;
  if (component == 2)
    return imaginary * ky;
  return k2 == 0.0 ? Complex{} : -imaginary * kx / k2;
}

std::vector<double> directInverseDerivative(const Parameters &p,
                                            const SpectralField &w,
                                            std::size_t component) {
  std::vector<double> result(p.mx() * p.my());
  for (std::size_t py = 0; py < p.my(); ++py) {
    const double physicalY =
        p.ly() * static_cast<double>(py) / static_cast<double>(p.my());
    for (std::size_t px = 0; px < p.mx(); ++px) {
      const double physicalX =
          p.lx() * static_cast<double>(px) / static_cast<double>(p.mx());
      Complex boundarySum{};
      double value = 0.0;
      for (std::size_t y = 0; y < p.ny; ++y) {
        const long kyIndex = signedWave(y, p.ny);
        const double ky = 2.0 * nsPi * static_cast<double>(kyIndex) / p.ly();
        for (std::size_t x = 0; x < p.nxf(); ++x) {
          const double kx = 2.0 * nsPi * static_cast<double>(x) / p.lx();
          const Complex phase =
              std::exp(Complex(0.0, kx * physicalX + ky * physicalY));
          const Complex term = w[spectralIndex(x, y, p.nxf())] *
                               factor(component, kx, ky) * phase;
          if (x == 0)
            boundarySum += term;
          else
            value += 2.0 * term.real();
        }
      }
      result[py * p.mx() + px] = value + boundarySum.real();
    }
  }
  return result;
}

SpectralField directNonlinear(const Parameters &p, const SpectralField &w) {
  const auto wx = directInverseDerivative(p, w, 0);
  const auto psiY = directInverseDerivative(p, w, 1);
  const auto wy = directInverseDerivative(p, w, 2);
  const auto psiX = directInverseDerivative(p, w, 3);
  std::vector<double> product(p.mx() * p.my());
  for (std::size_t i = 0; i < product.size(); ++i) {
    product[i] = wx[i] * psiY[i] - wy[i] * psiX[i];
  }

  SpectralField result(p.ny * p.nxf());
  const double scale = 1.0 / static_cast<double>(p.mx() * p.my());
  for (std::size_t y = 0; y < p.ny; ++y) {
    const long kyIndex = signedWave(y, p.ny);
    const double ky = 2.0 * nsPi * static_cast<double>(kyIndex) / p.ly();
    for (std::size_t x = 0; x < p.nxf(); ++x) {
      const double kx = 2.0 * nsPi * static_cast<double>(x) / p.lx();
      Complex sum{};
      for (std::size_t py = 0; py < p.my(); ++py) {
        const double physicalY =
            p.ly() * static_cast<double>(py) / static_cast<double>(p.my());
        for (std::size_t px = 0; px < p.mx(); ++px) {
          const double physicalX =
              p.lx() * static_cast<double>(px) / static_cast<double>(p.mx());
          sum += product[py * p.mx() + px] *
                 std::exp(Complex(0.0, -(kx * physicalX + ky * physicalY)));
        }
      }
      result[spectralIndex(x, y, p.nxf())] = sum * scale;
    }
  }
  enforceRealityConstraints(result, p);
  return result;
}

void checkInvariantTransfer(const Parameters &p, const SpectralField &w,
                            const SpectralField &nonlinear) {
  double energyTransfer = 0.0;
  double enstrophyTransfer = 0.0;
  double energyMagnitude = 0.0;
  double enstrophyMagnitude = 0.0;
  for (std::size_t y = 0; y < p.ny; ++y) {
    const double ky =
        2.0 * nsPi * static_cast<double>(signedWave(y, p.ny)) / p.ly();
    for (std::size_t x = 0; x < p.nxf(); ++x) {
      const double kx = 2.0 * nsPi * static_cast<double>(x) / p.lx();
      const double k2 = kx * kx + ky * ky;
      const double multiplicity = x == 0 || x == p.nx / 2 ? 1.0 : 2.0;
      const double transfer =
          multiplicity * std::real(std::conj(w[spectralIndex(x, y, p.nxf())]) *
                                   nonlinear[spectralIndex(x, y, p.nxf())]);
      enstrophyTransfer += transfer;
      enstrophyMagnitude += std::abs(transfer);
      if (k2 > 0.0) {
        energyTransfer += transfer / k2;
        energyMagnitude += std::abs(transfer / k2);
      }
    }
  }
  constexpr double tolerance = 5.0e-12;
  if (std::abs(energyTransfer) > tolerance * std::max(1.0, energyMagnitude))
    throw std::runtime_error("nonlinear term does not conserve energy: " +
                             std::to_string(energyTransfer));
  if (std::abs(enstrophyTransfer) >
      tolerance * std::max(1.0, enstrophyMagnitude))
    throw std::runtime_error("nonlinear term does not conserve enstrophy: " +
                             std::to_string(enstrophyTransfer));
}

void runCase(std::size_t nx, std::size_t ny, double aspectRatio,
             bool betaPlane) {
  Parameters p;
  p.nx = nx;
  p.ny = ny;
  p.aspectRatio = aspectRatio;
  p.betaPlane = betaPlane;
  p.beta = 3.0;
  p.threadCount = 1;
  p.forcingEnabled = false;
  validateParameters(p);

  std::vector<double> physical(p.nx * p.ny);
  for (std::size_t y = 0; y < p.ny; ++y) {
    const double yy =
        2.0 * nsPi * static_cast<double>(y) / static_cast<double>(p.ny);
    for (std::size_t x = 0; x < p.nx; ++x) {
      const double xx =
          2.0 * nsPi * static_cast<double>(x) / static_cast<double>(p.nx);
      physical[y * p.nx + x] =
          std::sin(xx) + 0.5 * std::cos(2.0 * yy) + 0.3 * std::sin(xx + yy) +
          0.2 * std::cos(2.0 * xx - yy) +
          0.6 * std::sin((p.nx / 2 - 1) * xx + (p.ny / 2 - 1) * yy) +
          0.4 * std::cos((p.nx / 2 - 2) * xx - (p.ny / 2 - 1) * yy);
    }
  }

  BaseTransform transform(p);
  SpectralField w;
  transform.forward(physical, w);
  enforceRealityConstraints(w, p);
  std::vector<double> reconstructed;
  transform.inverse(w, reconstructed);
  double roundTripError = 0.0;
  for (std::size_t i = 0; i < physical.size(); ++i)
    roundTripError =
        std::max(roundTripError, std::abs(reconstructed[i] - physical[i]));
  if (roundTripError > 2.0e-12)
    throw std::runtime_error("normalized FFT round trip failed: " +
                             std::to_string(roundTripError));

  const SpectralField expected = directNonlinear(p, w);
  SpectralField actual;
  {
    auto backend = makeBackend(p);
    backend->evaluate(w, actual);
  }
  double maxError = 0.0;
  double scale = 0.0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (!std::isfinite(actual[i].real()) || !std::isfinite(actual[i].imag()))
      throw std::runtime_error("non-finite nonlinear coefficient");
    maxError = std::max(maxError, std::abs(actual[i] - expected[i]));
    scale = std::max(scale, std::abs(expected[i]));
  }
  if (maxError > 2.0e-12 * std::max(1.0, scale))
    throw std::runtime_error(
        "dealiased nonlinear evaluator disagrees with direct DFT: " +
        std::to_string(maxError));
  checkInvariantTransfer(p, w, actual);
}
} // namespace

int main(int argc, char **argv) {
  backendInitialize(argc, argv);
  try {
    runCase(8, 8, 1.0, false);
    runCase(8, 12, 2.0, true);
    runCase(12, 8, 0.6, false);
    backendFinalize();
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    backendFinalize();
    return 1;
  }
}
