#include "backend.hpp"
#include "fftw_utils.hpp"
#include "spectral.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
#ifdef NS2D_HAVE_FFTW_THREADS
bool fftwThreadsInitialized = false;
#endif

class CpuBackend final : public NonlinearBackend {
public:
  explicit CpuBackend(const Parameters &p)
      : p_(p), padded_(p.my() * p.mxf()), product_(p.mx() * p.my()),
        fields_{FftwRealField(p.mx() * p.my()), FftwRealField(p.mx() * p.my()),
                FftwRealField(p.mx() * p.my()),
                FftwRealField(p.mx() * p.my())} {
    for (std::size_t component = 0; component < inverse_.size(); ++component)
      inverse_[component].reset(fftw_plan_dft_c2r_2d(
          static_cast<int>(p.my()), static_cast<int>(p.mx()), fftwData(padded_),
          fields_[component].data(), FFTW_ESTIMATE));
    forward_.reset(
        fftw_plan_dft_r2c_2d(static_cast<int>(p.my()), static_cast<int>(p.mx()),
                             product_.data(), fftwData(padded_), FFTW_ESTIMATE));
    if (std::any_of(inverse_.begin(), inverse_.end(),
                    [](const FftwPlan &plan) { return !plan; }) ||
        !forward_)
      throw std::runtime_error("FFTW could not create dealiased plans");
  }

  void evaluate(const SpectralField &w, SpectralField &result) override {
    if (w.size() != p_.spectralSize())
      throw std::runtime_error("invalid nonlinear input size");
    for (std::size_t component = 0; component < inverse_.size(); ++component) {
      std::fill(padded_.begin(), padded_.end(), Complex{});
      fillComponent(w, component);
      inverse_[component].execute();
    }
    const std::size_t count = product_.size();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (count >= 16384)
#endif
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(count); ++i) {
      product_[static_cast<std::size_t>(i)] =
          fields_[0][static_cast<std::size_t>(i)] *
              fields_[1][static_cast<std::size_t>(i)] -
          fields_[2][static_cast<std::size_t>(i)] *
              fields_[3][static_cast<std::size_t>(i)];
    }
    forward_.execute();
    extract(result);
  }

private:
  void fillComponent(const SpectralField &w, std::size_t component) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (p_.spectralSize() >= 16384)
#endif
    for (std::ptrdiff_t rawY = 0; rawY < static_cast<std::ptrdiff_t>(p_.ny);
         ++rawY) {
      const std::size_t y = static_cast<std::size_t>(rawY);
      const std::size_t py = paddedWaveRow(y, p_.ny, p_.my());
      const double ky = waveNumberY(p_, y);
      for (std::size_t x = 0; x < p_.nxf(); ++x) {
        const double kx = waveNumberX(p_, x);
        padded_[spectralIndex(x, py, p_.mxf())] =
            w[spectralIndex(x, y, p_.nxf())] *
            nonlinearFactor(component, kx, ky);
      }
    }
  }

  void extract(SpectralField &result) {
    result.resize(p_.spectralSize());
    const double scale = 1.0 / static_cast<double>(p_.mx() * p_.my());
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (p_.spectralSize() >= 16384)
#endif
    for (std::ptrdiff_t rawY = 0; rawY < static_cast<std::ptrdiff_t>(p_.ny);
         ++rawY) {
      const std::size_t y = static_cast<std::size_t>(rawY);
      const std::size_t py = paddedWaveRow(y, p_.ny, p_.my());
      for (std::size_t x = 0; x < p_.nxf(); ++x)
        result[spectralIndex(x, y, p_.nxf())] =
            padded_[spectralIndex(x, py, p_.mxf())] * scale;
    }
    enforceRealityConstraints(result, p_);
  }

  Parameters p_;
  FftwSpectralField padded_;
  FftwRealField product_;
  std::array<FftwRealField, 4> fields_;
  std::array<FftwPlan, 4> inverse_;
  FftwPlan forward_;
};
} // namespace

void backendInitialize(int &, char **&) {}
void backendFinalize() {
#ifdef NS2D_HAVE_FFTW_THREADS
  if (fftwThreadsInitialized) {
    fftw_cleanup_threads();
    fftwThreadsInitialized = false;
  }
#endif
}
void backendAbort(int) {}
void backendBarrier() {}
bool backendIsRoot() { return true; }
const char *backendName() {
#ifdef _OPENMP
  return "CPU/OpenMP";
#else
  return "CPU";
#endif
}
std::uint64_t backendSynchronizeSeed(std::uint64_t seed) { return seed; }

std::unique_ptr<NonlinearBackend> makeBackend(const Parameters &p) {
  int threads = 1;
#ifdef _OPENMP
  threads = p.threadCount > 0 ? p.threadCount : omp_get_max_threads();
  omp_set_num_threads(threads);
#endif
  if (threads > 1) {
#ifdef NS2D_HAVE_FFTW_THREADS
    if (!fftw_init_threads())
      throw std::runtime_error("FFTW thread initialization failed");
    fftwThreadsInitialized = true;
    fftw_plan_with_nthreads(threads);
#else
    // OpenMP loops still use the requested thread count; FFTW remains serial.
#endif
  }
  return std::make_unique<CpuBackend>(p);
}
