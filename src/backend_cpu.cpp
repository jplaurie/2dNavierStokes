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
bool fftwThreadsInitialized = false;

class CpuBackend final : public NonlinearBackend {
public:
  explicit CpuBackend(const Parameters &p)
      : p_(p), padded_(p.my() * p.mxf()), product_(p.mx() * p.my()),
        fields_{FftwRealField(p.mx() * p.my()), FftwRealField(p.mx() * p.my()),
                FftwRealField(p.mx() * p.my()),
                FftwRealField(p.mx() * p.my())} {
    for (std::size_t component = 0; component < inverse_.size(); ++component)
      inverse_[component] = fftw_plan_dft_c2r_2d(
          static_cast<int>(p.my()), static_cast<int>(p.mx()), fftwData(padded_),
          fields_[component].data(), FFTW_ESTIMATE);
    forward_ =
        fftw_plan_dft_r2c_2d(static_cast<int>(p.my()), static_cast<int>(p.mx()),
                             product_.data(), fftwData(padded_), FFTW_ESTIMATE);
    if (!inverse_[0] || !inverse_[1] || !inverse_[2] || !inverse_[3] ||
        !forward_) {
      releasePlans();
      throw std::runtime_error("FFTW could not create dealiased plans");
    }
  }

  ~CpuBackend() override { releasePlans(); }

  void evaluate(const SpectralField &w, SpectralField &result) override {
    if (w.size() != p_.ny * p_.nxf())
      throw std::runtime_error("invalid nonlinear input size");
    for (std::size_t component = 0; component < inverse_.size(); ++component) {
      std::fill(padded_.begin(), padded_.end(), Complex{});
      fillComponent(w, component);
      fftw_execute(inverse_[component]);
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
    fftw_execute(forward_);
    extract(result);
  }

private:
  void releasePlans() noexcept {
    for (fftw_plan &plan : inverse_) {
      if (plan)
        fftw_destroy_plan(plan);
      plan = nullptr;
    }
    if (forward_)
      fftw_destroy_plan(forward_);
    forward_ = nullptr;
  }

  void fillComponent(const SpectralField &w, std::size_t component) {
    const Complex imaginary(0.0, 1.0);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (p_.ny * p_.nxf() >= 16384)
#endif
    for (std::ptrdiff_t rawY = 0; rawY < static_cast<std::ptrdiff_t>(p_.ny);
         ++rawY) {
      const std::size_t y = static_cast<std::size_t>(rawY);
      const long kyIndex = signedWave(y, p_.ny);
      const std::size_t py =
          kyIndex >= 0
              ? static_cast<std::size_t>(kyIndex)
              : static_cast<std::size_t>(static_cast<long>(p_.my()) + kyIndex);
      const double ky = 2.0 * nsPi * static_cast<double>(kyIndex) / p_.ly();
      for (std::size_t x = 0; x < p_.nxf(); ++x) {
        const double kx = 2.0 * nsPi * static_cast<double>(x) / p_.lx();
        const double k2 = kx * kx + ky * ky;
        Complex factor;
        if (component == 0)
          factor = imaginary * kx;
        else if (component == 1)
          factor = k2 == 0.0 ? Complex{} : -imaginary * ky / k2;
        else if (component == 2)
          factor = imaginary * ky;
        else
          factor = k2 == 0.0 ? Complex{} : -imaginary * kx / k2;
        padded_[spectralIndex(x, py, p_.mxf())] =
            w[spectralIndex(x, y, p_.nxf())] * factor;
      }
    }
  }

  void extract(SpectralField &result) {
    result.resize(p_.ny * p_.nxf());
    const double scale = 1.0 / static_cast<double>(p_.mx() * p_.my());
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (p_.ny * p_.nxf() >= 16384)
#endif
    for (std::ptrdiff_t rawY = 0; rawY < static_cast<std::ptrdiff_t>(p_.ny);
         ++rawY) {
      const std::size_t y = static_cast<std::size_t>(rawY);
      const long ky = signedWave(y, p_.ny);
      const std::size_t py =
          ky >= 0 ? static_cast<std::size_t>(ky)
                  : static_cast<std::size_t>(static_cast<long>(p_.my()) + ky);
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
  std::array<fftw_plan, 4> inverse_{};
  fftw_plan forward_ = nullptr;
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
