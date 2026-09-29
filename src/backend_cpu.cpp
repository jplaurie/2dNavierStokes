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
    explicit CpuBackend(const Parameters &parameters)
        : parameters_(parameters), padded_(parameters.my() * parameters.mxf()),
          product_(parameters.mx() * parameters.my()),
          fields_{FftwRealField(parameters.mx() * parameters.my()),
                  FftwRealField(parameters.mx() * parameters.my()),
                  FftwRealField(parameters.mx() * parameters.my()),
                  FftwRealField(parameters.mx() * parameters.my())} {
        for (const NonlinearComponent component : nonlinearComponents) {
            const std::size_t field = componentIndex(component);
            inverse_[field].reset(fftw_plan_dft_c2r_2d(
                static_cast<int>(parameters.my()), static_cast<int>(parameters.mx()),
                fftwData(padded_), fields_[field].data(), FFTW_ESTIMATE));
        }
        forward_.reset(fftw_plan_dft_r2c_2d(static_cast<int>(parameters.my()),
                                            static_cast<int>(parameters.mx()), product_.data(),
                                            fftwData(padded_), FFTW_ESTIMATE));
        if (std::any_of(inverse_.begin(), inverse_.end(),
                        [](const FftwPlan &plan) { return !plan; }) ||
            !forward_)
            throw std::runtime_error("FFTW could not create dealiased plans");
    }

    void evaluate(const SpectralField &vorticity, SpectralField &result) override {
        if (vorticity.size() != parameters_.spectralSize())
            throw std::runtime_error("invalid nonlinear input size");
        for (const NonlinearComponent component : nonlinearComponents) {
            const std::size_t field = componentIndex(component);
            std::fill(padded_.begin(), padded_.end(), Complex{});
            fillComponent(vorticity, component);
            inverse_[field].execute();
        }
        const std::size_t count = product_.size();
        const auto &vorticityX = fields_[componentIndex(NonlinearComponent::vorticityX)];
        const auto &streamfunctionY = fields_[componentIndex(NonlinearComponent::streamfunctionY)];
        const auto &vorticityY = fields_[componentIndex(NonlinearComponent::vorticityY)];
        const auto &streamfunctionX = fields_[componentIndex(NonlinearComponent::streamfunctionX)];
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (count >= 16384)
#endif
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(count); ++i) {
            const auto index = static_cast<std::size_t>(i);
            product_[index] = vorticityX[index] * streamfunctionY[index] -
                              vorticityY[index] * streamfunctionX[index];
        }
        forward_.execute();
        extract(result);
    }

  private:
    void fillComponent(const SpectralField &vorticity, NonlinearComponent component) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (parameters_.spectralSize() >= 16384)
#endif
        for (std::ptrdiff_t rawY = 0; rawY < static_cast<std::ptrdiff_t>(parameters_.ny); ++rawY) {
            const std::size_t y = static_cast<std::size_t>(rawY);
            const std::size_t py = paddedWaveRow(y, parameters_.ny, parameters_.my());
            const double ky = waveNumberY(parameters_, y);
            for (std::size_t x = 0; x < parameters_.nxf(); ++x) {
                const double kx = waveNumberX(parameters_, x);
                padded_[spectralIndex(x, py, parameters_.mxf())] =
                    vorticity[spectralIndex(x, y, parameters_.nxf())] *
                    nonlinearFactor(component, kx, ky);
            }
        }
    }

    void extract(SpectralField &result) {
        result.resize(parameters_.spectralSize());
        const double scale = 1.0 / static_cast<double>(parameters_.mx() * parameters_.my());
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (parameters_.spectralSize() >= 16384)
#endif
        for (std::ptrdiff_t rawY = 0; rawY < static_cast<std::ptrdiff_t>(parameters_.ny); ++rawY) {
            const std::size_t y = static_cast<std::size_t>(rawY);
            const std::size_t py = paddedWaveRow(y, parameters_.ny, parameters_.my());
            for (std::size_t x = 0; x < parameters_.nxf(); ++x)
                result[spectralIndex(x, y, parameters_.nxf())] =
                    padded_[spectralIndex(x, py, parameters_.mxf())] * scale;
        }
        enforceRealityConstraints(result, parameters_);
    }

    Parameters parameters_;
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

std::unique_ptr<NonlinearBackend> makeBackend(const Parameters &parameters) {
    int threads = 1;
#ifdef _OPENMP
    threads = parameters.threadCount > 0 ? parameters.threadCount : omp_get_max_threads();
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
    return std::make_unique<CpuBackend>(parameters);
}
