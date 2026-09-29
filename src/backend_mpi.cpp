#include "backend.hpp"
#include "fftw_utils.hpp"
#include "spectral.hpp"

#include <fftw3-mpi.h>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
int rank = 0;
#ifdef NS2D_HAVE_FFTW_THREADS
bool fftwThreadsInitialized = false;
#endif

void mpiCheck(int status, const char *operation) {
    if (status == MPI_SUCCESS)
        return;
    char message[MPI_MAX_ERROR_STRING]{};
    int length = 0;
    MPI_Error_string(status, message, &length);
    throw std::runtime_error(std::string(operation) + ": " +
                             std::string(message, static_cast<std::size_t>(length)));
}

class MpiBackend final : public NonlinearBackend {
  public:
    explicit MpiBackend(const Parameters &parameters) : parameters_(parameters) {
        const std::size_t complexValues = parameters.spectralSize();
        if (complexValues > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::runtime_error("spectral field exceeds MPI count limit");
        allocLocal_ = fftw_mpi_local_size_2d(static_cast<ptrdiff_t>(parameters.my()),
                                             static_cast<ptrdiff_t>(parameters.mxf()),
                                             MPI_COMM_WORLD, &localRows_, &firstRow_);
        if (allocLocal_ <= 0 && parameters.my() > 0)
            throw std::runtime_error("FFTW-MPI returned an invalid local size");
        padded_.resize(static_cast<std::size_t>(allocLocal_));
        product_.resize(static_cast<std::size_t>(2 * allocLocal_));
        for (auto &field : fields_)
            field.resize(static_cast<std::size_t>(2 * allocLocal_));
        local_.resize(complexValues);
        for (const NonlinearComponent component : nonlinearComponents) {
            const std::size_t field = componentIndex(component);
            inverse_[field].reset(fftw_mpi_plan_dft_c2r_2d(
                static_cast<ptrdiff_t>(parameters.my()), static_cast<ptrdiff_t>(parameters.mx()),
                fftwData(padded_), fields_[field].data(), MPI_COMM_WORLD, FFTW_ESTIMATE));
        }
        forward_.reset(fftw_mpi_plan_dft_r2c_2d(
            static_cast<ptrdiff_t>(parameters.my()), static_cast<ptrdiff_t>(parameters.mx()),
            product_.data(), fftwData(padded_), MPI_COMM_WORLD, FFTW_ESTIMATE));
        if (std::any_of(inverse_.begin(), inverse_.end(),
                        [](const FftwPlan &plan) { return !plan; }) ||
            !forward_)
            throw std::runtime_error("FFTW-MPI could not create dealiased plans");
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
        const std::size_t rowStride = 2 * parameters_.mxf();
        const std::size_t count = static_cast<std::size_t>(localRows_) * rowStride;
        const auto &vorticityX = fields_[componentIndex(NonlinearComponent::vorticityX)];
        const auto &streamfunctionY = fields_[componentIndex(NonlinearComponent::streamfunctionY)];
        const auto &vorticityY = fields_[componentIndex(NonlinearComponent::vorticityY)];
        const auto &streamfunctionX = fields_[componentIndex(NonlinearComponent::streamfunctionX)];
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (count >= 16384)
#endif
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(count); ++i) {
            const auto n = static_cast<std::size_t>(i);
            product_[n] = vorticityX[n] * streamfunctionY[n] - vorticityY[n] * streamfunctionX[n];
        }
        std::fill(product_.begin() + static_cast<std::ptrdiff_t>(count), product_.end(), 0.0);
        forward_.execute();

        std::fill(local_.begin(), local_.end(), Complex{});
        const double scale = 1.0 / static_cast<double>(parameters_.mx() * parameters_.my());
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (parameters_.spectralSize() >= 16384)
#endif
        for (std::ptrdiff_t rawY = 0; rawY < static_cast<std::ptrdiff_t>(parameters_.ny); ++rawY) {
            const std::size_t y = static_cast<std::size_t>(rawY);
            const long ky = signedWave(y, parameters_.ny);
            const ptrdiff_t py = ky >= 0 ? static_cast<ptrdiff_t>(ky)
                                         : static_cast<ptrdiff_t>(parameters_.my()) + ky;
            if (py < firstRow_ || py >= firstRow_ + localRows_)
                continue;
            const std::size_t localY = static_cast<std::size_t>(py - firstRow_);
            for (std::size_t x = 0; x < parameters_.nxf(); ++x)
                local_[spectralIndex(x, y, parameters_.nxf())] =
                    padded_[spectralIndex(x, localY, parameters_.mxf())] * scale;
        }
        result.resize(local_.size());
        mpiCheck(MPI_Allreduce(local_.data(), result.data(), static_cast<int>(local_.size()),
                               MPI_CXX_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD),
                 "MPI_Allreduce nonlinear result");
        enforceRealityConstraints(result, parameters_);
    }

  private:
    void fillComponent(const SpectralField &vorticity, NonlinearComponent component) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (parameters_.spectralSize() >= 16384)
#endif
        for (std::ptrdiff_t rawY = 0; rawY < static_cast<std::ptrdiff_t>(parameters_.ny); ++rawY) {
            const std::size_t y = static_cast<std::size_t>(rawY);
            const long kyIndex = signedWave(y, parameters_.ny);
            const ptrdiff_t py = kyIndex >= 0 ? static_cast<ptrdiff_t>(kyIndex)
                                              : static_cast<ptrdiff_t>(parameters_.my()) + kyIndex;
            if (py < firstRow_ || py >= firstRow_ + localRows_)
                continue;
            const std::size_t localY = static_cast<std::size_t>(py - firstRow_);
            const double ky = waveNumberY(parameters_, y);
            for (std::size_t x = 0; x < parameters_.nxf(); ++x) {
                const double kx = waveNumberX(parameters_, x);
                padded_[spectralIndex(x, localY, parameters_.mxf())] =
                    vorticity[spectralIndex(x, y, parameters_.nxf())] *
                    nonlinearFactor(component, kx, ky);
            }
        }
    }

    Parameters parameters_;
    ptrdiff_t allocLocal_ = 0, localRows_ = 0, firstRow_ = 0;
    FftwSpectralField padded_;
    FftwRealField product_;
    std::array<FftwRealField, 4> fields_;
    SpectralField local_;
    std::array<FftwPlan, 4> inverse_;
    FftwPlan forward_;
};
} // namespace

void backendInitialize(int &argc, char **&argv) {
    int provided = MPI_THREAD_SINGLE;
#ifdef _OPENMP
    constexpr int required = MPI_THREAD_FUNNELED;
#else
    constexpr int required = MPI_THREAD_SINGLE;
#endif
    mpiCheck(MPI_Init_thread(&argc, &argv, required, &provided), "MPI_Init_thread");
    mpiCheck(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "MPI_Comm_rank");
    if (provided < required) {
        if (rank == 0)
            throw std::runtime_error("MPI runtime lacks required thread support");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
#if defined(NS2D_HAVE_FFTW_THREADS) && defined(_OPENMP)
    if (!fftw_init_threads())
        throw std::runtime_error("FFTW thread initialization failed");
    fftwThreadsInitialized = true;
#endif
    fftw_mpi_init();
}

void backendFinalize() {
    fftw_mpi_cleanup();
#ifdef NS2D_HAVE_FFTW_THREADS
    if (fftwThreadsInitialized) {
        fftw_cleanup_threads();
        fftwThreadsInitialized = false;
    }
#endif
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized)
        MPI_Finalize();
}
void backendBarrier() { mpiCheck(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier"); }
void backendAbort(int exitCode) { MPI_Abort(MPI_COMM_WORLD, exitCode); }
bool backendIsRoot() { return rank == 0; }
const char *backendName() {
#ifdef _OPENMP
    return "MPI/OpenMP";
#else
    return "MPI";
#endif
}
std::uint64_t backendSynchronizeSeed(std::uint64_t seed) {
    mpiCheck(MPI_Bcast(&seed, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD), "MPI_Bcast random seed");
    return seed;
}

std::unique_ptr<NonlinearBackend> makeBackend(const Parameters &parameters) {
    int threads = 1;
#ifdef _OPENMP
    threads = parameters.threadCount > 0 ? parameters.threadCount : omp_get_max_threads();
    omp_set_num_threads(threads);
#endif
    if (threads > 1) {
#ifdef NS2D_HAVE_FFTW_THREADS
        fftw_plan_with_nthreads(threads);
#else
        // OpenMP loops still use the requested thread count; FFTW remains serial.
#endif
    }
    return std::make_unique<MpiBackend>(parameters);
}
