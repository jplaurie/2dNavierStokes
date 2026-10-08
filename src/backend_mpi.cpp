#include "backend.hpp"
#include "fftw_utils.hpp"
#include "host_stepper.hpp"
#include "parallel.hpp"
#include "spectral.hpp"

#include <fftw3-mpi.h>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
int rank = 0;
#ifdef NS2D_HAVE_FFTW_THREADS
bool fftwThreadsInitialized = false;
#endif

struct LocalMode {
    std::size_t baseIndex;
    std::size_t paddedIndex;
    double kx;
    double ky;
};

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
        if (parameters.spectralSize() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::runtime_error("spectral field exceeds MPI count limit");
        allocLocal_ = fftw_mpi_local_size_2d(static_cast<ptrdiff_t>(parameters.my()),
                                             static_cast<ptrdiff_t>(parameters.mxf()),
                                             MPI_COMM_WORLD, &localRows_, &firstRow_);
        if (allocLocal_ < localRows_ * static_cast<ptrdiff_t>(parameters.mxf()))
            throw std::runtime_error("FFTW-MPI returned an invalid local size");
        padded_.resize(static_cast<std::size_t>(allocLocal_));
        product_.resize(static_cast<std::size_t>(2 * allocLocal_));
        for (auto &field : fields_)
            field.resize(static_cast<std::size_t>(2 * allocLocal_));
        for (const NonlinearComponent component : nonlinearComponents) {
            const std::size_t field = componentIndex(component);
            inverse_[field].reset(fftw_mpi_plan_dft_c2r_2d(
                static_cast<ptrdiff_t>(parameters.my()), static_cast<ptrdiff_t>(parameters.mx()),
                fftwData(padded_), fields_[field].data(), MPI_COMM_WORLD, fftwPlanningFlags()));
        }
        forward_.reset(fftw_mpi_plan_dft_r2c_2d(
            static_cast<ptrdiff_t>(parameters.my()), static_cast<ptrdiff_t>(parameters.mx()),
            product_.data(), fftwData(padded_), MPI_COMM_WORLD, fftwPlanningFlags()));
        if (std::any_of(inverse_.begin(), inverse_.end(),
                        [](const FftwPlan &plan) { return !plan; }) ||
            !forward_)
            throw std::runtime_error("FFTW-MPI could not create dealiased plans");

        for (std::size_t y = 0; y < parameters.ny; ++y) {
            const long kyIndex = signedWave(y, parameters.ny);
            const ptrdiff_t paddedRow = kyIndex >= 0
                                            ? static_cast<ptrdiff_t>(kyIndex)
                                            : static_cast<ptrdiff_t>(parameters.my()) + kyIndex;
            if (paddedRow < firstRow_ || paddedRow >= firstRow_ + localRows_)
                continue;
            const std::size_t localY = static_cast<std::size_t>(paddedRow - firstRow_);
            for (std::size_t x = 0; x < parameters.nxf(); ++x)
                localModes_.push_back({spectralIndex(x, y, parameters.nxf()),
                                       spectralIndex(x, localY, parameters.mxf()),
                                       waveNumberX(parameters, x), waveNumberY(parameters, y)});
        }
        localRealityLine_.resize(parameters_.ny);
        globalRealityLine_.resize(parameters_.ny);
    }

    void evaluate(const SpectralField &vorticity, SpectralField &result) override {
        if (vorticity.size() != parameters_.spectralSize())
            throw std::runtime_error("invalid nonlinear input size");
        SpectralField localState(static_cast<std::size_t>(allocLocal_));
        for (const LocalMode &mode : localModes_)
            localState[mode.paddedIndex] = vorticity[mode.baseIndex];
        SpectralField localResult;
        evaluateLocal(localState, localResult);
        SpectralField localBase(parameters_.spectralSize());
        for (const LocalMode &mode : localModes_)
            localBase[mode.baseIndex] = localResult[mode.paddedIndex];
        result.resize(localBase.size());
        mpiCheck(MPI_Allreduce(localBase.data(), result.data(), static_cast<int>(localBase.size()),
                               MPI_CXX_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD),
                 "MPI_Allreduce standalone nonlinear result");
        enforceRealityConstraints(result, parameters_);
    }

    NoiseLayout noiseLayout() const override { return NoiseLayout::fullField; }

    void initializeTimeStepping(const IntegrationCoefficients &coefficients,
                                const std::vector<double> &deterministicForcing,
                                const std::vector<std::size_t> &,
                                const SpectralField &state) override {
        if (state.size() != parameters_.spectralSize())
            throw std::runtime_error("invalid initial state size");
        state_.assign(static_cast<std::size_t>(allocLocal_), Complex{});
        if (!deterministicForcing.empty())
            deterministicForcing_.assign(state_.size(), 0.0);
        const auto globalFields = coefficients.fields();
        const auto localFields = coefficients_.fields();
        for (std::size_t field = 0; field < globalFields.size(); ++field)
            if (!globalFields[field]->empty())
                localFields[field]->assign(state_.size(), Complex{});
        for (const LocalMode &mode : localModes_) {
            state_[mode.paddedIndex] = state[mode.baseIndex];
            if (!deterministicForcing.empty())
                deterministicForcing_[mode.paddedIndex] = deterministicForcing[mode.baseIndex];
            for (std::size_t field = 0; field < globalFields.size(); ++field)
                if (!globalFields[field]->empty())
                    (*localFields[field])[mode.paddedIndex] =
                        (*globalFields[field])[mode.baseIndex];
        }
        localNoise_.resize(state_.size());
        workspace_.initialize(state_.size(), parameters_.nonlinearStageCount());
    }

    void advanceTimeStep(const SpectralField &noise) override {
        if (!noise.empty() && noise.size() != parameters_.spectralSize())
            throw std::runtime_error("invalid MPI stochastic-noise field");
        std::fill(localNoise_.begin(), localNoise_.end(), Complex{});
        if (!noise.empty())
            for (const LocalMode &mode : localModes_)
                localNoise_[mode.paddedIndex] = noise[mode.baseIndex];
        const auto rhs = [&](const SpectralField &input, SpectralField &output) {
            evaluateLocal(input, output);
            if (!deterministicForcing_.empty())
                forEachIndex(output.size(),
                             [&](std::size_t i) { output[i] += deterministicForcing_[i]; });
        };
        advanceHostTimeStep(
            parameters_, coefficients_, state_, noise.empty() ? SpectralField{} : localNoise_,
            workspace_, rhs,
            [&](SpectralField &localState) { enforceLocalConstraints(localState); });
    }

    void downloadState(SpectralField &state) override { reduceLocalField(state_, state); }

    void evaluateCurrent(SpectralField &nonlinearTerm) override {
        SpectralField localResult;
        evaluateLocal(state_, localResult);
        reduceLocalField(localResult, nonlinearTerm);
    }

  private:
    void fillComponent(const SpectralField &localState, NonlinearComponent component) {
        std::fill(padded_.begin(), padded_.end(), Complex{});
        forEachIndex(localModes_.size(), [&](std::size_t i) {
            const LocalMode &mode = localModes_[i];
            padded_[mode.paddedIndex] =
                localState[mode.paddedIndex] * nonlinearFactor(component, mode.kx, mode.ky);
        });
    }

    void evaluateLocal(const SpectralField &localState, SpectralField &result) {
        if (localState.size() != static_cast<std::size_t>(allocLocal_))
            throw std::runtime_error("invalid distributed nonlinear input size");
        for (const NonlinearComponent component : nonlinearComponents) {
            const std::size_t field = componentIndex(component);
            fillComponent(localState, component);
            inverse_[field].execute();
        }
        const std::size_t rowStride = 2 * parameters_.mxf();
        const std::size_t count = static_cast<std::size_t>(localRows_) * rowStride;
        const auto &vorticityX = fields_[componentIndex(NonlinearComponent::vorticityX)];
        const auto &streamfunctionY = fields_[componentIndex(NonlinearComponent::streamfunctionY)];
        const auto &vorticityY = fields_[componentIndex(NonlinearComponent::vorticityY)];
        const auto &streamfunctionX = fields_[componentIndex(NonlinearComponent::streamfunctionX)];
        forEachIndex(count, [&](std::size_t i) {
            product_[i] = vorticityX[i] * streamfunctionY[i] - vorticityY[i] * streamfunctionX[i];
        });
        std::fill(product_.begin() + static_cast<std::ptrdiff_t>(count), product_.end(), 0.0);
        forward_.execute();
        result.assign(static_cast<std::size_t>(allocLocal_), Complex{});
        const double scale = 1.0 / static_cast<double>(parameters_.mx() * parameters_.my());
        forEachIndex(localModes_.size(), [&](std::size_t i) {
            const LocalMode &mode = localModes_[i];
            result[mode.paddedIndex] = padded_[mode.paddedIndex] * scale;
        });
        enforceLocalConstraints(result);
    }

    void enforceLocalConstraints(SpectralField &field) {
        std::fill(localRealityLine_.begin(), localRealityLine_.end(), Complex{});
        for (const LocalMode &mode : localModes_) {
            const std::size_t x = mode.baseIndex % parameters_.nxf();
            const std::size_t y = mode.baseIndex / parameters_.nxf();
            if (mode.baseIndex == 0 || x == parameters_.nx / 2 || y == parameters_.ny / 2)
                field[mode.paddedIndex] = Complex{};
            else if (x == 0 && y < parameters_.ny / 2)
                localRealityLine_[y] = field[mode.paddedIndex];
        }
        mpiCheck(MPI_Allreduce(localRealityLine_.data(), globalRealityLine_.data(),
                               static_cast<int>(parameters_.ny), MPI_CXX_DOUBLE_COMPLEX, MPI_SUM,
                               MPI_COMM_WORLD),
                 "MPI_Allreduce real-transform symmetry line");
        for (const LocalMode &mode : localModes_) {
            const std::size_t x = mode.baseIndex % parameters_.nxf();
            const std::size_t y = mode.baseIndex / parameters_.nxf();
            if (x == 0 && y > parameters_.ny / 2)
                field[mode.paddedIndex] = std::conj(globalRealityLine_[parameters_.ny - y]);
        }
    }

    void reduceLocalField(const SpectralField &localField, SpectralField &globalField) const {
        SpectralField localBase(parameters_.spectralSize());
        for (const LocalMode &mode : localModes_)
            localBase[mode.baseIndex] = localField[mode.paddedIndex];
        if (rank == 0)
            globalField.resize(localBase.size());
        else
            globalField.clear();
        mpiCheck(MPI_Reduce(localBase.data(), rank == 0 ? globalField.data() : nullptr,
                            static_cast<int>(localBase.size()), MPI_CXX_DOUBLE_COMPLEX, MPI_SUM, 0,
                            MPI_COMM_WORLD),
                 "MPI_Reduce distributed spectral field");
        if (rank == 0)
            enforceRealityConstraints(globalField, parameters_);
    }

    Parameters parameters_;
    ptrdiff_t allocLocal_ = 0, localRows_ = 0, firstRow_ = 0;
    FftwSpectralField padded_;
    FftwRealField product_;
    std::array<FftwRealField, 4> fields_;
    std::array<FftwPlan, 4> inverse_;
    FftwPlan forward_;
    SpectralField state_, localNoise_;
    SpectralField localRealityLine_, globalRealityLine_;
    std::vector<double> deterministicForcing_;
    IntegrationCoefficients coefficients_;
    HostIntegrationWorkspace workspace_;
    std::vector<LocalMode> localModes_;
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
    if (provided < required)
        throw std::runtime_error("MPI runtime lacks required thread support");
#if defined(NS2D_HAVE_FFTW_THREADS) && defined(_OPENMP)
    if (!fftw_init_threads())
        throw std::runtime_error("FFTW thread initialization failed");
    fftwThreadsInitialized = true;
#endif
    fftw_mpi_init();
}

void backendFinalize() {
    fftw_mpi_gather_wisdom(MPI_COMM_WORLD);
    if (rank == 0)
        saveFftwWisdom();
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
void backendAbort(int exitCode) { MPI_Abort(MPI_COMM_WORLD, exitCode); }
void backendBarrier() { mpiCheck(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier"); }
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
#endif
    }
    configureFftw(parameters, rank == 0);
    fftw_mpi_broadcast_wisdom(MPI_COMM_WORLD);
    return std::make_unique<MpiBackend>(parameters);
}
