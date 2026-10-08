#include <cuda_runtime.h>
#include <cufft.h>

__host__ __device__ inline cufftDoubleComplex operator+(cufftDoubleComplex a,
                                                        cufftDoubleComplex b) {
    return {a.x + b.x, a.y + b.y};
}
__host__ __device__ inline cufftDoubleComplex operator-(cufftDoubleComplex a,
                                                        cufftDoubleComplex b) {
    return {a.x - b.x, a.y - b.y};
}
__host__ __device__ inline cufftDoubleComplex operator*(cufftDoubleComplex a,
                                                        cufftDoubleComplex b) {
    return {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
}
__host__ __device__ inline cufftDoubleComplex operator*(double a, cufftDoubleComplex b) {
    return {a * b.x, a * b.y};
}

#include "backend.hpp"
#include "fftw_utils.hpp"
#include "spectral.hpp"

#include <array>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
#ifdef NS2D_CUDA_MIXED
using TransformComplex = cufftComplex;
using TransformReal = float;
constexpr cufftType inverseTransformType = CUFFT_C2R;
constexpr cufftType forwardTransformType = CUFFT_R2C;
#else
using TransformComplex = cufftDoubleComplex;
using TransformReal = double;
constexpr cufftType inverseTransformType = CUFFT_Z2D;
constexpr cufftType forwardTransformType = CUFFT_D2Z;
#endif

void cudaCheck(cudaError_t status, const char *operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
void cufftCheck(cufftResult status, const char *operation) {
    if (status != CUFFT_SUCCESS)
        throw std::runtime_error(std::string(operation) + " failed (cuFFT status " +
                                 std::to_string(static_cast<int>(status)) + ")");
}

__global__ void populateComponents(const cufftDoubleComplex *input, TransformComplex *fields,
                                   std::size_t ny, std::size_t nxf, std::size_t my, std::size_t mxf,
                                   double lx, double ly) {
    const std::size_t flat = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t paddedCount = my * mxf;
    if (flat >= paddedCount)
        return;
    const std::size_t py = flat / mxf;
    const std::size_t x = flat % mxf;
    TransformComplex *vorticityX = fields;
    TransformComplex *streamfunctionY = fields + paddedCount;
    TransformComplex *vorticityY = fields + 2 * paddedCount;
    TransformComplex *streamfunctionX = fields + 3 * paddedCount;
    long kyIndex = 0;
    std::size_t y = 0;
    bool retained = x < nxf;
    if (retained && py < ny / 2) {
        kyIndex = static_cast<long>(py);
        y = py;
    } else if (retained && py >= my - ny / 2) {
        kyIndex = static_cast<long>(py) - static_cast<long>(my);
        y = static_cast<std::size_t>(static_cast<long>(ny) + kyIndex);
    } else {
        retained = false;
    }
    if (!retained) {
        vorticityX[flat] = {0, 0};
        streamfunctionY[flat] = {0, 0};
        vorticityY[flat] = {0, 0};
        streamfunctionX[flat] = {0, 0};
        return;
    }
    constexpr double twoPi = 6.283185307179586476925286766559;
    const double kx = twoPi * static_cast<double>(x) / lx;
    const double ky = twoPi * static_cast<double>(kyIndex) / ly;
    const double k2 = kx * kx + ky * ky;
    const cufftDoubleComplex source = input[y * nxf + x];
    const auto multiplyByImaginary = [source](double factor) {
        return TransformComplex{static_cast<TransformReal>(-source.y * factor),
                                static_cast<TransformReal>(source.x * factor)};
    };
    vorticityX[flat] = multiplyByImaginary(kx);
    streamfunctionY[flat] = multiplyByImaginary(k2 == 0.0 ? 0.0 : -ky / k2);
    vorticityY[flat] = multiplyByImaginary(ky);
    streamfunctionX[flat] = multiplyByImaginary(k2 == 0.0 ? 0.0 : -kx / k2);
}

__global__ void multiplyFields(const TransformReal *fields, TransformReal *product,
                               std::size_t count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count)
        return;
    const TransformReal vorticityX = fields[index];
    const TransformReal streamfunctionY = fields[count + index];
    const TransformReal vorticityY = fields[2 * count + index];
    const TransformReal streamfunctionX = fields[3 * count + index];
    product[index] = vorticityX * streamfunctionY - vorticityY * streamfunctionX;
}

__global__ void extractModes(const TransformComplex *padded, cufftDoubleComplex *result,
                             std::size_t ny, std::size_t nxf, std::size_t my, std::size_t mxf,
                             double scale) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= ny * nxf)
        return;
    const std::size_t y = index / nxf;
    const std::size_t x = index % nxf;
    const long ky =
        y < ny / 2 ? static_cast<long>(y) : static_cast<long>(y) - static_cast<long>(ny);
    const std::size_t py = ky >= 0 ? static_cast<std::size_t>(ky)
                                   : static_cast<std::size_t>(static_cast<long>(my) + ky);
    const TransformComplex value = padded[py * mxf + x];
    result[index] = {static_cast<double>(value.x) * scale, static_cast<double>(value.y) * scale};
}

template <class T> class DeviceBuffer {
  public:
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
    ~DeviceBuffer() { cudaFree(data_); }
    void allocate(std::size_t count) {
        if (data_)
            throw std::logic_error("device buffer already allocated");
        if (count)
            cudaCheck(cudaMalloc(&data_, count * sizeof(T)), "allocate CUDA buffer");
    }
    T *data() const { return data_; }

  private:
    T *data_ = nullptr;
};

class CufftPlan {
  public:
    CufftPlan() = default;
    ~CufftPlan() {
        if (plan_)
            cufftDestroy(plan_);
    }
    CufftPlan(const CufftPlan &) = delete;
    CufftPlan &operator=(const CufftPlan &) = delete;
    cufftHandle *address() { return &plan_; }
    [[nodiscard]] cufftHandle get() const { return plan_; }

  private:
    cufftHandle plan_ = 0;
};

class CudaStream {
  public:
    CudaStream() { cudaCheck(cudaStreamCreate(&stream_), "create CUDA integration stream"); }
    ~CudaStream() { cudaStreamDestroy(stream_); }
    CudaStream(const CudaStream &) = delete;
    CudaStream &operator=(const CudaStream &) = delete;
    [[nodiscard]] cudaStream_t get() const { return stream_; }

  private:
    cudaStream_t stream_ = nullptr;
};

struct DeviceStages {
    cufftDoubleComplex *a{}, *b{}, *c{}, *n1{}, *n2{}, *n3{}, *n4{};
};

enum class IntegrationStage : int { a, b, c, finish };

__global__ void constrainModes(cufftDoubleComplex *vorticity, std::size_t ny, std::size_t nxf) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= ny * nxf)
        return;
    const std::size_t x = i % nxf, y = i / nxf;
    if (i == 0 || x == nxf - 1 || y == ny / 2)
        vorticity[i] = {0.0, 0.0};
    else if (x == 0 && y > ny / 2) {
        const cufftDoubleComplex partner = vorticity[(ny - y) * nxf];
        vorticity[i] = {partner.x, -partner.y};
    }
}

__global__ void addForcing(cufftDoubleComplex *n, const double *forcing, std::size_t count) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count)
        n[i].x += forcing[i];
}

__global__ void addSparseNoise(cufftDoubleComplex *vorticity, const cufftDoubleComplex *noise,
                               const std::size_t *indices, std::size_t count) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        const std::size_t destination = indices[i];
        vorticity[destination] = vorticity[destination] + noise[i];
    }
}

__global__ void integrateStage(IntegrationStage stage, Integrator method, double h,
                               std::size_t count,
                               CoefficientPointers<cufftDoubleComplex> coefficients,
                               cufftDoubleComplex *vorticity, DeviceStages stages,
                               const cufftDoubleComplex *noise) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    if (stage == IntegrationStage::a)
        stages.a[i] = integrationStageA(method, h, i, coefficients, vorticity[i], stages.n1[i]);
    else if (stage == IntegrationStage::b)
        stages.b[i] =
            integrationStageB(method, i, coefficients, vorticity[i], stages.n1[i], stages.n2[i]);
    else if (stage == IntegrationStage::c)
        stages.c[i] = integrationStageC(i, coefficients, vorticity[i], stages.n1[i], stages.n3[i]);
    else {
        const cufftDoubleComplex zero{0.0, 0.0};
        vorticity[i] = integrationFinish(
            method, h, i, coefficients, vorticity[i], stages.a[i], stages.n1[i], stages.n2[i],
            stages.n3 ? stages.n3[i] : zero, stages.n4 ? stages.n4[i] : zero);
        if (noise)
            vorticity[i] = vorticity[i] + noise[i];
    }
}

class CudaBackend final : public NonlinearBackend {
  public:
    explicit CudaBackend(const Parameters &parameters)
        : parameters_(parameters), graphEnabled_(parameters.cudaGraphEnabled) {
        static_assert(sizeof(Complex) == sizeof(cufftDoubleComplex));
        if (parameters.mx() > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
            parameters.my() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::runtime_error("CUDA grid dimensions exceed cuFFT limits");
        baseCount_ = parameters.spectralSize();
        paddedCount_ = parameters.my() * parameters.mxf();
        realCount_ = parameters.my() * parameters.mx();
        if (paddedCount_ > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
            realCount_ > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::runtime_error("CUDA transform allocation exceeds cuFFT stride limits");
        input_.allocate(baseCount_);
        fields_.allocate(4 * paddedCount_);
        realFields_.allocate(4 * realCount_);
        product_.allocate(realCount_);
        paddedOutput_.allocate(paddedCount_);
        result_.allocate(baseCount_);

        int dimensions[] = {static_cast<int>(parameters.my()), static_cast<int>(parameters.mx())};
        cufftCheck(cufftPlanMany(inverse_.address(), 2, dimensions, nullptr, 1,
                                 static_cast<int>(paddedCount_), nullptr, 1,
                                 static_cast<int>(realCount_), inverseTransformType, 4),
                   "create batched inverse plan");
        cufftCheck(
            cufftPlan2d(forward_.address(), dimensions[0], dimensions[1], forwardTransformType),
            "create forward plan");
        cufftCheck(cufftSetStream(inverse_.get(), stream_.get()),
                   "attach inverse transforms to integration stream");
        cufftCheck(cufftSetStream(forward_.get(), stream_.get()),
                   "attach forward transform to integration stream");
    }

    ~CudaBackend() override {
        if (graphExec_)
            cudaGraphExecDestroy(graphExec_);
        if (graph_)
            cudaGraphDestroy(graph_);
    }

    NoiseLayout noiseLayout() const override { return NoiseLayout::forcedModes; }

    void initializeTimeStepping(const IntegrationCoefficients &c,
                                const std::vector<double> &forcing,
                                const std::vector<std::size_t> &stochasticIndices,
                                const SpectralField &vorticity) override {
        if (graphExec_) {
            cudaGraphExecDestroy(graphExec_);
            graphExec_ = nullptr;
        }
        if (graph_) {
            cudaGraphDestroy(graph_);
            graph_ = nullptr;
        }
        const auto arrays = c.fields();
        for (std::size_t i = 0; i < arrays.size(); ++i) {
            coefficientBuffers_[i].allocate(arrays[i]->size());
            if (!arrays[i]->empty())
                cudaCheck(cudaMemcpy(coefficientBuffers_[i].data(), arrays[i]->data(),
                                     arrays[i]->size() * sizeof(cufftDoubleComplex),
                                     cudaMemcpyHostToDevice),
                          "upload integration coefficients");
        }
        coefficients_ = {coefficientBuffers_[0].data(), coefficientBuffers_[1].data(),
                         coefficientBuffers_[2].data(), coefficientBuffers_[3].data(),
                         coefficientBuffers_[4].data(), coefficientBuffers_[5].data(),
                         coefficientBuffers_[6].data(), coefficientBuffers_[7].data(),
                         coefficientBuffers_[8].data(), coefficientBuffers_[9].data()};
        for (std::size_t i : {0UL, 3UL, 4UL})
            stageBuffers_[i].allocate(baseCount_); // a, n1, n2
        if (parameters_.usesStageB()) {
            stageBuffers_[1].allocate(baseCount_);
            stageBuffers_[5].allocate(baseCount_);
        }
        if (parameters_.usesStageC()) {
            stageBuffers_[2].allocate(baseCount_);
            stageBuffers_[6].allocate(baseCount_);
        }
        stages_ = {stageBuffers_[0].data(), stageBuffers_[1].data(), stageBuffers_[2].data(),
                   stageBuffers_[3].data(), stageBuffers_[4].data(), stageBuffers_[5].data(),
                   stageBuffers_[6].data()};
        forcing_.allocate(forcing.size());
        if (!forcing.empty())
            cudaCheck(cudaMemcpy(forcing_.data(), forcing.data(), forcing.size() * sizeof(double),
                                 cudaMemcpyHostToDevice),
                      "upload deterministic forcing");
        if (parameters_.usesStochasticForcing()) {
            compactNoise_ = !stochasticIndices.empty();
            noiseCount_ = compactNoise_ ? stochasticIndices.size() : baseCount_;
            for (const std::size_t index : stochasticIndices)
                if (index >= baseCount_)
                    throw std::runtime_error("compact stochastic index is out of range");
            noise_.allocate(noiseCount_);
            if (compactNoise_) {
                noiseIndices_.allocate(noiseCount_);
                cudaCheck(cudaMemcpy(noiseIndices_.data(), stochasticIndices.data(),
                                     noiseCount_ * sizeof(std::size_t), cudaMemcpyHostToDevice),
                          "upload stochastic mode indices");
            }
        }
        uploadState(vorticity);
        if (graphEnabled_) {
            // cuFFT may perform one-time setup on first execution, which is not
            // permitted while a stream is being captured.
            evaluateDevice(input_.data(), result_.data());
            cudaCheck(cudaStreamSynchronize(stream_.get()), "warm CUDA graph transform plans");
        }
    }

    void advanceTimeStep(const SpectralField &noise) override {
        if (!noise.empty()) {
            if (!noise_.data() || noise.size() != noiseCount_)
                throw std::runtime_error("invalid device noise field");
            cudaCheck(cudaMemcpyAsync(noise_.data(), noise.data(),
                                      noiseCount_ * sizeof(cufftDoubleComplex),
                                      cudaMemcpyHostToDevice, stream_.get()),
                      "upload stochastic increment");
            // The solver reuses this host buffer immediately after the method returns.
            cudaCheck(cudaStreamSynchronize(stream_.get()), "prepare CUDA stochastic increment");
        }
        if (graphEnabled_) {
            if (!graphExec_)
                captureStepGraph();
            cudaCheck(cudaGraphLaunch(graphExec_, stream_.get()), "launch CUDA timestep graph");
        } else {
            executeStep();
        }
    }

    void executeStep() {
        rightHandSide(input_.data(), stages_.n1);
        launchStage(IntegrationStage::a);
        rightHandSide(stages_.a, stages_.n2);
        if (stages_.n3) {
            launchStage(IntegrationStage::b);
            rightHandSide(stages_.b, stages_.n3);
        }
        if (stages_.n4) {
            launchStage(IntegrationStage::c);
            rightHandSide(stages_.c, stages_.n4);
        }
        launchStage(IntegrationStage::finish);
        if (compactNoise_) {
            addSparseNoise<<<blocks(noiseCount_), threads, 0, stream_.get()>>>(
                input_.data(), noise_.data(), noiseIndices_.data(), noiseCount_);
            cudaCheck(cudaGetLastError(), "add compact stochastic increment");
        }
        constrain(input_.data());
    }

    void downloadState(SpectralField &vorticity) override {
        cudaCheck(cudaStreamSynchronize(stream_.get()), "synchronize integrated vorticity");
        downloadStateDevice(vorticity);
    }

    void evaluateCurrent(SpectralField &output) override {
        evaluateDevice(input_.data(), result_.data());
        cudaCheck(cudaStreamSynchronize(stream_.get()), "synchronize current nonlinear evaluation");
        downloadResult(output);
    }

    void evaluate(const SpectralField &vorticity, SpectralField &output) override {
        uploadState(vorticity);
        evaluateDevice(input_.data(), result_.data());
        cudaCheck(cudaStreamSynchronize(stream_.get()), "synchronize nonlinear evaluation");
        downloadResult(output);
    }

  private:
    void downloadStateDevice(SpectralField &vorticity) {
        vorticity.resize(baseCount_);
        cudaCheck(cudaMemcpy(vorticity.data(), input_.data(),
                             baseCount_ * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost),
                  "download integrated vorticity");
    }

    void downloadResult(SpectralField &output) {
        output.resize(baseCount_);
        cudaCheck(cudaMemcpy(output.data(), result_.data(), baseCount_ * sizeof(cufftDoubleComplex),
                             cudaMemcpyDeviceToHost),
                  "download nonlinear term");
    }

    static constexpr int threads = 256;
    int blocks(std::size_t count) const {
        return static_cast<int>((count + threads - 1) / threads);
    }

    void uploadState(const SpectralField &vorticity) {
        if (vorticity.size() != baseCount_)
            throw std::runtime_error("invalid nonlinear input size");
        cudaCheck(cudaMemcpy(input_.data(), vorticity.data(),
                             baseCount_ * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice),
                  "upload vorticity");
    }

    void constrain(cufftDoubleComplex *vorticity) {
        constrainModes<<<blocks(baseCount_), threads, 0, stream_.get()>>>(vorticity, parameters_.ny,
                                                                          parameters_.nxf());
        cudaCheck(cudaGetLastError(), "constrain device spectral field");
    }

    void launchStage(IntegrationStage stage) {
        integrateStage<<<blocks(baseCount_), threads, 0, stream_.get()>>>(
            stage, parameters_.integrator, parameters_.timeStep, baseCount_, coefficients_,
            input_.data(), stages_, compactNoise_ ? nullptr : noise_.data());
        cudaCheck(cudaGetLastError(), "integrate device Runge-Kutta stage");
    }

    void rightHandSide(const cufftDoubleComplex *vorticity, cufftDoubleComplex *output) {
        evaluateDevice(vorticity, output);
        if (forcing_.data()) {
            addForcing<<<blocks(baseCount_), threads, 0, stream_.get()>>>(output, forcing_.data(),
                                                                          baseCount_);
            cudaCheck(cudaGetLastError(), "add deterministic device forcing");
        }
    }

    void evaluateDevice(const cufftDoubleComplex *vorticity, cufftDoubleComplex *output) {
        populateComponents<<<blocks(paddedCount_), threads, 0, stream_.get()>>>(
            vorticity, fields_.data(), parameters_.ny, parameters_.nxf(), parameters_.my(),
            parameters_.mxf(), parameters_.lx(), parameters_.ly());
        cudaCheck(cudaGetLastError(), "launch spectral derivative kernel");
#ifdef NS2D_CUDA_MIXED
        cufftCheck(cufftExecC2R(inverse_.get(), fields_.data(), realFields_.data()),
                   "execute batched inverse transform");
#else
        cufftCheck(cufftExecZ2D(inverse_.get(), fields_.data(), realFields_.data()),
                   "execute batched inverse transform");
#endif
        multiplyFields<<<blocks(realCount_), threads, 0, stream_.get()>>>(
            realFields_.data(), product_.data(), realCount_);
        cudaCheck(cudaGetLastError(), "launch nonlinear product kernel");
#ifdef NS2D_CUDA_MIXED
        cufftCheck(cufftExecR2C(forward_.get(), product_.data(), paddedOutput_.data()),
                   "execute forward transform");
#else
        cufftCheck(cufftExecD2Z(forward_.get(), product_.data(), paddedOutput_.data()),
                   "execute forward transform");
#endif
        extractModes<<<blocks(baseCount_), threads, 0, stream_.get()>>>(
            paddedOutput_.data(), output, parameters_.ny, parameters_.nxf(), parameters_.my(),
            parameters_.mxf(), 1.0 / static_cast<double>(parameters_.mx() * parameters_.my()));
        cudaCheck(cudaGetLastError(), "launch spectral extraction kernel");
        constrain(output);
    }

    void captureStepGraph() {
        cudaCheck(cudaStreamBeginCapture(stream_.get(), cudaStreamCaptureModeThreadLocal),
                  "begin CUDA timestep graph capture");
        executeStep();
        cudaCheck(cudaStreamEndCapture(stream_.get(), &graph_), "end CUDA timestep graph capture");
        cudaCheck(cudaGraphInstantiate(&graphExec_, graph_, nullptr, nullptr, 0),
                  "instantiate CUDA timestep graph");
    }

    std::array<DeviceBuffer<cufftDoubleComplex>, 10> coefficientBuffers_;
    std::array<DeviceBuffer<cufftDoubleComplex>, 7> stageBuffers_;
    DeviceBuffer<cufftDoubleComplex> noise_;
    DeviceBuffer<std::size_t> noiseIndices_;
    DeviceBuffer<double> forcing_;
    CoefficientPointers<cufftDoubleComplex> coefficients_;
    DeviceStages stages_;

    Parameters parameters_;
    std::size_t baseCount_ = 0, paddedCount_ = 0, realCount_ = 0;
    std::size_t noiseCount_ = 0;
    bool compactNoise_ = false;
    bool graphEnabled_ = false;
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t graphExec_ = nullptr;
    DeviceBuffer<cufftDoubleComplex> input_, result_;
    DeviceBuffer<TransformComplex> fields_, paddedOutput_;
    DeviceBuffer<TransformReal> realFields_, product_;
    CudaStream stream_;
    CufftPlan inverse_, forward_;
};
} // namespace

void backendInitialize(int &, char **&) { cudaCheck(cudaFree(nullptr), "initialize CUDA"); }
void backendFinalize() { saveFftwWisdom(); }
void backendAbort(int) {}
void backendBarrier() { cudaCheck(cudaDeviceSynchronize(), "synchronize CUDA device"); }
bool backendIsRoot() { return true; }
const char *backendName() {
#ifdef NS2D_CUDA_MIXED
    return "CUDA mixed (FP64 state / FP32 FFT)";
#else
    return "CUDA";
#endif
}
std::uint64_t backendSynchronizeSeed(std::uint64_t seed) { return seed; }
std::unique_ptr<NonlinearBackend> makeBackend(const Parameters &parameters) {
    configureFftw(parameters);
    return std::make_unique<CudaBackend>(parameters);
}
