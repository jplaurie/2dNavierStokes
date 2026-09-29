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
#include "spectral.hpp"

#include <array>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
void cudaCheck(cudaError_t status, const char *operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
void cufftCheck(cufftResult status, const char *operation) {
    if (status != CUFFT_SUCCESS)
        throw std::runtime_error(std::string(operation) + " failed (cuFFT status " +
                                 std::to_string(static_cast<int>(status)) + ")");
}

__global__ void populateComponents(const cufftDoubleComplex *input, cufftDoubleComplex *fields,
                                   std::size_t ny, std::size_t nxf, std::size_t my, std::size_t mxf,
                                   double lx, double ly) {
    const std::size_t flat = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t paddedCount = my * mxf;
    if (flat >= paddedCount)
        return;
    const std::size_t py = flat / mxf;
    const std::size_t x = flat % mxf;
    cufftDoubleComplex *vorticityX = fields;
    cufftDoubleComplex *streamfunctionY = fields + paddedCount;
    cufftDoubleComplex *vorticityY = fields + 2 * paddedCount;
    cufftDoubleComplex *streamfunctionX = fields + 3 * paddedCount;
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
        vorticityX[flat] = {0.0, 0.0};
        streamfunctionY[flat] = {0.0, 0.0};
        vorticityY[flat] = {0.0, 0.0};
        streamfunctionX[flat] = {0.0, 0.0};
        return;
    }
    constexpr double twoPi = 6.283185307179586476925286766559;
    const double kx = twoPi * static_cast<double>(x) / lx;
    const double ky = twoPi * static_cast<double>(kyIndex) / ly;
    const double k2 = kx * kx + ky * ky;
    const cufftDoubleComplex source = input[y * nxf + x];
    const auto multiplyByImaginary = [source](double factor) {
        return cufftDoubleComplex{-source.y * factor, source.x * factor};
    };
    vorticityX[flat] = multiplyByImaginary(kx);
    streamfunctionY[flat] = multiplyByImaginary(k2 == 0.0 ? 0.0 : -ky / k2);
    vorticityY[flat] = multiplyByImaginary(ky);
    streamfunctionX[flat] = multiplyByImaginary(k2 == 0.0 ? 0.0 : -kx / k2);
}

__global__ void multiplyFields(const double *fields, double *product, std::size_t count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count)
        return;
    const double vorticityX = fields[index];
    const double streamfunctionY = fields[count + index];
    const double vorticityY = fields[2 * count + index];
    const double streamfunctionX = fields[3 * count + index];
    product[index] = vorticityX * streamfunctionY - vorticityY * streamfunctionX;
}

__global__ void extractModes(const cufftDoubleComplex *padded, cufftDoubleComplex *result,
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
    const cufftDoubleComplex value = padded[py * mxf + x];
    result[index] = {value.x * scale, value.y * scale};
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
    explicit CudaBackend(const Parameters &parameters) : parameters_(parameters) {
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
                                 static_cast<int>(realCount_), CUFFT_Z2D, 4),
                   "create batched inverse plan");
        cufftCheck(cufftPlan2d(forward_.address(), dimensions[0], dimensions[1], CUFFT_D2Z),
                   "create forward plan");
    }

    bool supportsDeviceTimeStepping() const override { return true; }

    void initializeDeviceState(const IntegrationCoefficients &c, const std::vector<double> &forcing,
                               const std::vector<std::size_t> &stochasticIndices,
                               const SpectralField &vorticity) override {
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
    }

    void advanceDeviceState(const SpectralField &noise) override {
        if (!noise.empty()) {
            if (!noise_.data() || noise.size() != noiseCount_)
                throw std::runtime_error("invalid device noise field");
            cudaCheck(cudaMemcpy(noise_.data(), noise.data(),
                                 noiseCount_ * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice),
                      "upload stochastic increment");
        }
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
            addSparseNoise<<<blocks(noiseCount_), threads>>>(input_.data(), noise_.data(),
                                                             noiseIndices_.data(), noiseCount_);
            cudaCheck(cudaGetLastError(), "add compact stochastic increment");
        }
        constrain(input_.data());
    }

    void downloadStateAndEvaluate(SpectralField &vorticity, SpectralField &output) override {
        evaluateDevice(input_.data(), result_.data());
        downloadState(vorticity);
        downloadResult(output);
    }

    void evaluate(const SpectralField &vorticity, SpectralField &output) override {
        uploadState(vorticity);
        evaluateDevice(input_.data(), result_.data());
        downloadResult(output);
    }

  private:
    void downloadState(SpectralField &vorticity) {
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
        constrainModes<<<blocks(baseCount_), threads>>>(vorticity, parameters_.ny,
                                                        parameters_.nxf());
        cudaCheck(cudaGetLastError(), "constrain device spectral field");
    }

    void launchStage(IntegrationStage stage) {
        integrateStage<<<blocks(baseCount_), threads>>>(
            stage, parameters_.integrator, parameters_.timeStep, baseCount_, coefficients_,
            input_.data(), stages_, compactNoise_ ? nullptr : noise_.data());
        cudaCheck(cudaGetLastError(), "integrate device Runge-Kutta stage");
    }

    void rightHandSide(const cufftDoubleComplex *vorticity, cufftDoubleComplex *output) {
        evaluateDevice(vorticity, output);
        if (forcing_.data()) {
            addForcing<<<blocks(baseCount_), threads>>>(output, forcing_.data(), baseCount_);
            cudaCheck(cudaGetLastError(), "add deterministic device forcing");
        }
    }

    void evaluateDevice(const cufftDoubleComplex *vorticity, cufftDoubleComplex *output) {
        populateComponents<<<blocks(paddedCount_), threads>>>(
            vorticity, fields_.data(), parameters_.ny, parameters_.nxf(), parameters_.my(),
            parameters_.mxf(), parameters_.lx(), parameters_.ly());
        cudaCheck(cudaGetLastError(), "launch spectral derivative kernel");
        cufftCheck(cufftExecZ2D(inverse_.get(), fields_.data(), realFields_.data()),
                   "execute batched inverse transform");
        multiplyFields<<<blocks(realCount_), threads>>>(realFields_.data(), product_.data(),
                                                        realCount_);
        cudaCheck(cudaGetLastError(), "launch nonlinear product kernel");
        cufftCheck(cufftExecD2Z(forward_.get(), product_.data(), paddedOutput_.data()),
                   "execute forward transform");
        extractModes<<<blocks(baseCount_), threads>>>(
            paddedOutput_.data(), output, parameters_.ny, parameters_.nxf(), parameters_.my(),
            parameters_.mxf(), 1.0 / static_cast<double>(parameters_.mx() * parameters_.my()));
        cudaCheck(cudaGetLastError(), "launch spectral extraction kernel");
        constrain(output);
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
    DeviceBuffer<cufftDoubleComplex> input_, fields_, paddedOutput_, result_;
    DeviceBuffer<double> realFields_, product_;
    CufftPlan inverse_, forward_;
};
} // namespace

void backendInitialize(int &, char **&) { cudaCheck(cudaFree(nullptr), "initialize CUDA"); }
void backendFinalize() {}
void backendAbort(int) {}
void backendBarrier() {}
bool backendIsRoot() { return true; }
const char *backendName() { return "CUDA"; }
std::uint64_t backendSynchronizeSeed(std::uint64_t seed) { return seed; }
std::unique_ptr<NonlinearBackend> makeBackend(const Parameters &parameters) {
    return std::make_unique<CudaBackend>(parameters);
}
