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
__host__ __device__ inline cufftDoubleComplex operator*(double a,
                                                        cufftDoubleComplex b) {
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
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
}
void cufftCheck(cufftResult status, const char *operation) {
  if (status != CUFFT_SUCCESS)
    throw std::runtime_error(std::string(operation) + " failed (cuFFT status " +
                             std::to_string(static_cast<int>(status)) + ")");
}

__global__ void populateComponents(const cufftDoubleComplex *input,
                                   cufftDoubleComplex *fields, std::size_t ny,
                                   std::size_t nxf, std::size_t my,
                                   std::size_t mxf, double lx, double ly) {
  const std::size_t flat =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t paddedCount = my * mxf;
  if (flat >= paddedCount)
    return;
  const std::size_t py = flat / mxf;
  const std::size_t x = flat % mxf;
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
#pragma unroll
    for (int component = 0; component < 4; ++component)
      fields[static_cast<std::size_t>(component) * paddedCount + flat] = {0.0,
                                                                          0.0};
    return;
  }
  constexpr double twoPi = 6.283185307179586476925286766559;
  const double kx = twoPi * static_cast<double>(x) / lx;
  const double ky = twoPi * static_cast<double>(kyIndex) / ly;
  const double k2 = kx * kx + ky * ky;
  const cufftDoubleComplex source = input[y * nxf + x];
  const double factors[4] = {kx, k2 == 0.0 ? 0.0 : -ky / k2, ky,
                             k2 == 0.0 ? 0.0 : -kx / k2};
#pragma unroll
  for (int component = 0; component < 4; ++component) {
    const double factor = factors[component];
    fields[static_cast<std::size_t>(component) * paddedCount + flat] = {
        -source.y * factor, source.x * factor};
  }
}

__global__ void multiplyFields(const double *fields, double *product,
                               std::size_t count) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  double value = fields[i] * fields[count + i] -
                 fields[2 * count + i] * fields[3 * count + i];
  product[i] = value;
}

__global__ void extractModes(const cufftDoubleComplex *padded,
                             cufftDoubleComplex *result, std::size_t ny,
                             std::size_t nxf, std::size_t my, std::size_t mxf,
                             double scale) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= ny * nxf)
    return;
  const std::size_t y = index / nxf;
  const std::size_t x = index % nxf;
  const long ky = y < ny / 2 ? static_cast<long>(y)
                             : static_cast<long>(y) - static_cast<long>(ny);
  const std::size_t py =
      ky >= 0 ? static_cast<std::size_t>(ky)
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
      cudaCheck(cudaMalloc(&data_, count * sizeof(T)),
                "allocate integration buffer");
  }
  T *data() const { return data_; }

private:
  T *data_ = nullptr;
};

struct DeviceStages {
  cufftDoubleComplex *a{}, *b{}, *c{}, *n1{}, *n2{}, *n3{}, *n4{};
};

__global__ void constrainModes(cufftDoubleComplex *w, std::size_t ny,
                               std::size_t nxf) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= ny * nxf)
    return;
  const std::size_t x = i % nxf, y = i / nxf;
  if (i == 0 || x == nxf - 1 || y == ny / 2)
    w[i] = {0.0, 0.0};
  else if (x == 0 && y > ny / 2) {
    const cufftDoubleComplex partner = w[(ny - y) * nxf];
    w[i] = {partner.x, -partner.y};
  }
}

__global__ void addForcing(cufftDoubleComplex *n, const double *forcing,
                           std::size_t count) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    n[i].x += forcing[i];
}

__global__ void
integrateStage(int stage, Integrator method, double h, std::size_t count,
               CoefficientPointers<cufftDoubleComplex> coefficients,
               cufftDoubleComplex *w, DeviceStages s,
               const cufftDoubleComplex *noise) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  if (stage == 0)
    s.a[i] = integrationStageA(method, h, i, coefficients, w[i], s.n1[i]);
  else if (stage == 1)
    s.b[i] = integrationStageB(method, i, coefficients, w[i], s.n1[i], s.n2[i]);
  else if (stage == 2)
    s.c[i] = integrationStageC(i, coefficients, w[i], s.n1[i], s.n3[i]);
  else {
    const cufftDoubleComplex zero{0.0, 0.0};
    w[i] = integrationFinish(method, h, i, coefficients, w[i], s.a[i], s.n1[i],
                             s.n2[i], s.n3 ? s.n3[i] : zero,
                             s.n4 ? s.n4[i] : zero);
    if (noise)
      w[i] = w[i] + noise[i];
  }
}

class CudaBackend final : public NonlinearBackend {
public:
  explicit CudaBackend(const Parameters &p) : p_(p) {
    static_assert(sizeof(Complex) == sizeof(cufftDoubleComplex));
    if (p.mx() > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        p.my() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
      throw std::runtime_error("CUDA grid dimensions exceed cuFFT limits");
    baseCount_ = p.ny * p.nxf();
    paddedCount_ = p.my() * p.mxf();
    realCount_ = p.my() * p.mx();
    if (paddedCount_ >
            static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        realCount_ > static_cast<std::size_t>(std::numeric_limits<int>::max()))
      throw std::runtime_error(
          "CUDA transform allocation exceeds cuFFT stride limits");
    try {
      cudaCheck(cudaMalloc(&input_, baseCount_ * sizeof(cufftDoubleComplex)),
                "cudaMalloc input");
      cudaCheck(
          cudaMalloc(&fields_, 4 * paddedCount_ * sizeof(cufftDoubleComplex)),
          "cudaMalloc fields");
      cudaCheck(cudaMalloc(&realFields_, 4 * realCount_ * sizeof(double)),
                "cudaMalloc real fields");
      cudaCheck(cudaMalloc(&product_, realCount_ * sizeof(double)),
                "cudaMalloc product");
      cudaCheck(
          cudaMalloc(&paddedOutput_, paddedCount_ * sizeof(cufftDoubleComplex)),
          "cudaMalloc output");
      cudaCheck(cudaMalloc(&result_, baseCount_ * sizeof(cufftDoubleComplex)),
                "cudaMalloc result");

      int dimensions[] = {static_cast<int>(p.my()), static_cast<int>(p.mx())};
      cufftCheck(cufftPlanMany(&inverse_, 2, dimensions, nullptr, 1,
                               static_cast<int>(paddedCount_), nullptr, 1,
                               static_cast<int>(realCount_), CUFFT_Z2D, 4),
                 "create batched inverse plan");
      cufftCheck(
          cufftPlan2d(&forward_, dimensions[0], dimensions[1], CUFFT_D2Z),
          "create forward plan");
    } catch (...) {
      release();
      throw;
    }
  }

  ~CudaBackend() override { release(); }

  void release() noexcept {
    if (inverse_)
      cufftDestroy(inverse_);
    if (forward_)
      cufftDestroy(forward_);
    cudaFree(input_);
    cudaFree(fields_);
    cudaFree(realFields_);
    cudaFree(product_);
    cudaFree(paddedOutput_);
    cudaFree(result_);
    inverse_ = 0;
    forward_ = 0;
    input_ = fields_ = paddedOutput_ = result_ = nullptr;
    realFields_ = product_ = nullptr;
  }

  bool deviceTimeStepping() const override { return true; }

  void initializeTimeStepping(const IntegrationCoefficients &c,
                              const std::vector<double> &forcing,
                              const SpectralField &w) override {
    const auto arrays = c.fields();
    for (std::size_t i = 0; i < arrays.size(); ++i) {
      coefficientBuffers_[i].allocate(arrays[i]->size());
      if (!arrays[i]->empty())
        cudaCheck(cudaMemcpy(coefficientBuffers_[i].data(), arrays[i]->data(),
                             arrays[i]->size() * sizeof(cufftDoubleComplex),
                             cudaMemcpyHostToDevice),
                  "upload integration coefficients");
    }
    coefficients_ = {
        coefficientBuffers_[0].data(), coefficientBuffers_[1].data(),
        coefficientBuffers_[2].data(), coefficientBuffers_[3].data(),
        coefficientBuffers_[4].data(), coefficientBuffers_[5].data(),
        coefficientBuffers_[6].data(), coefficientBuffers_[7].data(),
        coefficientBuffers_[8].data(), coefficientBuffers_[9].data()};
    for (std::size_t i : {0UL, 3UL, 4UL})
      stageBuffers_[i].allocate(baseCount_); // a, n1, n2
    if (p_.integrator == Integrator::etd3 ||
        p_.integrator == Integrator::etd4) {
      stageBuffers_[1].allocate(baseCount_);
      stageBuffers_[5].allocate(baseCount_);
    }
    if (p_.integrator == Integrator::etd4) {
      stageBuffers_[2].allocate(baseCount_);
      stageBuffers_[6].allocate(baseCount_);
    }
    stages_ = {stageBuffers_[0].data(), stageBuffers_[1].data(),
               stageBuffers_[2].data(), stageBuffers_[3].data(),
               stageBuffers_[4].data(), stageBuffers_[5].data(),
               stageBuffers_[6].data()};
    forcing_.allocate(forcing.size());
    if (!forcing.empty())
      cudaCheck(cudaMemcpy(forcing_.data(), forcing.data(),
                           forcing.size() * sizeof(double),
                           cudaMemcpyHostToDevice),
                "upload deterministic forcing");
    if (p_.forcingEnabled && p_.forcingProfile != ForcingProfile::singleMode)
      noise_.allocate(baseCount_);
    uploadState(w);
  }

  void advance(const SpectralField &noise) override {
    if (!noise.empty()) {
      if (!noise_.data() || noise.size() != baseCount_)
        throw std::runtime_error("invalid device noise field");
      cudaCheck(cudaMemcpy(noise_.data(), noise.data(),
                           baseCount_ * sizeof(cufftDoubleComplex),
                           cudaMemcpyHostToDevice),
                "upload stochastic increment");
    }
    rightHandSide(input_, stages_.n1);
    launchStage(0);
    rightHandSide(stages_.a, stages_.n2);
    if (stages_.n3) {
      launchStage(1);
      rightHandSide(stages_.b, stages_.n3);
    }
    if (stages_.n4) {
      launchStage(2);
      rightHandSide(stages_.c, stages_.n4);
    }
    launchStage(3);
    constrain(input_);
  }

  void downloadState(SpectralField &w) override {
    w.resize(baseCount_);
    cudaCheck(cudaMemcpy(w.data(), input_,
                         baseCount_ * sizeof(cufftDoubleComplex),
                         cudaMemcpyDeviceToHost),
              "download integrated vorticity");
  }

  void evaluate(const SpectralField &w, SpectralField &output) override {
    uploadState(w);
    evaluateDevice(input_, result_);
    output.resize(baseCount_);
    cudaCheck(cudaMemcpy(output.data(), result_,
                         baseCount_ * sizeof(cufftDoubleComplex),
                         cudaMemcpyDeviceToHost),
              "download nonlinear term");
  }

private:
  static constexpr int threads = 256;
  int blocks(std::size_t count) const {
    return static_cast<int>((count + threads - 1) / threads);
  }

  void uploadState(const SpectralField &w) {
    if (w.size() != baseCount_)
      throw std::runtime_error("invalid nonlinear input size");
    cudaCheck(cudaMemcpy(input_, w.data(),
                         baseCount_ * sizeof(cufftDoubleComplex),
                         cudaMemcpyHostToDevice),
              "upload vorticity");
  }

  void constrain(cufftDoubleComplex *w) {
    constrainModes<<<blocks(baseCount_), threads>>>(w, p_.ny, p_.nxf());
    cudaCheck(cudaGetLastError(), "constrain device spectral field");
  }

  void launchStage(int stage) {
    integrateStage<<<blocks(baseCount_), threads>>>(
        stage, p_.integrator, p_.timeStep, baseCount_, coefficients_, input_,
        stages_, noise_.data());
    cudaCheck(cudaGetLastError(), "integrate device Runge-Kutta stage");
  }

  void rightHandSide(const cufftDoubleComplex *w, cufftDoubleComplex *output) {
    evaluateDevice(w, output);
    if (forcing_.data()) {
      addForcing<<<blocks(baseCount_), threads>>>(output, forcing_.data(),
                                                  baseCount_);
      cudaCheck(cudaGetLastError(), "add deterministic device forcing");
    }
  }

  void evaluateDevice(const cufftDoubleComplex *w, cufftDoubleComplex *output) {
    populateComponents<<<blocks(paddedCount_), threads>>>(
        w, fields_, p_.ny, p_.nxf(), p_.my(), p_.mxf(), p_.lx(), p_.ly());
    cudaCheck(cudaGetLastError(), "launch spectral derivative kernel");
    cufftCheck(cufftExecZ2D(inverse_, fields_, realFields_),
               "execute batched inverse transform");
    multiplyFields<<<blocks(realCount_), threads>>>(realFields_, product_,
                                                    realCount_);
    cudaCheck(cudaGetLastError(), "launch nonlinear product kernel");
    cufftCheck(cufftExecD2Z(forward_, product_, paddedOutput_),
               "execute forward transform");
    extractModes<<<blocks(baseCount_), threads>>>(
        paddedOutput_, output, p_.ny, p_.nxf(), p_.my(), p_.mxf(),
        1.0 / static_cast<double>(p_.mx() * p_.my()));
    cudaCheck(cudaGetLastError(), "launch spectral extraction kernel");
    constrain(output);
  }

  std::array<DeviceBuffer<cufftDoubleComplex>, 10> coefficientBuffers_;
  std::array<DeviceBuffer<cufftDoubleComplex>, 7> stageBuffers_;
  DeviceBuffer<cufftDoubleComplex> noise_;
  DeviceBuffer<double> forcing_;
  CoefficientPointers<cufftDoubleComplex> coefficients_;
  DeviceStages stages_;

  Parameters p_;
  std::size_t baseCount_ = 0, paddedCount_ = 0, realCount_ = 0;
  cufftDoubleComplex *input_ = nullptr, *fields_ = nullptr,
                     *paddedOutput_ = nullptr, *result_ = nullptr;
  double *realFields_ = nullptr, *product_ = nullptr;
  cufftHandle inverse_ = 0, forward_ = 0;
};
} // namespace

void backendInitialize(int &, char **&) {
  cudaCheck(cudaFree(nullptr), "initialize CUDA");
}
void backendFinalize() {}
void backendAbort(int) {}
void backendBarrier() {}
bool backendIsRoot() { return true; }
const char *backendName() { return "CUDA"; }
std::uint64_t backendSynchronizeSeed(std::uint64_t seed) { return seed; }
std::unique_ptr<NonlinearBackend> makeBackend(const Parameters &p) {
  return std::make_unique<CudaBackend>(p);
}
