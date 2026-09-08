#pragma once

#include "backend.hpp"

#include <fftw3.h>

#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

// Fixed FFTW alignment and deterministic planning make the arithmetic sequence
// independent of allocator addresses and planning-time benchmark noise.
template <class T> struct FftwAllocator {
  using value_type = T;
  FftwAllocator() = default;
  template <class U> FftwAllocator(const FftwAllocator<U> &) {}
  T *allocate(std::size_t count) {
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T))
      throw std::bad_array_new_length();
    auto *memory = static_cast<T *>(fftw_malloc(count * sizeof(T)));
    if (!memory)
      throw std::bad_alloc();
    return memory;
  }
  void deallocate(T *memory, std::size_t) { fftw_free(memory); }
  template <class U> bool operator==(const FftwAllocator<U> &) const {
    return true;
  }
};
using FftwSpectralField = std::vector<Complex, FftwAllocator<Complex>>;
using FftwRealField = std::vector<double, FftwAllocator<double>>;

class FftwPlan {
public:
  FftwPlan() = default;
  ~FftwPlan() { reset(); }
  FftwPlan(const FftwPlan &) = delete;
  FftwPlan &operator=(const FftwPlan &) = delete;
  void reset(fftw_plan plan = nullptr) noexcept {
    if (plan_)
      fftw_destroy_plan(plan_);
    plan_ = plan;
  }
  void execute() const { fftw_execute(plan_); }
  [[nodiscard]] explicit operator bool() const { return plan_ != nullptr; }

private:
  fftw_plan plan_ = nullptr;
};

template <class Allocator>
inline fftw_complex *fftwData(std::vector<Complex, Allocator> &field) {
  static_assert(sizeof(Complex) == sizeof(fftw_complex));
  return reinterpret_cast<fftw_complex *>(field.data());
}

template <class Allocator>
inline const fftw_complex *
fftwData(const std::vector<Complex, Allocator> &field) {
  static_assert(sizeof(Complex) == sizeof(fftw_complex));
  return reinterpret_cast<const fftw_complex *>(field.data());
}

class BaseTransform {
public:
  explicit BaseTransform(const Parameters &parameters);
  BaseTransform(const BaseTransform &) = delete;
  BaseTransform &operator=(const BaseTransform &) = delete;

  void forward(const std::vector<double> &real, SpectralField &spectral);
  void inverse(const SpectralField &spectral, std::vector<double> &real);

private:
  Parameters p_;
  FftwRealField planningReal_;
  FftwSpectralField planningComplex_;
  FftwPlan forward_, inverse_;
};
