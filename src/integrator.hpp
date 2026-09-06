#pragma once

#include "parameters.hpp"

#include <array>
#include <complex>
#include <vector>

#ifdef __CUDACC__
#define NS2D_HD __host__ __device__
#else
#define NS2D_HD
#endif

// The same per-mode stage algebra is compiled for the CPU and GPU.
template <class T> struct CoefficientPointers {
  const T *e1{}, *e2{}, *q1{}, *q2{}, *q3{}, *q4{}, *q5{};
  const T *f1{}, *f2{}, *f3{};
};

struct IntegrationCoefficients {
  using Field = std::vector<std::complex<double>>;
  Field e1, e2, q1, q2, q3, q4, q5, f1, f2, f3;

  auto fields() const {
    return std::array{&e1, &e2, &q1, &q2, &q3, &q4, &q5, &f1, &f2, &f3};
  }
  CoefficientPointers<std::complex<double>> pointers() const {
    return {e1.data(), e2.data(), q1.data(), q2.data(), q3.data(),
            q4.data(), q5.data(), f1.data(), f2.data(), f3.data()};
  }
};

template <class T>
NS2D_HD T integrationStageA(Integrator method, double h, std::size_t i,
                            CoefficientPointers<T> c, T w, T n1) {
  if (method == Integrator::integratingFactorRk2)
    return c.e1[i] * (w + h * n1);
  if (method == Integrator::etd2)
    return c.e1[i] * w + c.q1[i] * n1;
  return c.e2[i] * w + c.q1[i] * n1;
}

template <class T>
NS2D_HD T integrationStageB(Integrator method, std::size_t i,
                            CoefficientPointers<T> c, T w, T n1, T n2) {
  if (method == Integrator::etd3)
    return c.e1[i] * w + c.q2[i] * (2.0 * n2 - n1);
  return c.e2[i] * w + c.q2[i] * n1 + c.q3[i] * n2;
}

template <class T>
NS2D_HD T integrationStageC(std::size_t i, CoefficientPointers<T> c, T w, T n1,
                            T n3) {
  return c.e1[i] * w + c.q4[i] * n1 + c.q5[i] * n3;
}

template <class T>
NS2D_HD T integrationFinish(Integrator method, double h, std::size_t i,
                            CoefficientPointers<T> c, T w, T a, T n1, T n2,
                            T n3, T n4) {
  if (method == Integrator::integratingFactorRk2)
    return c.e1[i] * w + (0.5 * h) * (c.e1[i] * n1 + n2);
  if (method == Integrator::etd2)
    return a + c.f1[i] * (n2 - n1);
  if (method == Integrator::etd3)
    return c.e1[i] * w + c.f1[i] * n1 + 4.0 * c.f2[i] * n2 + c.f3[i] * n3;
  return c.e1[i] * w + c.f1[i] * n1 + 2.0 * c.f2[i] * (n2 + n3) + c.f3[i] * n4;
}

#undef NS2D_HD
