#pragma once

#include <cstddef>

template <class Operation> void forEachIndex(std::size_t count, Operation operation) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (count >= 16384)
#endif
    for (std::ptrdiff_t raw = 0; raw < static_cast<std::ptrdiff_t>(count); ++raw)
        operation(static_cast<std::size_t>(raw));
}
