#pragma once

#include "parameters.hpp"

#include <cstdint>
#include <filesystem>
#include <vector>

struct Hdf5Field {
    std::size_t nx = 0, ny = 0;
    double lengthX = 0.0, lengthY = 0.0, time = 0.0;
    std::uint64_t frame = 0;
    std::vector<double> vorticity;
};

[[nodiscard]] bool hdf5Available();
void writeHdf5Field(const std::filesystem::path &path, const Parameters &parameters, double time,
                    std::uint64_t frame, const std::vector<double> &physical);
Hdf5Field readHdf5Field(const std::filesystem::path &path);
