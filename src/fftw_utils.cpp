#include "fftw_utils.hpp"

#include <algorithm>
#include <filesystem>

namespace {
unsigned planningFlags = FFTW_ESTIMATE;
std::filesystem::path wisdomPath;
} // namespace

void configureFftw(const Parameters &parameters, bool importWisdom) {
    switch (parameters.fftwPlanning) {
    case FftwPlanning::estimate:
        planningFlags = FFTW_ESTIMATE;
        break;
    case FftwPlanning::measure:
        planningFlags = FFTW_MEASURE;
        break;
    case FftwPlanning::patient:
        planningFlags = FFTW_PATIENT;
        break;
    }
    wisdomPath = parameters.fftwWisdomFile;
    if (importWisdom && !wisdomPath.empty() && std::filesystem::exists(wisdomPath) &&
        !fftw_import_wisdom_from_filename(wisdomPath.c_str()))
        throw std::runtime_error("cannot import FFTW wisdom: " + wisdomPath.string());
}

void saveFftwWisdom() {
    if (!wisdomPath.empty() && !fftw_export_wisdom_to_filename(wisdomPath.c_str()))
        throw std::runtime_error("cannot export FFTW wisdom: " + wisdomPath.string());
}

unsigned fftwPlanningFlags() { return planningFlags; }

BaseTransform::BaseTransform(const Parameters &parameters)
    : parameters_(parameters), planningReal_(parameters.nx * parameters.ny),
      planningComplex_(parameters.spectralSize()) {
    forward_.reset(fftw_plan_dft_r2c_2d(static_cast<int>(parameters.ny),
                                        static_cast<int>(parameters.nx), planningReal_.data(),
                                        fftwData(planningComplex_), fftwPlanningFlags()));
    inverse_.reset(fftw_plan_dft_c2r_2d(static_cast<int>(parameters.ny),
                                        static_cast<int>(parameters.nx), fftwData(planningComplex_),
                                        planningReal_.data(), fftwPlanningFlags()));
    if (!forward_ || !inverse_)
        throw std::runtime_error("FFTW could not create base-grid plans");
}

void BaseTransform::forward(const std::vector<double> &real, SpectralField &spectral) {
    if (real.size() != parameters_.nx * parameters_.ny)
        throw std::runtime_error("invalid real field size");
    planningReal_.assign(real.begin(), real.end());
    forward_.execute();
    spectral.assign(planningComplex_.begin(), planningComplex_.end());
    const double scale = 1.0 / static_cast<double>(parameters_.nx * parameters_.ny);
    for (Complex &value : spectral)
        value *= scale;
}

void BaseTransform::inverse(const SpectralField &spectral, std::vector<double> &real) {
    if (spectral.size() != parameters_.spectralSize())
        throw std::runtime_error("invalid spectral field size");
    planningComplex_.assign(spectral.begin(),
                            spectral.end()); // FFTW may overwrite c2r input.
    inverse_.execute();
    real.assign(planningReal_.begin(), planningReal_.end());
}
