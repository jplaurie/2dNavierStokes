#include "parameters.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <type_traits>

namespace {
constexpr double pi = 3.141592653589793238462643383279502884;

#ifndef NS2D_VERSION
#define NS2D_VERSION "unknown"
#endif
#ifndef NS2D_GIT_COMMIT
#define NS2D_GIT_COMMIT "unknown"
#endif
#ifndef NS2D_GIT_DIRTY
#define NS2D_GIT_DIRTY "unknown"
#endif

bool parseBool(const std::string &text, const std::string &key) {
    if (text == "true" || text == "1")
        return true;
    if (text == "false" || text == "0")
        return false;
    throw std::runtime_error(key + " must be true or false, got: " + text);
}

template <class T> T parseNumber(const std::string &text, const std::string &key) {
    if constexpr (std::is_unsigned_v<T>) {
        if (!text.empty() && text.front() == '-')
            throw std::runtime_error(key + " cannot be negative: " + text);
    }
    std::istringstream input(text);
    T value{};
    input >> value;
    if (!input || !(input >> std::ws).eof())
        throw std::runtime_error("invalid value for " + key + ": " + text);
    return value;
}

std::string trim(const std::string &value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
        return {};
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

Integrator parseIntegrator(const std::string &text) {
    if (text == "etd2")
        return Integrator::etd2;
    if (text == "etd3")
        return Integrator::etd3;
    if (text == "etd4")
        return Integrator::etd4;
    if (text == "rk2")
        return Integrator::integratingFactorRk2;
    throw std::runtime_error("integrator must be etd2, etd3, etd4, or rk2");
}

ForcingProfile parseForcingProfile(const std::string &text) {
    if (text == "annulus")
        return ForcingProfile::annulus;
    if (text == "exponential")
        return ForcingProfile::exponential;
    if (text == "singleMode")
        return ForcingProfile::singleMode;
    throw std::runtime_error("forcingProfile must be annulus, exponential, or singleMode");
}

FftwPlanning parseFftwPlanning(const std::string &text) {
    if (text == "estimate")
        return FftwPlanning::estimate;
    if (text == "measure")
        return FftwPlanning::measure;
    if (text == "patient")
        return FftwPlanning::patient;
    throw std::runtime_error("fftwPlanning must be estimate, measure, or patient");
}

FieldOutputFormat parseFieldOutputFormat(const std::string &text) {
    if (text == "text")
        return FieldOutputFormat::text;
    if (text == "hdf5")
        return FieldOutputFormat::hdf5;
    if (text == "both")
        return FieldOutputFormat::both;
    throw std::runtime_error("fieldOutputFormat must be text, hdf5, or both");
}

struct ParameterSetting {
    std::string key;
    std::string value;
    std::size_t lineNumber;
};

std::optional<ParameterSetting> parseParameterLine(std::string line, std::size_t lineNumber) {
    if (const auto comment = line.find('#'); comment != std::string::npos)
        line.erase(comment);
    line = trim(line);
    if (line.empty())
        return std::nullopt;

    std::replace(line.begin(), line.end(), '=', ' ');
    std::istringstream fields(line);
    ParameterSetting setting{{}, {}, lineNumber};
    std::string extra;
    fields >> setting.key >> setting.value;
    if (setting.key.empty() || setting.value.empty() || (fields >> extra))
        throw std::runtime_error("invalid parameter line " + std::to_string(lineNumber));
    return setting;
}

void applyParameter(const ParameterSetting &setting, Parameters &parameters) {
    const std::string &key = setting.key;
    const std::string &value = setting.value;
    const auto readNumber = [&]<class T>(T &destination) {
        destination = parseNumber<T>(value, key);
    };

    if (key == "nx")
        readNumber(parameters.nx);
    else if (key == "ny")
        readNumber(parameters.ny);
    else if (key == "aspectRatio")
        readNumber(parameters.aspectRatio);
    else if (key == "timeStep")
        readNumber(parameters.timeStep);
    else if (key == "numberOfSteps")
        readNumber(parameters.numberOfSteps);
    else if (key == "outputIntervalSteps")
        readNumber(parameters.outputIntervalSteps);
    else if (key == "integrator")
        parameters.integrator = parseIntegrator(value);
    else if (key == "betaPlane")
        parameters.betaPlane = parseBool(value, key);
    else if (key == "beta")
        readNumber(parameters.beta);
    else if (key == "viscosity")
        readNumber(parameters.viscosity);
    else if (key == "viscosityOrder")
        readNumber(parameters.viscosityOrder);
    else if (key == "linearDrag")
        readNumber(parameters.linearDrag);
    else if (key == "dragOrder")
        readNumber(parameters.dragOrder);
    else if (key == "forcingEnabled")
        parameters.forcingEnabled = parseBool(value, key);
    else if (key == "forcingProfile")
        parameters.forcingProfile = parseForcingProfile(value);
    else if (key == "forcingWavenumber")
        readNumber(parameters.forcingWavenumber);
    else if (key == "forcingWidth")
        readNumber(parameters.forcingWidth);
    else if (key == "forcingAmplitude")
        readNumber(parameters.forcingAmplitude);
    else if (key == "forcingShapeOrder")
        readNumber(parameters.forcingShapeOrder);
    else if (key == "targetEnergyInjectionRate")
        readNumber(parameters.targetEnergyInjectionRate);
    else if (key == "randomSeed")
        readNumber(parameters.randomSeed);
    else if (key == "writeModeDiagnostics")
        parameters.writeModeDiagnostics = parseBool(value, key);
    else if (key == "fieldOutputFormat")
        parameters.fieldOutputFormat = parseFieldOutputFormat(value);
    else if (key == "hdf5CompressionLevel")
        readNumber(parameters.hdf5CompressionLevel);
    else if (key == "fftwPlanning")
        parameters.fftwPlanning = parseFftwPlanning(value);
    else if (key == "fftwWisdomFile")
        parameters.fftwWisdomFile = value;
    else if (key == "cudaGraphEnabled")
        parameters.cudaGraphEnabled = parseBool(value, key);
    else if (key == "threadCount")
        readNumber(parameters.threadCount);
    else if (key == "overwriteOutput")
        parameters.overwriteOutput = parseBool(value, key);
    else if (key == "initialConditionFile")
        parameters.initialConditionFile = value;
    else if (key == "dataDirectory")
        parameters.dataDirectory = value;
    else if (key == "outputDirectory")
        parameters.outputDirectory = value;
    else
        throw std::runtime_error("unknown parameter key on line " +
                                 std::to_string(setting.lineNumber) + ": " + key);
}
} // namespace

double Parameters::lx() const { return 2.0 * pi * aspectRatio; }
double Parameters::ly() const { return 2.0 * pi; }

std::size_t Parameters::spectrumBins() const {
    const double binWidth = std::min(2.0 * pi / lx(), 2.0 * pi / ly());
    const double maximumKx = pi * static_cast<double>(nx) / lx();
    const double maximumKy = pi * static_cast<double>(ny) / ly();
    const double count = std::floor(std::hypot(maximumKx, maximumKy) / binWidth) + 1.0;
    if (!std::isfinite(count) || count < 1.0 ||
        count >= static_cast<double>(std::numeric_limits<std::size_t>::max()))
        throw std::runtime_error("aspectRatio produces an invalid spectrum grid");
    return static_cast<std::size_t>(count);
}

const char *integratorName(Integrator integrator) {
    switch (integrator) {
    case Integrator::etd2:
        return "etd2";
    case Integrator::etd3:
        return "etd3";
    case Integrator::etd4:
        return "etd4";
    case Integrator::integratingFactorRk2:
        return "rk2";
    }
    throw std::logic_error("unknown integrator");
}

const char *forcingProfileName(ForcingProfile profile) {
    switch (profile) {
    case ForcingProfile::annulus:
        return "annulus";
    case ForcingProfile::exponential:
        return "exponential";
    case ForcingProfile::singleMode:
        return "singleMode";
    }
    throw std::logic_error("unknown forcing profile");
}

const char *fftwPlanningName(FftwPlanning planning) {
    switch (planning) {
    case FftwPlanning::estimate:
        return "estimate";
    case FftwPlanning::measure:
        return "measure";
    case FftwPlanning::patient:
        return "patient";
    }
    throw std::logic_error("unknown FFTW planning mode");
}

const char *fieldOutputFormatName(FieldOutputFormat format) {
    switch (format) {
    case FieldOutputFormat::text:
        return "text";
    case FieldOutputFormat::hdf5:
        return "hdf5";
    case FieldOutputFormat::both:
        return "both";
    }
    throw std::logic_error("unknown field output format");
}

Parameters readParameters(const std::filesystem::path &path) {
    std::ifstream input(path);
    if (!input)
        throw std::runtime_error("cannot open parameter file: " + path.string());
    Parameters parameters;
    std::string line;
    std::size_t lineNumber = 0;
    while (std::getline(input, line)) {
        ++lineNumber;
        if (const auto setting = parseParameterLine(line, lineNumber))
            applyParameter(*setting, parameters);
    }
    validateParameters(parameters);
    return parameters;
}

void validateParameters(const Parameters &parameters) {
    if (parameters.nx < 4 || parameters.ny < 4 || parameters.nx % 4 != 0 || parameters.ny % 4 != 0)
        throw std::runtime_error("nx and ny must be multiples of four and at least four");
    const auto fftLimit = static_cast<std::size_t>(std::numeric_limits<int>::max());
    if (parameters.nx > 2 * (fftLimit / 3) || parameters.ny > 2 * (fftLimit / 3))
        throw std::runtime_error("3/2-rule grid dimensions exceed FFT library limits");
    const auto allocationLimit =
        static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max());
    const auto validateProduct = [](std::size_t first, std::size_t second,
                                    const char *description) {
        if (first != 0 && second > allocationLimit / first)
            throw std::runtime_error(std::string(description) +
                                     " exceeds addressable array limits");
    };
    validateProduct(parameters.nx, parameters.ny, "base grid");
    validateProduct(parameters.nxf(), parameters.ny, "base spectral grid");
    validateProduct(parameters.mx(), parameters.my(), "dealiased grid");
    validateProduct(parameters.mxf(), parameters.my(), "dealiased spectral grid");
    if (!(parameters.aspectRatio > 0.0) || !std::isfinite(parameters.aspectRatio))
        throw std::runtime_error("aspectRatio must be finite and positive");
    (void)parameters.spectrumBins();
    if (!(parameters.timeStep > 0.0) || !std::isfinite(parameters.timeStep))
        throw std::runtime_error("timeStep must be finite and positive");
    if (parameters.numberOfSteps == 0 || parameters.outputIntervalSteps == 0)
        throw std::runtime_error("numberOfSteps and outputIntervalSteps must be positive");
    if (parameters.threadCount < 0)
        throw std::runtime_error("threadCount cannot be negative");
    if (parameters.viscosity < 0.0 || parameters.linearDrag < 0.0 ||
        !std::isfinite(parameters.viscosity) || !std::isfinite(parameters.linearDrag) ||
        !std::isfinite(parameters.viscosityOrder) || !std::isfinite(parameters.dragOrder) ||
        !std::isfinite(parameters.beta))
        throw std::runtime_error(
            "viscosity, drag, their orders, and beta must be finite; coefficients "
            "cannot be negative");
    if (parameters.forcingWavenumber <= 0.0 || parameters.forcingWidth < 0.0 ||
        parameters.forcingAmplitude < 0.0 || parameters.targetEnergyInjectionRate < 0.0 ||
        !std::isfinite(parameters.forcingWavenumber) || !std::isfinite(parameters.forcingWidth) ||
        !std::isfinite(parameters.forcingAmplitude) ||
        !std::isfinite(parameters.forcingShapeOrder) ||
        !std::isfinite(parameters.targetEnergyInjectionRate))
        throw std::runtime_error("forcing parameters are invalid or non-finite");
    if (parameters.forcingProfile == ForcingProfile::exponential &&
        !(parameters.forcingShapeOrder > 0.0))
        throw std::runtime_error("forcingShapeOrder must be positive for exponential forcing");
    if (parameters.forcingProfile == ForcingProfile::singleMode) {
        if (parameters.forcingWavenumber != std::floor(parameters.forcingWavenumber))
            throw std::runtime_error("singleMode forcingWavenumber must be an integer mode index");
        if (parameters.targetEnergyInjectionRate > 0.0)
            throw std::runtime_error(
                "targetEnergyInjectionRate applies only to stochastic forcing");
        if (parameters.forcingWavenumber >= static_cast<double>(parameters.nx) / 2.0 ||
            parameters.forcingWavenumber >= static_cast<double>(parameters.ny) / 2.0)
            throw std::runtime_error(
                "singleMode forcingWavenumber must lie below both Nyquist modes");
    }
    if (!parameters.initialConditionFile.empty() &&
        !std::filesystem::exists(parameters.initialConditionFile))
        throw std::runtime_error("initialConditionFile does not exist: " +
                                 parameters.initialConditionFile.string());
    if (parameters.dataDirectory.empty() || parameters.outputDirectory.empty())
        throw std::runtime_error("output directories cannot be empty");
    if (parameters.hdf5CompressionLevel < 0 || parameters.hdf5CompressionLevel > 9)
        throw std::runtime_error("hdf5CompressionLevel must be between 0 and 9");
#ifndef NS2D_HAVE_HDF5
    if (parameters.fieldOutputFormat != FieldOutputFormat::text)
        throw std::runtime_error(
            "fieldOutputFormat requests HDF5, but this build has no HDF5 support");
#endif
}

void writeParameterRecord(const Parameters &parameters, const std::string &backend,
                          const std::filesystem::path &recordDirectory) {
    const auto path = (recordDirectory.empty() ? parameters.outputDirectory : recordDirectory) /
                      "resolved_parameters.txt";
    std::ofstream out(path);
    if (!out)
        throw std::runtime_error("cannot write parameter record: " + path.string());
    out << std::boolalpha << std::setprecision(17);
    const auto write = [&](const char *name, const auto &value) {
        out << name << ' ' << value << '\n';
    };
    write("backend", backend);
    write("solverVersion", NS2D_VERSION);
    write("gitCommit", NS2D_GIT_COMMIT);
    write("gitDirty", NS2D_GIT_DIRTY);
    write("nx", parameters.nx);
    write("ny", parameters.ny);
    write("aspectRatio", parameters.aspectRatio);
    write("boundaryCondition", "periodic");
    write("domainLengthX", parameters.lx());
    write("domainLengthY", parameters.ly());
    write("timeStep", parameters.timeStep);
    write("numberOfSteps", parameters.numberOfSteps);
    write("outputIntervalSteps", parameters.outputIntervalSteps);
    write("integrator", integratorName(parameters.integrator));
    write("betaPlane", parameters.betaPlane);
    write("beta", parameters.beta);
    write("viscosity", parameters.viscosity);
    write("viscosityOrder", parameters.viscosityOrder);
    write("linearDrag", parameters.linearDrag);
    write("dragOrder", parameters.dragOrder);
    write("forcingEnabled", parameters.forcingEnabled);
    write("forcingProfile", forcingProfileName(parameters.forcingProfile));
    write("forcingWavenumber", parameters.forcingWavenumber);
    write("forcingWidth", parameters.forcingWidth);
    write("forcingAmplitude", parameters.forcingAmplitude);
    write("forcingShapeOrder", parameters.forcingShapeOrder);
    write("targetEnergyInjectionRate", parameters.targetEnergyInjectionRate);
    write("randomSeed", parameters.randomSeed);
    write("writeModeDiagnostics", parameters.writeModeDiagnostics);
    write("fieldOutputFormat", fieldOutputFormatName(parameters.fieldOutputFormat));
    write("hdf5CompressionLevel", parameters.hdf5CompressionLevel);
    write("fftwPlanning", fftwPlanningName(parameters.fftwPlanning));
    write("fftwWisdomFile", parameters.fftwWisdomFile.string());
    write("cudaGraphEnabled", parameters.cudaGraphEnabled);
    write("threadCount", parameters.threadCount);
    write("overwriteOutput", parameters.overwriteOutput);
    write("initialConditionFile", parameters.initialConditionFile.string());
    write("dataDirectory", parameters.dataDirectory.string());
    write("outputDirectory", parameters.outputDirectory.string());
    out.close();
    if (!out)
        throw std::runtime_error("failed while writing parameter record: " + path.string());
}
