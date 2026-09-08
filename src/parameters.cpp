#include "parameters.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>

namespace {
constexpr double pi = 3.141592653589793238462643383279502884;

bool parseBool(const std::string &text, const std::string &key) {
  if (text == "true" || text == "1")
    return true;
  if (text == "false" || text == "0")
    return false;
  throw std::runtime_error(key + " must be true or false, got: " + text);
}

template <class T>
T parseNumber(const std::string &text, const std::string &key) {
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
  throw std::runtime_error(
      "forcingProfile must be annulus, exponential, or singleMode");
}
} // namespace

double Parameters::lx() const { return 2.0 * pi * aspectRatio; }
double Parameters::ly() const { return 2.0 * pi; }

std::size_t Parameters::spectrumBins() const {
  const double binWidth = std::min(2.0 * pi / lx(), 2.0 * pi / ly());
  const double maximumKx = pi * static_cast<double>(nx) / lx();
  const double maximumKy = pi * static_cast<double>(ny) / ly();
  const double count =
      std::floor(std::hypot(maximumKx, maximumKy) / binWidth) + 1.0;
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

Parameters readParameters(const std::filesystem::path &path) {
  std::ifstream input(path);
  if (!input)
    throw std::runtime_error("cannot open parameter file: " + path.string());
  Parameters p;
  std::string line;
  std::size_t lineNumber = 0;
  while (std::getline(input, line)) {
    ++lineNumber;
    if (const auto comment = line.find('#'); comment != std::string::npos)
      line.erase(comment);
    line = trim(line);
    if (line.empty())
      continue;
    std::replace(line.begin(), line.end(), '=', ' ');
    std::istringstream fields(line);
    std::string key, value, extra;
    fields >> key >> value;
    if (key.empty() || value.empty() || (fields >> extra))
      throw std::runtime_error("invalid parameter line " +
                               std::to_string(lineNumber));
    const auto readNumber = [&]<class T>(T &destination) {
      destination = parseNumber<T>(value, key);
    };

    if (key == "nx")
      readNumber(p.nx);
    else if (key == "ny")
      readNumber(p.ny);
    else if (key == "aspectRatio")
      readNumber(p.aspectRatio);
    else if (key == "timeStep")
      readNumber(p.timeStep);
    else if (key == "numberOfSteps")
      readNumber(p.numberOfSteps);
    else if (key == "outputIntervalSteps")
      readNumber(p.outputIntervalSteps);
    else if (key == "integrator")
      p.integrator = parseIntegrator(value);
    else if (key == "betaPlane")
      p.betaPlane = parseBool(value, key);
    else if (key == "beta")
      readNumber(p.beta);
    else if (key == "viscosity")
      readNumber(p.viscosity);
    else if (key == "viscosityOrder")
      readNumber(p.viscosityOrder);
    else if (key == "linearDrag")
      readNumber(p.linearDrag);
    else if (key == "dragOrder")
      readNumber(p.dragOrder);
    else if (key == "forcingEnabled")
      p.forcingEnabled = parseBool(value, key);
    else if (key == "forcingProfile")
      p.forcingProfile = parseForcingProfile(value);
    else if (key == "forcingWavenumber")
      readNumber(p.forcingWavenumber);
    else if (key == "forcingWidth")
      readNumber(p.forcingWidth);
    else if (key == "forcingAmplitude")
      readNumber(p.forcingAmplitude);
    else if (key == "forcingShapeOrder")
      readNumber(p.forcingShapeOrder);
    else if (key == "targetEnergyInjectionRate")
      readNumber(p.targetEnergyInjectionRate);
    else if (key == "randomSeed")
      readNumber(p.randomSeed);
    else if (key == "writeModeDiagnostics")
      p.writeModeDiagnostics = parseBool(value, key);
    else if (key == "threadCount")
      readNumber(p.threadCount);
    else if (key == "overwriteOutput")
      p.overwriteOutput = parseBool(value, key);
    else if (key == "initialConditionFile")
      p.initialConditionFile = value;
    else if (key == "dataDirectory")
      p.dataDirectory = value;
    else if (key == "outputDirectory")
      p.outputDirectory = value;
    else
      throw std::runtime_error("unknown parameter key on line " +
                               std::to_string(lineNumber) + ": " + key);
  }
  validateParameters(p);
  return p;
}

void validateParameters(const Parameters &p) {
  if (p.nx < 4 || p.ny < 4 || p.nx % 4 != 0 || p.ny % 4 != 0)
    throw std::runtime_error(
        "nx and ny must be multiples of four and at least four");
  const auto fftLimit =
      static_cast<std::size_t>(std::numeric_limits<int>::max());
  if (p.nx > 2 * (fftLimit / 3) || p.ny > 2 * (fftLimit / 3))
    throw std::runtime_error(
        "3/2-rule grid dimensions exceed FFT library limits");
  const auto allocationLimit =
      static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max());
  const auto validateProduct = [](std::size_t first, std::size_t second,
                                  const char *description) {
    if (first != 0 && second > allocationLimit / first)
      throw std::runtime_error(std::string(description) +
                               " exceeds addressable array limits");
  };
  validateProduct(p.nx, p.ny, "base grid");
  validateProduct(p.nxf(), p.ny, "base spectral grid");
  validateProduct(p.mx(), p.my(), "dealiased grid");
  validateProduct(p.mxf(), p.my(), "dealiased spectral grid");
  if (!(p.aspectRatio > 0.0) || !std::isfinite(p.aspectRatio))
    throw std::runtime_error("aspectRatio must be finite and positive");
  (void)p.spectrumBins();
  if (!(p.timeStep > 0.0) || !std::isfinite(p.timeStep))
    throw std::runtime_error("timeStep must be finite and positive");
  if (p.numberOfSteps == 0 || p.outputIntervalSteps == 0)
    throw std::runtime_error(
        "numberOfSteps and outputIntervalSteps must be positive");
  if (p.threadCount < 0)
    throw std::runtime_error("threadCount cannot be negative");
  if (p.viscosity < 0.0 || p.linearDrag < 0.0 || !std::isfinite(p.viscosity) ||
      !std::isfinite(p.linearDrag) || !std::isfinite(p.viscosityOrder) ||
      !std::isfinite(p.dragOrder) || !std::isfinite(p.beta))
    throw std::runtime_error(
        "viscosity, drag, their orders, and beta must be finite; coefficients "
        "cannot be negative");
  if (p.forcingWavenumber <= 0.0 || p.forcingWidth < 0.0 ||
      p.forcingAmplitude < 0.0 || p.targetEnergyInjectionRate < 0.0 ||
      !std::isfinite(p.forcingWavenumber) || !std::isfinite(p.forcingWidth) ||
      !std::isfinite(p.forcingAmplitude) ||
      !std::isfinite(p.forcingShapeOrder) ||
      !std::isfinite(p.targetEnergyInjectionRate))
    throw std::runtime_error("forcing parameters are invalid or non-finite");
  if (p.forcingProfile == ForcingProfile::exponential &&
      !(p.forcingShapeOrder > 0.0))
    throw std::runtime_error(
        "forcingShapeOrder must be positive for exponential forcing");
  if (p.forcingProfile == ForcingProfile::singleMode) {
    if (p.forcingWavenumber != std::floor(p.forcingWavenumber))
      throw std::runtime_error(
          "singleMode forcingWavenumber must be an integer mode index");
    if (p.targetEnergyInjectionRate > 0.0)
      throw std::runtime_error(
          "targetEnergyInjectionRate applies only to stochastic forcing");
    if (p.forcingWavenumber >= static_cast<double>(p.nx) / 2.0 ||
        p.forcingWavenumber >= static_cast<double>(p.ny) / 2.0)
      throw std::runtime_error(
          "singleMode forcingWavenumber must lie below both Nyquist modes");
  }
  if (!p.initialConditionFile.empty() &&
      !std::filesystem::exists(p.initialConditionFile))
    throw std::runtime_error("initialConditionFile does not exist: " +
                             p.initialConditionFile.string());
  if (p.dataDirectory.empty() || p.outputDirectory.empty())
    throw std::runtime_error("output directories cannot be empty");
}

void writeParameterRecord(const Parameters &p, const std::string &backend,
                          const std::filesystem::path &recordDirectory) {
  const auto path =
      (recordDirectory.empty() ? p.outputDirectory : recordDirectory) /
      "resolved_parameters.txt";
  std::ofstream out(path);
  if (!out)
    throw std::runtime_error("cannot write parameter record: " + path.string());
  out << std::boolalpha << std::setprecision(17);
  const auto write = [&](const char *name, const auto &value) {
    out << name << ' ' << value << '\n';
  };
  write("backend", backend);
  write("nx", p.nx);
  write("ny", p.ny);
  write("aspectRatio", p.aspectRatio);
  write("domainLengthX", p.lx());
  write("domainLengthY", p.ly());
  write("timeStep", p.timeStep);
  write("numberOfSteps", p.numberOfSteps);
  write("outputIntervalSteps", p.outputIntervalSteps);
  write("integrator", integratorName(p.integrator));
  write("betaPlane", p.betaPlane);
  write("beta", p.beta);
  write("viscosity", p.viscosity);
  write("viscosityOrder", p.viscosityOrder);
  write("linearDrag", p.linearDrag);
  write("dragOrder", p.dragOrder);
  write("forcingEnabled", p.forcingEnabled);
  write("forcingProfile", forcingProfileName(p.forcingProfile));
  write("forcingWavenumber", p.forcingWavenumber);
  write("forcingWidth", p.forcingWidth);
  write("forcingAmplitude", p.forcingAmplitude);
  write("forcingShapeOrder", p.forcingShapeOrder);
  write("targetEnergyInjectionRate", p.targetEnergyInjectionRate);
  write("randomSeed", p.randomSeed);
  write("writeModeDiagnostics", p.writeModeDiagnostics);
  write("threadCount", p.threadCount);
  write("overwriteOutput", p.overwriteOutput);
  write("initialConditionFile", p.initialConditionFile.string());
  write("dataDirectory", p.dataDirectory.string());
  write("outputDirectory", p.outputDirectory.string());
  out.close();
  if (!out)
    throw std::runtime_error("failed while writing parameter record: " +
                             path.string());
}
