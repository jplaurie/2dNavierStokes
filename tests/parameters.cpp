#include "parameters.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace {
class TemporaryDirectory {
  public:
    TemporaryDirectory() {
        const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
        path_ = std::filesystem::temp_directory_path() /
                ("ns2d_parameter_tests_" + std::to_string(suffix));
        std::filesystem::create_directory(path_);
    }

    ~TemporaryDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }
    const std::filesystem::path &path() const { return path_; }

  private:
    std::filesystem::path path_;
};

void writeText(const std::filesystem::path &path, const std::string &text) {
    std::ofstream output(path);
    output << text;
    if (!output)
        throw std::runtime_error("could not create parameter test input");
}

void require(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}

template <class Operation>
void requireFailure(Operation operation, const std::string &expectedMessage) {
    try {
        operation();
    } catch (const std::runtime_error &error) {
        if (std::string(error.what()).find(expectedMessage) != std::string::npos)
            return;
        throw;
    }
    throw std::runtime_error("invalid parameter input was accepted");
}
} // namespace

int main() {
    TemporaryDirectory temporary;
    const auto validFile = temporary.path() / "valid.params";
    writeText(validFile, "# Both assignment styles are supported.\n"
                         "nx = 8\n"
                         "ny 12\n"
                         "aspectRatio 2.0\n"
                         "timeStep 0.005\n"
                         "numberOfSteps 20\n"
                         "outputIntervalSteps 4\n"
                         "integrator rk2\n"
                         "forcingEnabled false\n"
                         "forcingProfile singleMode\n"
                         "forcingWavenumber 2\n"
                         "threadCount 3\n"
                         "fieldOutputFormat text\n"
                         "hdf5CompressionLevel 4\n"
                         "fftwPlanning patient\n"
                         "fftwWisdomFile plans.wisdom\n"
                         "cudaGraphEnabled true\n"
                         "overwriteOutput true\n"
                         "dataDirectory data\n"
                         "outputDirectory output\n");

    const Parameters parameters = readParameters(validFile);
    require(parameters.nx == 8 && parameters.ny == 12, "grid dimensions were not parsed");
    require(parameters.aspectRatio == 2.0, "aspect ratio was not parsed");
    require(parameters.timeStep == 0.005, "time step was not parsed");
    require(parameters.numberOfSteps == 20 && parameters.outputIntervalSteps == 4,
            "run length was not parsed");
    require(parameters.integrator == Integrator::integratingFactorRk2,
            "integrator was not converted to its enum");
    require(parameters.forcingProfile == ForcingProfile::singleMode,
            "forcing profile was not converted to its enum");
    require(!parameters.forcingEnabled && parameters.threadCount == 3 && parameters.overwriteOutput,
            "boolean or integer settings were not parsed");
    require(parameters.fieldOutputFormat == FieldOutputFormat::text &&
                parameters.hdf5CompressionLevel == 4 &&
                parameters.fftwPlanning == FftwPlanning::patient &&
                parameters.fftwWisdomFile == "plans.wisdom" && parameters.cudaGraphEnabled,
            "structural runtime settings were not parsed");

    const auto unknownFile = temporary.path() / "unknown.params";
    writeText(unknownFile, "nx 8\nunknownSetting 1\n");
    requireFailure([&] { (void)readParameters(unknownFile); }, "unknown parameter key on line 2");

    const auto malformedFile = temporary.path() / "malformed.params";
    writeText(malformedFile, "nx 8 extra\n");
    requireFailure([&] { (void)readParameters(malformedFile); }, "invalid parameter line 1");

    const auto negativeFile = temporary.path() / "negative.params";
    writeText(negativeFile, "numberOfSteps -1\n");
    requireFailure([&] { (void)readParameters(negativeFile); }, "numberOfSteps cannot be negative");

    const auto compressionFile = temporary.path() / "compression.params";
    writeText(compressionFile, "hdf5CompressionLevel 10\n");
    requireFailure([&] { (void)readParameters(compressionFile); },
                   "hdf5CompressionLevel must be between 0 and 9");
}
