#include "output.hpp"
#include "spectral.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace {
constexpr std::size_t checkpointChunkComplexValues = 4096;

bool finiteSpectralField(const SpectralField &field) {
    return std::all_of(field.begin(), field.end(), [](Complex x) {
        return std::isfinite(x.real()) && std::isfinite(x.imag());
    });
}

std::filesystem::path vorticityPath(const Parameters &parameters, std::uint64_t frame) {
    std::ostringstream name;
    name << "vorticity_" << std::setw(8) << std::setfill('0') << frame << ".dat";
    return parameters.dataDirectory / name.str();
}

std::filesystem::path legacyVorticityPath(const Parameters &parameters, int frame) {
    std::ostringstream name;
    name << "w." << std::setw(6) << std::setfill('0') << frame;
    return parameters.dataDirectory / name.str();
}

std::filesystem::path checkpointPath(const Parameters &parameters, std::uint64_t frame) {
    std::ostringstream name;
    name << "checkpoint_" << std::setw(8) << std::setfill('0') << frame << ".bin";
    return parameters.dataDirectory / name.str();
}

SpectralField readCheckpoint(const Parameters &parameters, std::uint64_t frame) {
    const auto path = checkpointPath(parameters, frame);
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("cannot open spectral checkpoint: " + path.string());
    char magic[8]{};
    std::uint64_t nx = 0, ny = 0, count = 0;
    input.read(magic, sizeof(magic));
    input.read(reinterpret_cast<char *>(&nx), sizeof(nx));
    input.read(reinterpret_cast<char *>(&ny), sizeof(ny));
    input.read(reinterpret_cast<char *>(&count), sizeof(count));
    const std::string format(magic, 7);
    if (!input || (format != "NS2DCP1" && format != "NS2DCP2") || nx != parameters.nx ||
        ny != parameters.ny || count != parameters.spectralSize())
        throw std::runtime_error("invalid spectral checkpoint header: " + path.string());
    SpectralField field(static_cast<std::size_t>(count));
    if (format == "NS2DCP1") {
        // Compatibility with checkpoints produced during early development.
        input.read(reinterpret_cast<char *>(field.data()),
                   static_cast<std::streamsize>(field.size() * sizeof(Complex)));
    } else {
        std::vector<double> buffer(2 * checkpointChunkComplexValues);
        for (std::size_t offset = 0; offset < field.size();) {
            const std::size_t chunk = std::min(checkpointChunkComplexValues, field.size() - offset);
            input.read(reinterpret_cast<char *>(buffer.data()),
                       static_cast<std::streamsize>(2 * chunk * sizeof(double)));
            if (!input)
                break;
            for (std::size_t i = 0; i < chunk; ++i)
                field[offset + i] = Complex(buffer[2 * i], buffer[2 * i + 1]);
            offset += chunk;
        }
    }
    if (!input || input.peek() != std::ifstream::traits_type::eof())
        throw std::runtime_error("invalid spectral checkpoint payload: " + path.string());
    if (!finiteSpectralField(field))
        throw std::runtime_error("spectral checkpoint contains non-finite values: " +
                                 path.string());
    return field;
}

std::ofstream numericOutput(const std::filesystem::path &path,
                            std::ios::openmode mode = std::ios::out) {
    std::ofstream out(path, mode);
    if (!out)
        throw std::runtime_error("cannot write output file: " + path.string());
    out << std::scientific << std::setprecision(12);
    return out;
}

std::vector<double> readRealField(const std::filesystem::path &path, std::size_t count) {
    std::ifstream input(path);
    if (!input)
        throw std::runtime_error("cannot open vorticity field: " + path.string());
    std::vector<double> field(count);
    for (double &value : field) {
        if (!(input >> value))
            throw std::runtime_error("vorticity field has too few values: " + path.string());
        if (!std::isfinite(value))
            throw std::runtime_error("vorticity field contains a non-finite value: " +
                                     path.string());
    }
    double extra = 0.0;
    if (input >> extra)
        throw std::runtime_error("vorticity field has too many values: " + path.string());
    return field;
}

void initializeCsv(const std::filesystem::path &path, const std::string &header, bool append,
                   bool overwrite) {
    if (append && std::filesystem::exists(path)) {
        std::ifstream input(path);
        std::string existingHeader;
        std::getline(input, existingHeader);
        if (existingHeader != header)
            throw std::runtime_error("CSV header does not match this solver version: " +
                                     path.string());
        return;
    }
    if (!append && std::filesystem::exists(path) && !overwrite)
        throw std::runtime_error("refusing to overwrite existing output: " + path.string());
    std::ofstream output(path, std::ios::trunc);
    if (!output)
        throw std::runtime_error("cannot initialize CSV file: " + path.string());
    output << header << '\n';
    output.close();
    if (!output)
        throw std::runtime_error("failed while initializing CSV file: " + path.string());
}

std::uint64_t lastCsvFrame(const std::filesystem::path &path) {
    constexpr std::streamoff tailBytes = 16384;
    std::ifstream input(path, std::ios::binary);
    input.seekg(0, std::ios::end);
    const std::streamoff size = input.tellg();
    if (size < 0)
        throw std::runtime_error("cannot inspect CSV file: " + path.string());
    const std::streamoff offset = std::max<std::streamoff>(0, size - tailBytes);
    input.seekg(offset);
    std::string line, lastLine;
    if (offset > 0)
        std::getline(input, line); // Discard the first possibly partial row.
    while (std::getline(input, line))
        if (!line.empty())
            lastLine = line;
    if (lastLine.empty())
        throw std::runtime_error("CSV is empty or has a row longer than 16 KiB: " + path.string());
    if (offset == 0 && lastLine.starts_with("time,frame,"))
        return 0; // Header-only file.
    const auto firstComma = lastLine.find(',');
    const auto secondComma = lastLine.find(',', firstComma + 1);
    if (firstComma == std::string::npos || secondComma == std::string::npos)
        throw std::runtime_error("malformed diagnostics CSV: " + path.string());
    const std::string value = lastLine.substr(firstComma + 1, secondComma - firstComma - 1);
    std::size_t consumed = 0;
    const auto frame = std::stoull(value, &consumed);
    if (consumed != value.size())
        throw std::runtime_error("invalid frame in diagnostics CSV: " + path.string());
    return frame;
}

bool containsModernSolverData(const std::filesystem::path &directory) {
    if (!std::filesystem::exists(directory))
        return false;
    for (const auto &entry : std::filesystem::directory_iterator(directory)) {
        const std::string name = entry.path().filename().string();
        if (name == "restart_state.txt" || name.starts_with("vorticity_") ||
            name.starts_with("checkpoint_"))
            return true;
    }
    return false;
}

bool containsSolverOutput(const std::filesystem::path &directory) {
    constexpr std::array names{"diagnostics.csv",
                               "spectra.csv",
                               "fluxes.csv",
                               "modes.csv",
                               "forcing_summary.csv",
                               "forcing_spectrum.csv",
                               "resolved_parameters.txt"};
    return std::any_of(names.begin(), names.end(),
                       [&](const char *name) { return std::filesystem::exists(directory / name); });
}

} // namespace

RestartState readRestart(const Parameters &parameters, BaseTransform &transform, bool isRoot) {
    std::filesystem::create_directories(parameters.dataDirectory);
    std::filesystem::create_directories(parameters.outputDirectory);
    RestartState state;
    std::vector<double> real;
    const auto restartPath = parameters.dataDirectory / "restart_state.txt";

    if (std::filesystem::exists(restartPath)) {
        std::ifstream input(restartPath);
        std::string format;
        std::getline(input, format);
        const bool versionOne = format == "ns2d_restart_v1";
        const bool versionThree = format == "ns2d_restart_v3";
        if (!versionOne && format != "ns2d_restart_v2" && !versionThree)
            throw std::runtime_error("unsupported restart format: " + restartPath.string());
        std::size_t savedNx = 0, savedNy = 0;
        double savedAspectRatio = 0.0;
        std::string key;
        if (!(input >> key >> state.time) || key != "time" || !(input >> key >> state.frame) ||
            key != "frame" || !(input >> key >> savedNx) || key != "nx" ||
            !(input >> key >> savedNy) || key != "ny" || !(input >> key >> savedAspectRatio) ||
            key != "aspectRatio" || !(input >> key >> state.randomSeed) || key != "randomSeed")
            throw std::runtime_error("malformed restart metadata: " + restartPath.string());
        if (!std::isfinite(state.time) || state.time < 0.0 ||
            state.frame == std::numeric_limits<std::uint64_t>::max())
            throw std::runtime_error("invalid time or frame in restart metadata: " +
                                     restartPath.string());
        input >> std::ws;
        std::getline(input, key, ' ');
        if (key != "randomEngine")
            throw std::runtime_error("restart is missing randomEngine state");
        std::getline(input, state.randomEngineState);
        if (!versionOne) {
            std::getline(input, key, ' ');
            if (key != "randomDistribution")
                throw std::runtime_error("restart is missing randomDistribution state");
            std::getline(input, state.randomDistributionState);
        }
        if (versionThree) {
            int version = 0;
            if (!(input >> key >> version) || key != "numericsVersion" || version != 3)
                throw std::runtime_error("unsupported restart numerics version");
        } else if (isRoot) {
            std::cout << "warning: loading an older checkpoint; continuation uses "
                         "the updated "
                         "linear and stochastic integration methods\n";
        }
        if (savedNx != parameters.nx || savedNy != parameters.ny ||
            savedAspectRatio != parameters.aspectRatio)
            throw std::runtime_error(
                "restart grid or aspect ratio does not match the parameter file");
        state.vorticity = readCheckpoint(parameters, state.frame);
        state.restarting = true;
        if (isRoot)
            std::cout << "restarting at time " << state.time << " from frame " << state.frame
                      << '\n';
    } else if (std::filesystem::exists(parameters.dataDirectory / "curframe.dat")) {
        double legacyTime = 0.0;
        int legacyFrame = -1;
        std::ifstream input(parameters.dataDirectory / "curframe.dat");
        if (!(input >> legacyTime >> legacyFrame))
            throw std::runtime_error("invalid legacy curframe.dat");
        if (!std::isfinite(legacyTime) || legacyTime < 0.0)
            throw std::runtime_error("invalid time in legacy curframe.dat");
        if (legacyFrame < 0) {
            real.assign(parameters.nx * parameters.ny, 0.0);
            if (isRoot)
                std::cout << "legacy curframe.dat requests a zero-vorticity start\n";
        } else {
            state.time = legacyTime;
            state.frame = static_cast<std::uint64_t>(legacyFrame);
            real = readRealField(legacyVorticityPath(parameters, legacyFrame),
                                 parameters.nx * parameters.ny);
            state.restarting = true;
            if (isRoot)
                std::cout << "warning: importing a legacy restart without random-generator "
                             "state; future stochastic forcing will not reproduce the "
                             "uninterrupted trajectory\n";
        }
    } else {
        state.time = 0.0;
        state.frame = 0;
        if (parameters.initialConditionFile.empty()) {
            real.assign(parameters.nx * parameters.ny, 0.0);
            if (isRoot)
                std::cout << "starting from zero vorticity\n";
        } else {
            real = readRealField(parameters.initialConditionFile, parameters.nx * parameters.ny);
            if (isRoot)
                std::cout << "starting from " << parameters.initialConditionFile << '\n';
        }
    }
    if (state.vorticity.empty()) {
        transform.forward(real, state.vorticity);
        enforceRealityConstraints(state.vorticity, parameters);
    }
    if (!finiteSpectralField(state.vorticity))
        throw std::runtime_error("initial or restarted vorticity contains non-finite coefficients");
    return state;
}

void prepareOutputFiles(const Parameters &parameters, bool restarting, std::uint64_t restartFrame) {
    const auto diagnosticsPath = parameters.outputDirectory / "diagnostics.csv";
    if (!restarting && !parameters.overwriteOutput &&
        (containsModernSolverData(parameters.dataDirectory) ||
         containsSolverOutput(parameters.outputDirectory)))
        throw std::runtime_error("solver output already exists; use new directories or set "
                                 "overwriteOutput true");
    if (restarting) {
        for (const char *name : {"diagnostics.csv", "spectra.csv", "fluxes.csv", "modes.csv"}) {
            const auto path = parameters.outputDirectory / name;
            if (std::filesystem::exists(path) && lastCsvFrame(path) > restartFrame)
                throw std::runtime_error(path.string() + " contains frames newer than the restart");
        }
        if ((std::filesystem::exists(vorticityPath(parameters, restartFrame + 1)) ||
             std::filesystem::exists(checkpointPath(parameters, restartFrame + 1))) &&
            !parameters.overwriteOutput)
            throw std::runtime_error(
                "the next restart frame already exists; refusing to overwrite it");
    }
    initializeCsv(diagnosticsPath,
                  "time,frame,energy,enstrophy,energy_dissipation_drag,energy_dissipation_"
                  "viscosity,enstrophy_dissipation_drag,enstrophy_dissipation_viscosity",
                  restarting, parameters.overwriteOutput);
    initializeCsv(parameters.outputDirectory / "spectra.csv",
                  "time,frame,wavenumber,energy_spectrum,enstrophy_spectrum,segment_mean_"
                  "energy_spectrum,segment_mean_enstrophy_spectrum",
                  restarting, parameters.overwriteOutput);
    initializeCsv(parameters.outputDirectory / "fluxes.csv",
                  "time,frame,wavenumber,energy_flux,enstrophy_flux,segment_mean_energy_"
                  "flux,segment_mean_enstrophy_flux",
                  restarting, parameters.overwriteOutput);
    if (parameters.writeModeDiagnostics)
        initializeCsv(parameters.outputDirectory / "modes.csv",
                      "time,frame,omega_1_0_real,omega_1_0_imag,omega_0_1_real,"
                      "omega_0_1_imag,omega_1_1_real,omega_1_1_imag,omega_2_1_real,"
                      "omega_2_1_imag,omega_0_3_real,omega_0_3_imag",
                      restarting, parameters.overwriteOutput);
}

void writeVorticity(const Parameters &parameters, BaseTransform &transform,
                    const SpectralField &vorticity, std::uint64_t frame) {
    const auto path = vorticityPath(parameters, frame);
    if (std::filesystem::exists(path) && !parameters.overwriteOutput)
        throw std::runtime_error("refusing to overwrite vorticity snapshot: " + path.string());
    std::vector<double> real;
    transform.inverse(vorticity, real);
    const auto temporary = std::filesystem::path(path.string() + ".tmp");
    auto out = numericOutput(temporary);
    for (std::size_t y = 0; y < parameters.ny; ++y) {
        for (std::size_t x = 0; x < parameters.nx; ++x)
            out << real[y * parameters.nx + x] << (x + 1 == parameters.nx ? '\n' : ' ');
    }
    out.close();
    if (!out)
        throw std::runtime_error("failed while writing vorticity snapshot: " + path.string());
    std::filesystem::rename(temporary, path);
}

void writeRestart(const Parameters &parameters, double time, std::uint64_t frame,
                  const SpectralField &vorticity, const std::string &randomEngineState,
                  const std::string &randomDistributionState) {
    if (!finiteSpectralField(vorticity))
        throw std::runtime_error("refusing to checkpoint non-finite spectral coefficients");
    const auto binaryPath = checkpointPath(parameters, frame);
    if (std::filesystem::exists(binaryPath) && !parameters.overwriteOutput)
        throw std::runtime_error("refusing to overwrite spectral checkpoint: " +
                                 binaryPath.string());
    const auto binaryTemporary = std::filesystem::path(binaryPath.string() + ".tmp");
    std::ofstream checkpoint(binaryTemporary, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!checkpoint)
        throw std::runtime_error("cannot write spectral checkpoint: " + binaryPath.string());
    const char magic[8] = {'N', 'S', '2', 'D', 'C', 'P', '2', '\0'};
    const std::uint64_t nx = parameters.nx, ny = parameters.ny, count = vorticity.size();
    checkpoint.write(magic, sizeof(magic));
    checkpoint.write(reinterpret_cast<const char *>(&nx), sizeof(nx));
    checkpoint.write(reinterpret_cast<const char *>(&ny), sizeof(ny));
    checkpoint.write(reinterpret_cast<const char *>(&count), sizeof(count));
    std::vector<double> buffer(2 * checkpointChunkComplexValues);
    for (std::size_t offset = 0; offset < vorticity.size();) {
        const std::size_t chunk = std::min(checkpointChunkComplexValues, vorticity.size() - offset);
        for (std::size_t i = 0; i < chunk; ++i) {
            buffer[2 * i] = vorticity[offset + i].real();
            buffer[2 * i + 1] = vorticity[offset + i].imag();
        }
        checkpoint.write(reinterpret_cast<const char *>(buffer.data()),
                         static_cast<std::streamsize>(2 * chunk * sizeof(double)));
        offset += chunk;
    }
    checkpoint.close();
    if (!checkpoint)
        throw std::runtime_error("failed while writing spectral checkpoint: " +
                                 binaryPath.string());
    std::filesystem::rename(binaryTemporary, binaryPath);

    const auto path = parameters.dataDirectory / "restart_state.txt";
    const auto temporaryPath = parameters.dataDirectory / "restart_state.tmp";
    std::ofstream out(temporaryPath, std::ios::trunc);
    if (!out)
        throw std::runtime_error("cannot write restart metadata: " + path.string());
    out << std::setprecision(17) << "ns2d_restart_v3\n"
        << "time " << time << '\n'
        << "frame " << frame << '\n'
        << "nx " << parameters.nx << '\n'
        << "ny " << parameters.ny << '\n'
        << "aspectRatio " << parameters.aspectRatio << '\n'
        << "randomSeed " << parameters.randomSeed << '\n'
        << "randomEngine " << randomEngineState << '\n'
        << "randomDistribution " << randomDistributionState << '\n'
        << "numericsVersion 3\n";
    out.close();
    if (!out)
        throw std::runtime_error("failed while writing restart metadata: " +
                                 temporaryPath.string());
    std::filesystem::rename(temporaryPath, path);
}
