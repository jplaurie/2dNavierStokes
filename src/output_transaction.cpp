#include "output.hpp"

#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>

namespace {
constexpr std::array csvNames{"diagnostics.csv", "spectra.csv", "fluxes.csv", "modes.csv"};

std::array<std::filesystem::path, 3> frameFiles(const Parameters &parameters, std::uint64_t frame) {
    std::ostringstream suffix;
    suffix << std::setw(8) << std::setfill('0') << frame;
    return {parameters.dataDirectory / ("vorticity_" + suffix.str() + ".dat"),
            parameters.dataDirectory / ("checkpoint_" + suffix.str() + ".bin"),
            parameters.dataDirectory / ("vorticity_" + suffix.str() + ".h5")};
}

std::filesystem::path appended(const std::filesystem::path &path, const char *suffix) {
    return path.string() + suffix;
}

std::optional<std::uint64_t> committedFrame(const Parameters &parameters) {
    const auto path = parameters.dataDirectory / "restart_state.txt";
    if (!std::filesystem::exists(path))
        return std::nullopt;
    std::ifstream in(path);
    std::string format, key;
    double time;
    std::uint64_t frame;
    if (!(in >> format) ||
        (format != "ns2d_restart_v1" && format != "ns2d_restart_v2" &&
         format != "ns2d_restart_v3") ||
        !(in >> key >> time) || key != "time" || !(in >> key >> frame) || key != "frame")
        throw std::runtime_error("cannot recover output with malformed restart metadata");
    return frame;
}

struct Journal {
    std::uint64_t frame{};
    bool previousMetadata{};
    std::uint64_t previousFrame{};
    std::array<bool, 4> csvExisted{};
    std::array<std::uintmax_t, 4> csvSizes{};
    std::size_t frameFileCount = 3;
    std::array<bool, 3> frameExisted{};
};

Journal readJournal(const Parameters &parameters) {
    std::ifstream in(parameters.dataDirectory / "output_transaction.txt");
    std::string format, directory;
    Journal journal;
    if (!(in >> format >> std::quoted(directory) >> journal.frame >> journal.previousMetadata >>
          journal.previousFrame) ||
        (format != "ns2d_output_transaction_v1" && format != "ns2d_output_transaction_v2"))
        throw std::runtime_error("malformed output transaction journal");
    journal.frameFileCount = format == "ns2d_output_transaction_v1" ? 2 : 3;
    if (std::filesystem::canonical(parameters.outputDirectory) != std::filesystem::path(directory))
        throw std::runtime_error(
            "recover the interrupted run using its original outputDirectory: " + directory);
    for (std::size_t i = 0; i < csvNames.size(); ++i)
        if (!(in >> journal.csvExisted[i] >> journal.csvSizes[i]))
            throw std::runtime_error("malformed CSV offsets in output transaction journal");
    for (std::size_t i = 0; i < journal.frameFileCount; ++i)
        if (!(in >> journal.frameExisted[i]))
            throw std::runtime_error("malformed frame files in output transaction journal");
    if (!(in >> std::ws).eof())
        throw std::runtime_error("unexpected data in output transaction journal");
    return journal;
}

void removeJournal(const Parameters &parameters) {
    std::filesystem::remove(parameters.dataDirectory / "output_transaction.txt");
    std::filesystem::remove(parameters.dataDirectory / "output_transaction.tmp");
}
} // namespace

bool recoverOutputTransaction(const Parameters &parameters) {
    if (!std::filesystem::exists(parameters.dataDirectory / "output_transaction.txt"))
        return false;
    const Journal journal = readJournal(parameters);
    const auto committed = committedFrame(parameters);
    if (committed && *committed == journal.frame) {
        finishOutputTransaction(parameters);
        return false;
    }
    if (committed.has_value() != journal.previousMetadata ||
        (committed && *committed != journal.previousFrame))
        throw std::runtime_error(
            "restart metadata does not match the interrupted output transaction");
    // Validate every offset before changing any file; a short committed history
    // is damage, not an interrupted append that can safely be rolled back.
    for (std::size_t i = 0; i < csvNames.size(); ++i) {
        const auto path = parameters.outputDirectory / csvNames[i];
        if (journal.csvExisted[i] && (!std::filesystem::exists(path) ||
                                      std::filesystem::file_size(path) < journal.csvSizes[i]))
            throw std::runtime_error("committed CSV data is missing: " + path.string());
    }
    for (std::size_t i = 0; i < csvNames.size(); ++i) {
        const auto path = parameters.outputDirectory / csvNames[i];
        if (journal.csvExisted[i])
            std::filesystem::resize_file(path, journal.csvSizes[i]);
        else
            std::filesystem::remove(path);
    }
    const auto files = frameFiles(parameters, journal.frame);
    for (std::size_t i = 0; i < journal.frameFileCount; ++i) {
        const auto backup = appended(files[i], ".previous");
        if (std::filesystem::exists(backup))
            std::filesystem::rename(backup, files[i]);
        else if (!journal.frameExisted[i])
            std::filesystem::remove(files[i]);
        std::filesystem::remove(appended(files[i], ".tmp"));
    }
    std::filesystem::remove(parameters.dataDirectory / "restart_state.tmp");
    removeJournal(parameters);
    std::cout << "recovered interrupted output frame " << journal.frame << '\n';
    return journal.frame == 0 && !committed;
}

void beginOutputTransaction(const Parameters &parameters, std::uint64_t frame) {
    if (std::filesystem::exists(parameters.dataDirectory / "output_transaction.txt"))
        throw std::runtime_error("an output transaction is already active");
    Journal journal;
    journal.frame = frame;
    const auto previous = committedFrame(parameters);
    journal.previousMetadata = previous.has_value();
    journal.previousFrame = previous.value_or(0);
    const auto files = frameFiles(parameters, frame);
    for (std::size_t i = 0; i < files.size(); ++i) {
        journal.frameExisted[i] = std::filesystem::exists(files[i]);
        if (journal.frameExisted[i] &&
            (!parameters.overwriteOutput || !std::filesystem::is_regular_file(files[i])))
            throw std::runtime_error("refusing to overwrite output frame file: " +
                                     files[i].string());
        for (const char *suffix : {".tmp", ".previous"})
            if (std::filesystem::exists(appended(files[i], suffix)))
                throw std::runtime_error("untracked temporary output file: " +
                                         appended(files[i], suffix).string());
    }
    for (std::size_t i = 0; i < csvNames.size(); ++i) {
        const auto path = parameters.outputDirectory / csvNames[i];
        journal.csvExisted[i] = std::filesystem::exists(path);
        if (journal.csvExisted[i])
            journal.csvSizes[i] = std::filesystem::file_size(path);
    }
    const auto temporary = parameters.dataDirectory / "output_transaction.tmp";
    std::ofstream out(temporary);
    out << "ns2d_output_transaction_v2\n"
        << std::quoted(std::filesystem::canonical(parameters.outputDirectory).string()) << '\n'
        << journal.frame << ' ' << journal.previousMetadata << ' ' << journal.previousFrame << '\n';
    for (std::size_t i = 0; i < csvNames.size(); ++i)
        out << journal.csvExisted[i] << ' ' << journal.csvSizes[i] << '\n';
    for (const bool existed : journal.frameExisted)
        out << existed << '\n';
    out.close();
    if (!out)
        throw std::runtime_error("cannot write output transaction journal");
    std::filesystem::rename(temporary, parameters.dataDirectory / "output_transaction.txt");
    for (std::size_t i = 0; i < files.size(); ++i)
        if (journal.frameExisted[i])
            std::filesystem::rename(files[i], appended(files[i], ".previous"));
}

void finishOutputTransaction(const Parameters &parameters) {
    const Journal journal = readJournal(parameters);
    if (committedFrame(parameters) != std::optional{journal.frame})
        throw std::runtime_error("cannot finish an uncommitted output transaction");
    for (const auto &path : frameFiles(parameters, journal.frame)) {
        std::filesystem::remove(appended(path, ".previous"));
        std::filesystem::remove(appended(path, ".tmp"));
    }
    removeJournal(parameters);
}

void writeRunRecords(const Parameters &parameters, const std::string &backend, double time,
                     std::uint64_t frame, const std::vector<double> &amplitude,
                     std::size_t forcedModes, double energyCoefficient,
                     double enstrophyCoefficient) {
    const auto history = parameters.outputDirectory / "segments";
    constexpr std::array names{"resolved_parameters.txt", "forcing_summary.csv",
                               "forcing_spectrum.csv"};
    if (!std::filesystem::exists(history)) {
        std::filesystem::create_directories(history);
        // Preserve records from solver versions that predate segment histories.
        for (const char *name : names)
            if (std::filesystem::exists(parameters.outputDirectory / name)) {
                std::filesystem::create_directories(history / "imported");
                std::filesystem::copy_file(parameters.outputDirectory / name,
                                           history / "imported" / name);
            }
    }
    std::filesystem::path segment;
    for (std::uint64_t index = 1;; ++index) {
        std::ostringstream name;
        name << "segment_" << std::setw(8) << std::setfill('0') << index;
        segment = history / name.str();
        if (std::filesystem::create_directory(segment))
            break;
    }
    writeParameterRecord(parameters, backend, segment);
    Parameters recordParameters = parameters;
    recordParameters.outputDirectory = segment;
    writeForcingFiles(recordParameters, amplitude, forcedModes, energyCoefficient,
                      enstrophyCoefficient);
    std::ofstream manifest(segment / "segment.txt");
    manifest << std::setprecision(17) << "startTime " << time << "\nstartFrame " << frame
             << "\nstochasticUpdate exact_linear_covariance_v1\n";
    manifest.close();
    if (!manifest)
        throw std::runtime_error("cannot write run segment record");
    for (const char *name : names) {
        const auto temporary = parameters.outputDirectory / (std::string(name) + ".tmp");
        std::filesystem::copy_file(segment / name, temporary,
                                   std::filesystem::copy_options::overwrite_existing);
        std::filesystem::rename(temporary, parameters.outputDirectory / name);
    }
}
