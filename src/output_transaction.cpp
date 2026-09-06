#include "output.hpp"

#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>

namespace {
constexpr std::array csvNames{"diagnostics.csv", "spectra.csv", "fluxes.csv",
                              "modes.csv"};

std::array<std::filesystem::path, 2> frameFiles(const Parameters &p,
                                                std::uint64_t frame) {
  std::ostringstream suffix;
  suffix << std::setw(8) << std::setfill('0') << frame;
  return {p.dataDirectory / ("vorticity_" + suffix.str() + ".dat"),
          p.dataDirectory / ("checkpoint_" + suffix.str() + ".bin")};
}

std::filesystem::path appended(const std::filesystem::path &path,
                               const char *suffix) {
  return path.string() + suffix;
}

std::optional<std::uint64_t> committedFrame(const Parameters &p) {
  const auto path = p.dataDirectory / "restart_state.txt";
  if (!std::filesystem::exists(path))
    return std::nullopt;
  std::ifstream in(path);
  std::string format, key;
  double time;
  std::uint64_t frame;
  if (!(in >> format) ||
      (format != "ns2d_restart_v1" && format != "ns2d_restart_v2" &&
       format != "ns2d_restart_v3") ||
      !(in >> key >> time) || key != "time" || !(in >> key >> frame) ||
      key != "frame")
    throw std::runtime_error(
        "cannot recover output with malformed restart metadata");
  return frame;
}

struct Journal {
  std::uint64_t frame{};
  bool previousMetadata{};
  std::uint64_t previousFrame{};
  std::array<bool, 4> csvExisted{};
  std::array<std::uintmax_t, 4> csvSizes{};
  std::array<bool, 2> frameExisted{};
};

Journal readJournal(const Parameters &p) {
  std::ifstream in(p.dataDirectory / "output_transaction.txt");
  std::string format, directory;
  Journal j;
  if (!(in >> format >> std::quoted(directory) >> j.frame >>
        j.previousMetadata >> j.previousFrame) ||
      format != "ns2d_output_transaction_v1")
    throw std::runtime_error("malformed output transaction journal");
  if (std::filesystem::canonical(p.outputDirectory) !=
      std::filesystem::path(directory))
    throw std::runtime_error(
        "recover the interrupted run using its original outputDirectory: " +
        directory);
  for (std::size_t i = 0; i < csvNames.size(); ++i)
    if (!(in >> j.csvExisted[i] >> j.csvSizes[i]))
      throw std::runtime_error(
          "malformed CSV offsets in output transaction journal");
  for (auto &existed : j.frameExisted)
    if (!(in >> existed))
      throw std::runtime_error(
          "malformed frame files in output transaction journal");
  if (!(in >> std::ws).eof())
    throw std::runtime_error("unexpected data in output transaction journal");
  return j;
}

void removeJournal(const Parameters &p) {
  std::filesystem::remove(p.dataDirectory / "output_transaction.txt");
  std::filesystem::remove(p.dataDirectory / "output_transaction.tmp");
}
} // namespace

bool recoverOutputTransaction(const Parameters &p) {
  if (!std::filesystem::exists(p.dataDirectory / "output_transaction.txt"))
    return false;
  const Journal j = readJournal(p);
  const auto committed = committedFrame(p);
  if (committed && *committed == j.frame) {
    finishOutputTransaction(p);
    return false;
  }
  if (committed.has_value() != j.previousMetadata ||
      (committed && *committed != j.previousFrame))
    throw std::runtime_error(
        "restart metadata does not match the interrupted output transaction");
  // Validate every offset before changing any file; a short committed history
  // is damage, not an interrupted append that can safely be rolled back.
  for (std::size_t i = 0; i < csvNames.size(); ++i) {
    const auto path = p.outputDirectory / csvNames[i];
    if (j.csvExisted[i] && (!std::filesystem::exists(path) ||
                            std::filesystem::file_size(path) < j.csvSizes[i]))
      throw std::runtime_error("committed CSV data is missing: " +
                               path.string());
  }
  for (std::size_t i = 0; i < csvNames.size(); ++i) {
    const auto path = p.outputDirectory / csvNames[i];
    if (j.csvExisted[i])
      std::filesystem::resize_file(path, j.csvSizes[i]);
    else
      std::filesystem::remove(path);
  }
  const auto files = frameFiles(p, j.frame);
  for (std::size_t i = 0; i < files.size(); ++i) {
    const auto backup = appended(files[i], ".previous");
    if (std::filesystem::exists(backup))
      std::filesystem::rename(backup, files[i]);
    else if (!j.frameExisted[i])
      std::filesystem::remove(files[i]);
    std::filesystem::remove(appended(files[i], ".tmp"));
  }
  std::filesystem::remove(p.dataDirectory / "restart_state.tmp");
  removeJournal(p);
  std::cout << "recovered interrupted output frame " << j.frame << '\n';
  return j.frame == 0 && !committed;
}

void beginOutputTransaction(const Parameters &p, std::uint64_t frame) {
  if (std::filesystem::exists(p.dataDirectory / "output_transaction.txt"))
    throw std::runtime_error("an output transaction is already active");
  Journal j;
  j.frame = frame;
  const auto previous = committedFrame(p);
  j.previousMetadata = previous.has_value();
  j.previousFrame = previous.value_or(0);
  const auto files = frameFiles(p, frame);
  for (std::size_t i = 0; i < files.size(); ++i) {
    j.frameExisted[i] = std::filesystem::exists(files[i]);
    if (j.frameExisted[i] &&
        (!p.overwriteOutput || !std::filesystem::is_regular_file(files[i])))
      throw std::runtime_error("refusing to overwrite output frame file: " +
                               files[i].string());
    for (const char *suffix : {".tmp", ".previous"})
      if (std::filesystem::exists(appended(files[i], suffix)))
        throw std::runtime_error("untracked temporary output file: " +
                                 appended(files[i], suffix).string());
  }
  for (std::size_t i = 0; i < csvNames.size(); ++i) {
    const auto path = p.outputDirectory / csvNames[i];
    j.csvExisted[i] = std::filesystem::exists(path);
    if (j.csvExisted[i])
      j.csvSizes[i] = std::filesystem::file_size(path);
  }
  const auto temporary = p.dataDirectory / "output_transaction.tmp";
  std::ofstream out(temporary);
  out << "ns2d_output_transaction_v1\n"
      << std::quoted(std::filesystem::canonical(p.outputDirectory).string())
      << '\n'
      << j.frame << ' ' << j.previousMetadata << ' ' << j.previousFrame << '\n';
  for (std::size_t i = 0; i < csvNames.size(); ++i)
    out << j.csvExisted[i] << ' ' << j.csvSizes[i] << '\n';
  for (const bool existed : j.frameExisted)
    out << existed << '\n';
  out.close();
  if (!out)
    throw std::runtime_error("cannot write output transaction journal");
  std::filesystem::rename(temporary,
                          p.dataDirectory / "output_transaction.txt");
  for (std::size_t i = 0; i < files.size(); ++i)
    if (j.frameExisted[i])
      std::filesystem::rename(files[i], appended(files[i], ".previous"));
}

void finishOutputTransaction(const Parameters &p) {
  const Journal j = readJournal(p);
  if (committedFrame(p) != std::optional{j.frame})
    throw std::runtime_error("cannot finish an uncommitted output transaction");
  for (const auto &path : frameFiles(p, j.frame)) {
    std::filesystem::remove(appended(path, ".previous"));
    std::filesystem::remove(appended(path, ".tmp"));
  }
  removeJournal(p);
}

void writeRunRecords(const Parameters &p, const std::string &backend,
                     double time, std::uint64_t frame,
                     const std::vector<double> &amplitude,
                     std::size_t forcedModes, double energyCoefficient,
                     double enstrophyCoefficient) {
  const auto history = p.outputDirectory / "segments";
  constexpr std::array names{"resolved_parameters.txt", "forcing_summary.csv",
                             "forcing_spectrum.csv"};
  if (!std::filesystem::exists(history)) {
    std::filesystem::create_directories(history);
    // Preserve records from solver versions that predate segment histories.
    for (const char *name : names)
      if (std::filesystem::exists(p.outputDirectory / name)) {
        std::filesystem::create_directories(history / "imported");
        std::filesystem::copy_file(p.outputDirectory / name,
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
  writeParameterRecord(p, backend, segment);
  Parameters recordParameters = p;
  recordParameters.outputDirectory = segment;
  writeForcingFiles(recordParameters, amplitude, forcedModes, energyCoefficient,
                    enstrophyCoefficient);
  std::ofstream manifest(segment / "segment.txt");
  manifest << std::setprecision(17) << "startTime " << time << "\nstartFrame "
           << frame << "\nstochasticUpdate exact_linear_covariance_v1\n";
  manifest.close();
  if (!manifest)
    throw std::runtime_error("cannot write run segment record");
  for (const char *name : names) {
    const auto temporary = p.outputDirectory / (std::string(name) + ".tmp");
    std::filesystem::copy_file(
        segment / name, temporary,
        std::filesystem::copy_options::overwrite_existing);
    std::filesystem::rename(temporary, p.outputDirectory / name);
  }
}
