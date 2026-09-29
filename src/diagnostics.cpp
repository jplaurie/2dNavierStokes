#include "output.hpp"
#include "spectral.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace {
std::ofstream numericOutput(const std::filesystem::path &path,
                            std::ios::openmode mode = std::ios::out) {
    std::ofstream output(path, mode);
    if (!output)
        throw std::runtime_error("cannot write output file: " + path.string());
    output << std::scientific << std::setprecision(12);
    return output;
}

double waveNumber(const Parameters &parameters, std::size_t x, std::size_t y) {
    return std::hypot(waveNumberX(parameters, x), waveNumberY(parameters, y));
}
} // namespace

double writeDiagnostics(const Parameters &parameters, double time, std::uint64_t frame,
                        const SpectralField &vorticity, const SpectralField &nonlinearTerm,
                        DiagnosticsAverages &averages) {
    const std::size_t bins = parameters.spectrumBins();
    if (averages.energySpectrum.empty()) {
        averages.energySpectrum.assign(bins, 0.0);
        averages.enstrophySpectrum.assign(bins, 0.0);
        averages.energyFlux.assign(bins, 0.0);
        averages.enstrophyFlux.assign(bins, 0.0);
    }
    std::vector<double> energySpectrum(bins), enstrophySpectrum(bins), energyShell(bins),
        enstrophyShell(bins);
    double energy = 0.0, enstrophy = 0.0;
    double energyViscosity = 0.0, energyDrag = 0.0;
    double enstrophyViscosity = 0.0, enstrophyDrag = 0.0;
    const double binWidth = std::min(2.0 * nsPi / parameters.lx(), 2.0 * nsPi / parameters.ly());

    for (std::size_t y = 0; y < parameters.ny; ++y) {
        for (std::size_t x = 0; x < parameters.nxf(); ++x) {
            const double k = waveNumber(parameters, x, y);
            const double multiplicity = (x == 0 || x == parameters.nx / 2) ? 1.0 : 2.0;
            const std::size_t index = spectralIndex(x, y, parameters.nxf());
            const double w2 = std::norm(vorticity[index]);
            if (k > 0.0) {
                energy += 0.5 * multiplicity * w2 / (k * k);
                enstrophy += 0.5 * multiplicity * w2;
                const double k2 = k * k;
                double viscousDissipation = 0.0;
                double dragDissipation = 0.0;
                if (parameters.viscosity > 0.0)
                    viscousDissipation = multiplicity * parameters.viscosity *
                                         std::pow(k2, parameters.viscosityOrder) * w2;
                if (parameters.linearDrag > 0.0)
                    dragDissipation = multiplicity * parameters.linearDrag *
                                      std::pow(k2, parameters.dragOrder) * w2;
                energyViscosity += viscousDissipation / k2;
                energyDrag += dragDissipation / k2;
                enstrophyViscosity += viscousDissipation;
                enstrophyDrag += dragDissipation;
            }

            const std::size_t bin = static_cast<std::size_t>(k / binWidth);
            if (bin >= bins)
                throw std::logic_error("spectrum bin count is too small");
            const double halfMultiplicity = 0.5 * multiplicity;
            enstrophySpectrum[bin] += halfMultiplicity * w2;
            if (k > 0.0)
                energySpectrum[bin] += halfMultiplicity * w2 / (k * k);
            const double transfer =
                multiplicity * std::real(std::conj(vorticity[index]) * nonlinearTerm[index]);
            enstrophyShell[bin] += transfer;
            if (k > 0.0)
                energyShell[bin] += transfer / (k * k);
        }
    }

    std::vector<double> energyFlux(bins), enstrophyFlux(bins);
    double cumulativeEnergy = 0.0, cumulativeEnstrophy = 0.0;
    for (std::size_t i = bins; i-- > 0;) {
        cumulativeEnergy += energyShell[i];
        cumulativeEnstrophy += enstrophyShell[i];
        energyFlux[i] = cumulativeEnergy;
        enstrophyFlux[i] = cumulativeEnstrophy;
    }
    const auto finiteValues = [](const std::vector<double> &values) {
        return std::all_of(values.begin(), values.end(),
                           [](double value) { return std::isfinite(value); });
    };
    if (!std::isfinite(energy) || !std::isfinite(enstrophy) || !std::isfinite(energyViscosity) ||
        !std::isfinite(energyDrag) || !std::isfinite(enstrophyViscosity) ||
        !std::isfinite(enstrophyDrag) || !finiteValues(energySpectrum) ||
        !finiteValues(enstrophySpectrum) || !finiteValues(energyFlux) ||
        !finiteValues(enstrophyFlux))
        throw std::runtime_error(
            "diagnostics became non-finite; reduce the time step or coefficients");
    ++averages.count;
    for (std::size_t i = 0; i < bins; ++i) {
        averages.energySpectrum[i] += energySpectrum[i];
        averages.enstrophySpectrum[i] += enstrophySpectrum[i];
        averages.energyFlux[i] += energyFlux[i];
        averages.enstrophyFlux[i] += enstrophyFlux[i];
    }

    auto diagnostics = numericOutput(parameters.outputDirectory / "diagnostics.csv",
                                     std::ios::out | std::ios::app);
    diagnostics << time << ',' << frame << ',' << energy << ',' << enstrophy << ',' << energyDrag
                << ',' << energyViscosity << ',' << enstrophyDrag << ',' << enstrophyViscosity
                << '\n';
    diagnostics.close();
    if (!diagnostics)
        throw std::runtime_error("failed while writing diagnostics.csv");
    auto spectra =
        numericOutput(parameters.outputDirectory / "spectra.csv", std::ios::out | std::ios::app);
    auto fluxes =
        numericOutput(parameters.outputDirectory / "fluxes.csv", std::ios::out | std::ios::app);
    for (std::size_t i = 0; i < bins; ++i) {
        const double shellWavenumber = static_cast<double>(i) * binWidth;
        spectra << time << ',' << frame << ',' << shellWavenumber << ',' << energySpectrum[i] << ','
                << enstrophySpectrum[i] << ','
                << averages.energySpectrum[i] / static_cast<double>(averages.count) << ','
                << averages.enstrophySpectrum[i] / static_cast<double>(averages.count) << '\n';
        fluxes << time << ',' << frame << ',' << shellWavenumber << ',' << energyFlux[i] << ','
               << enstrophyFlux[i] << ','
               << averages.energyFlux[i] / static_cast<double>(averages.count) << ','
               << averages.enstrophyFlux[i] / static_cast<double>(averages.count) << '\n';
    }
    spectra.close();
    fluxes.close();
    if (!spectra || !fluxes)
        throw std::runtime_error("failed while writing spectra or fluxes CSV");
    if (parameters.writeModeDiagnostics) {
        auto modes =
            numericOutput(parameters.outputDirectory / "modes.csv", std::ios::out | std::ios::app);
        modes << time << ',' << frame;
        for (const auto [x, y] : {std::pair{1UL, 0UL}, std::pair{0UL, 1UL}, std::pair{1UL, 1UL},
                                  std::pair{2UL, 1UL}, std::pair{0UL, 3UL}}) {
            if (x >= parameters.nx / 2 || y >= parameters.ny / 2) {
                modes << ",,"; // This positive wave is outside the retained subspace.
            } else {
                const Complex value = vorticity[spectralIndex(x, y, parameters.nxf())];
                modes << ',' << value.real() << ',' << value.imag();
            }
        }
        modes << '\n';
        modes.close();
        if (!modes)
            throw std::runtime_error("failed while writing modes.csv");
    }
    return energy;
}

void writeForcingFiles(const Parameters &parameters, const std::vector<double> &amplitude,
                       std::size_t forcedModes, double energyInjectionCoefficient,
                       double enstrophyInjectionCoefficient) {
    {
        std::ofstream out(parameters.outputDirectory / "forcing_summary.csv", std::ios::trunc);
        if (!out)
            throw std::runtime_error("cannot write forcing_summary.csv");
        out << "enabled,profile,temporal_type,forced_modes,energy_injection_"
               "coefficient,"
               "enstrophy_injection_coefficient\n"
            << std::boolalpha << parameters.forcingEnabled << ','
            << forcingProfileName(parameters.forcingProfile) << ',';
        if (!parameters.forcingEnabled)
            out << "disabled";
        else if (parameters.forcingProfile == ForcingProfile::singleMode)
            out << "deterministic";
        else
            out << "stochastic";
        out << ',' << forcedModes << ',';
        if (parameters.forcingEnabled && parameters.forcingProfile != ForcingProfile::singleMode)
            out << std::setprecision(17) << energyInjectionCoefficient << ','
                << enstrophyInjectionCoefficient;
        else
            out << ',';
        out << '\n';
        out.close();
        if (!out)
            throw std::runtime_error("failed while writing forcing_summary.csv");
    }
    auto out = numericOutput(parameters.outputDirectory / "forcing_spectrum.csv");
    out << "kx,ky,amplitude,multiplicity\n";
    if (parameters.forcingEnabled) {
        for (std::size_t x = 0; x < parameters.nxf(); ++x) {
            const double kx = waveNumberX(parameters, x);
            const int multiplicity = (x == 0 || x == parameters.nx / 2) ? 1 : 2;
            for (std::size_t y = 0; y < parameters.ny; ++y) {
                const double ky = waveNumberY(parameters, y);
                out << kx << ',' << ky << ',' << amplitude[spectralIndex(x, y, parameters.nxf())]
                    << ',' << multiplicity << '\n';
            }
        }
    }
    out.close();
    if (!out)
        throw std::runtime_error("failed while writing forcing_spectrum.csv");
}
