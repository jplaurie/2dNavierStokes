#include "hdf5_io.hpp"

#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void writeField(std::ostream &output, const Hdf5Field &field) {
    for (std::size_t y = 0; y < field.ny; ++y)
        for (std::size_t x = 0; x < field.nx; ++x)
            output << field.vorticity[y * field.nx + x] << (x + 1 == field.nx ? '\n' : ' ');
}

void writeGnuplot(std::ostream &output, const Hdf5Field &field) {
    output << "# x y vorticity\n";
    for (std::size_t y = 0; y < field.ny; ++y) {
        const double physicalY = field.lengthY * static_cast<double>(y) / field.ny;
        for (std::size_t x = 0; x < field.nx; ++x) {
            const double physicalX = field.lengthX * static_cast<double>(x) / field.nx;
            output << physicalX << ' ' << physicalY << ' ' << field.vorticity[y * field.nx + x]
                   << '\n';
        }
        output << '\n';
    }
}
} // namespace

int main(int argc, char **argv) {
    try {
        if (argc < 3 || argc > 5)
            throw std::runtime_error(
                "usage: ns2d_hdf5_export INPUT.h5 OUTPUT [--format field|gnuplot]");
        std::string format = "field";
        if (argc == 5) {
            if (std::string(argv[3]) != "--format")
                throw std::runtime_error("expected --format before output format");
            format = argv[4];
        }
        if (format != "field" && format != "gnuplot")
            throw std::runtime_error("export format must be field or gnuplot");
        const Hdf5Field field = readHdf5Field(argv[1]);
        std::ofstream file;
        std::ostream *output = &std::cout;
        if (std::string(argv[2]) != "-") {
            file.open(argv[2]);
            if (!file)
                throw std::runtime_error("cannot create export file: " + std::string(argv[2]));
            output = &file;
        }
        *output << std::scientific << std::setprecision(12);
        if (format == "field")
            writeField(*output, field);
        else
            writeGnuplot(*output, field);
        if (file.is_open()) {
            file.close();
            if (!file)
                throw std::runtime_error("failed while writing HDF5 export");
        }
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
