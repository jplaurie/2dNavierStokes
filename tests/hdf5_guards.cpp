#include "hdf5_io.hpp"

#include <hdf5.h>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void require(bool valid, const char *message) {
    if (!valid)
        throw std::runtime_error(message);
}

struct TemporaryDirectory {
    std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        ("ns2d_hdf5_guards_" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    TemporaryDirectory() { std::filesystem::create_directory(path); }
    ~TemporaryDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(path, ignored);
    }
};

void replaceAttribute(const std::filesystem::path &path, const char *name, hid_t type, hid_t space,
                      const void *values) {
    const hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    require(file >= 0, "cannot open HDF5 fixture");
    require(H5Adelete(file, name) >= 0, "cannot remove fixture attribute");
    const hid_t attribute = H5Acreate2(file, name, type, space, H5P_DEFAULT, H5P_DEFAULT);
    require(attribute >= 0, "cannot create fixture attribute");
    require(H5Awrite(attribute, type, values) >= 0, "cannot write fixture attribute");
    require(H5Aclose(attribute) >= 0 && H5Fclose(file) >= 0, "cannot close HDF5 fixture");
}

void requireRejected(const std::filesystem::path &path, const std::string &message) {
    try {
        (void)readHdf5Field(path);
    } catch (const std::runtime_error &error) {
        if (std::string(error.what()).find(message) != std::string::npos)
            return;
        throw;
    }
    throw std::runtime_error("malformed HDF5 metadata was accepted");
}
} // namespace

int main() {
    try {
        TemporaryDirectory temporary;
        Parameters parameters;
        parameters.nx = parameters.ny = 4;
        const auto reference = temporary.path / "reference.h5";
        writeHdf5Field(reference, parameters, 0.0, 0, std::vector<double>(16, 1.0));
        const auto fixture = temporary.path / "fixture.h5";
        const auto reset = [&] {
            std::filesystem::copy_file(reference, fixture,
                                       std::filesystem::copy_options::overwrite_existing);
        };

        const hsize_t two = 2;
        const hid_t arraySpace = H5Screate_simple(1, &two, nullptr);
        const hid_t scalarSpace = H5Screate(H5S_SCALAR);
        require(arraySpace >= 0 && scalarSpace >= 0, "cannot create fixture attribute spaces");
        const unsigned long long integers[] = {4, 4};
        const double reals[] = {1.0, 1.0};
        for (const char *name : {"nx", "ny", "frame"}) {
            reset();
            replaceAttribute(fixture, name, H5T_NATIVE_ULLONG, arraySpace, integers);
            requireRejected(fixture, "must be scalar");
        }
        for (const char *name : {"length_x", "length_y", "time"}) {
            reset();
            replaceAttribute(fixture, name, H5T_NATIVE_DOUBLE, arraySpace, reals);
            requireRejected(fixture, "must be scalar");
        }

        const hid_t stringType = H5Tcopy(H5T_C_S1);
        require(stringType >= 0 && H5Tset_size(stringType, 32) >= 0,
                "cannot create fixture string type");
        const char formats[2][32] = {"ns2d_vorticity_v1", "ns2d_vorticity_v1"};
        reset();
        replaceAttribute(fixture, "format", stringType, arraySpace, formats);
        requireRejected(fixture, "must be scalar");
        reset();
        replaceAttribute(fixture, "nx", stringType, scalarSpace, formats);
        requireRejected(fixture, "numeric attribute type");
        reset();
        replaceAttribute(fixture, "format", H5T_NATIVE_ULLONG, scalarSpace, integers);
        requireRejected(fixture, "string attribute type");

        require(H5Tset_size(stringType, H5T_VARIABLE) >= 0, "cannot create variable string type");
        const char *format = "ns2d_vorticity_v1";
        reset();
        replaceAttribute(fixture, "format", stringType, scalarSpace, &format);
        require(readHdf5Field(fixture).vorticity == std::vector<double>(16, 1.0),
                "variable-length scalar string was not read correctly");
        require(H5Tclose(stringType) >= 0 && H5Sclose(arraySpace) >= 0 &&
                    H5Sclose(scalarSpace) >= 0,
                "cannot close fixture attribute types/spaces");
        std::cout << "HDF5 metadata guards passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
