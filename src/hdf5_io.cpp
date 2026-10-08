#include "hdf5_io.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

#ifdef NS2D_HAVE_HDF5
#include <hdf5.h>

namespace {
template <herr_t (*Close)(hid_t)> class Hdf5Handle {
  public:
    Hdf5Handle() = default;
    explicit Hdf5Handle(hid_t id) : id_(id) {}
    ~Hdf5Handle() {
        if (id_ >= 0)
            Close(id_);
    }
    Hdf5Handle(const Hdf5Handle &) = delete;
    Hdf5Handle &operator=(const Hdf5Handle &) = delete;
    Hdf5Handle(Hdf5Handle &&other) noexcept : id_(other.id_) { other.id_ = -1; }
    Hdf5Handle &operator=(Hdf5Handle &&other) noexcept {
        if (this != &other) {
            if (id_ >= 0)
                Close(id_);
            id_ = other.id_;
            other.id_ = -1;
        }
        return *this;
    }
    [[nodiscard]] hid_t get() const { return id_; }

  private:
    hid_t id_ = -1;
};

using File = Hdf5Handle<H5Fclose>;
using DataSet = Hdf5Handle<H5Dclose>;
using DataSpace = Hdf5Handle<H5Sclose>;
using PropertyList = Hdf5Handle<H5Pclose>;
using Attribute = Hdf5Handle<H5Aclose>;
using DataType = Hdf5Handle<H5Tclose>;

hid_t requireId(hid_t id, const std::string &operation) {
    if (id < 0)
        throw std::runtime_error(operation);
    return id;
}

void requireSuccess(herr_t status, const std::string &operation) {
    if (status < 0)
        throw std::runtime_error(operation);
}

template <class T>
void writeScalarAttribute(hid_t object, const char *name, hid_t type, const T &value) {
    DataSpace space(requireId(H5Screate(H5S_SCALAR), "cannot create HDF5 attribute space"));
    Attribute attribute(
        requireId(H5Acreate2(object, name, type, space.get(), H5P_DEFAULT, H5P_DEFAULT),
                  std::string("cannot create HDF5 attribute: ") + name));
    requireSuccess(H5Awrite(attribute.get(), type, &value),
                   std::string("cannot write HDF5 attribute: ") + name);
}

void writeStringAttribute(hid_t object, const char *name, const char *value) {
    DataType type(requireId(H5Tcopy(H5T_C_S1), "cannot create HDF5 string type"));
    requireSuccess(H5Tset_size(type.get(), std::strlen(value) + 1), "cannot size HDF5 string type");
    requireSuccess(H5Tset_strpad(type.get(), H5T_STR_NULLTERM),
                   "cannot configure HDF5 string type");
    DataSpace space(requireId(H5Screate(H5S_SCALAR), "cannot create HDF5 attribute space"));
    Attribute attribute(
        requireId(H5Acreate2(object, name, type.get(), space.get(), H5P_DEFAULT, H5P_DEFAULT),
                  std::string("cannot create HDF5 attribute: ") + name));
    requireSuccess(H5Awrite(attribute.get(), type.get(), value),
                   std::string("cannot write HDF5 attribute: ") + name);
}

template <class T> T readScalarAttribute(hid_t object, const char *name, hid_t type) {
    Attribute attribute(requireId(H5Aopen(object, name, H5P_DEFAULT),
                                  std::string("missing HDF5 attribute: ") + name));
    T value{};
    requireSuccess(H5Aread(attribute.get(), type, &value),
                   std::string("cannot read HDF5 attribute: ") + name);
    return value;
}

std::string readStringAttribute(hid_t object, const char *name) {
    Attribute attribute(requireId(H5Aopen(object, name, H5P_DEFAULT),
                                  std::string("missing HDF5 attribute: ") + name));
    DataType type(requireId(H5Aget_type(attribute.get()), "cannot inspect HDF5 string attribute"));
    const std::size_t size = H5Tget_size(type.get());
    if (size == 0 || size > 1024)
        throw std::runtime_error("invalid HDF5 string attribute size");
    std::vector<char> value(size + 1, '\0');
    requireSuccess(H5Aread(attribute.get(), type.get(), value.data()),
                   std::string("cannot read HDF5 attribute: ") + name);
    return value.data();
}
} // namespace
#endif

bool hdf5Available() {
#ifdef NS2D_HAVE_HDF5
    return true;
#else
    return false;
#endif
}

void writeHdf5Field(const std::filesystem::path &path, const Parameters &parameters, double time,
                    std::uint64_t frame, const std::vector<double> &physical) {
#ifdef NS2D_HAVE_HDF5
    if (physical.size() != parameters.nx * parameters.ny)
        throw std::runtime_error("invalid HDF5 vorticity size");
    if (!std::all_of(physical.begin(), physical.end(),
                     [](double value) { return std::isfinite(value); }))
        throw std::runtime_error("refusing to write non-finite HDF5 vorticity");
    File file(requireId(H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT),
                        "cannot create HDF5 field file: " + path.string()));
    writeStringAttribute(file.get(), "format", "ns2d_vorticity_v1");
    const auto nx = static_cast<unsigned long long>(parameters.nx);
    const auto ny = static_cast<unsigned long long>(parameters.ny);
    const auto savedFrame = static_cast<unsigned long long>(frame);
    writeScalarAttribute(file.get(), "nx", H5T_NATIVE_ULLONG, nx);
    writeScalarAttribute(file.get(), "ny", H5T_NATIVE_ULLONG, ny);
    writeScalarAttribute(file.get(), "length_x", H5T_NATIVE_DOUBLE, parameters.lx());
    writeScalarAttribute(file.get(), "length_y", H5T_NATIVE_DOUBLE, parameters.ly());
    writeScalarAttribute(file.get(), "time", H5T_NATIVE_DOUBLE, time);
    writeScalarAttribute(file.get(), "frame", H5T_NATIVE_ULLONG, savedFrame);

    const hsize_t dimensions[2] = {parameters.ny, parameters.nx};
    DataSpace space(
        requireId(H5Screate_simple(2, dimensions, nullptr), "cannot create HDF5 vorticity space"));
    PropertyList properties;
    hid_t creationProperties = H5P_DEFAULT;
    if (parameters.hdf5CompressionLevel > 0) {
        properties = PropertyList(
            requireId(H5Pcreate(H5P_DATASET_CREATE), "cannot create HDF5 dataset properties"));
        const hsize_t chunks[2] = {std::min<hsize_t>(parameters.ny, 64),
                                   std::min<hsize_t>(parameters.nx, 64)};
        requireSuccess(H5Pset_chunk(properties.get(), 2, chunks),
                       "cannot set HDF5 vorticity chunks");
        requireSuccess(H5Pset_deflate(properties.get(), parameters.hdf5CompressionLevel),
                       "cannot enable HDF5 deflate compression");
        creationProperties = properties.get();
    }
    DataSet dataset(requireId(H5Dcreate2(file.get(), "/vorticity", H5T_IEEE_F64LE, space.get(),
                                         H5P_DEFAULT, creationProperties, H5P_DEFAULT),
                              "cannot create HDF5 vorticity dataset"));
    requireSuccess(
        H5Dwrite(dataset.get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, physical.data()),
        "cannot write HDF5 vorticity dataset");
#else
    (void)path;
    (void)parameters;
    (void)time;
    (void)frame;
    (void)physical;
    throw std::runtime_error("this build has no HDF5 support");
#endif
}

Hdf5Field readHdf5Field(const std::filesystem::path &path) {
#ifdef NS2D_HAVE_HDF5
    File file(requireId(H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT),
                        "cannot open HDF5 field file: " + path.string()));
    if (readStringAttribute(file.get(), "format") != "ns2d_vorticity_v1")
        throw std::runtime_error("unsupported HDF5 vorticity format: " + path.string());
    DataSet dataset(requireId(H5Dopen2(file.get(), "/vorticity", H5P_DEFAULT),
                              "HDF5 file has no /vorticity dataset: " + path.string()));
    DataSpace space(requireId(H5Dget_space(dataset.get()), "cannot inspect HDF5 vorticity"));
    hsize_t dimensions[2]{};
    if (H5Sget_simple_extent_ndims(space.get()) != 2 ||
        H5Sget_simple_extent_dims(space.get(), dimensions, nullptr) < 0)
        throw std::runtime_error("HDF5 vorticity must have shape [ny,nx]");
    constexpr std::size_t maximumSize = std::numeric_limits<std::size_t>::max();
    if (dimensions[0] == 0 || dimensions[1] == 0 || dimensions[0] > maximumSize ||
        dimensions[1] > maximumSize / static_cast<std::size_t>(dimensions[0]))
        throw std::runtime_error("HDF5 vorticity dimensions exceed addressable memory");
    Hdf5Field result;
    result.ny = static_cast<std::size_t>(dimensions[0]);
    result.nx = static_cast<std::size_t>(dimensions[1]);
    const auto savedNx =
        readScalarAttribute<unsigned long long>(file.get(), "nx", H5T_NATIVE_ULLONG);
    const auto savedNy =
        readScalarAttribute<unsigned long long>(file.get(), "ny", H5T_NATIVE_ULLONG);
    if (savedNx != result.nx || savedNy != result.ny)
        throw std::runtime_error("HDF5 dimensions disagree with file metadata");
    result.lengthX = readScalarAttribute<double>(file.get(), "length_x", H5T_NATIVE_DOUBLE);
    result.lengthY = readScalarAttribute<double>(file.get(), "length_y", H5T_NATIVE_DOUBLE);
    result.time = readScalarAttribute<double>(file.get(), "time", H5T_NATIVE_DOUBLE);
    result.frame = static_cast<std::uint64_t>(
        readScalarAttribute<unsigned long long>(file.get(), "frame", H5T_NATIVE_ULLONG));
    if (!(result.lengthX > 0.0) || !(result.lengthY > 0.0) || !std::isfinite(result.lengthX) ||
        !std::isfinite(result.lengthY) || !std::isfinite(result.time))
        throw std::runtime_error("HDF5 vorticity metadata is invalid");
    result.vorticity.resize(result.nx * result.ny);
    requireSuccess(H5Dread(dataset.get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                           result.vorticity.data()),
                   "cannot read HDF5 vorticity dataset");
    if (!std::all_of(result.vorticity.begin(), result.vorticity.end(),
                     [](double value) { return std::isfinite(value); }))
        throw std::runtime_error("HDF5 vorticity contains non-finite values");
    return result;
#else
    (void)path;
    throw std::runtime_error("this build has no HDF5 support");
#endif
}
