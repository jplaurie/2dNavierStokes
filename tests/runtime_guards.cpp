#include "host_stepper.hpp"
#include "output.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
class TemporaryDirectory {
  public:
    TemporaryDirectory() {
        const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
        path_ = std::filesystem::temp_directory_path() /
                ("ns2d_runtime_guard_tests_" + std::to_string(suffix));
        std::filesystem::create_directory(path_);
    }

    ~TemporaryDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }

    [[nodiscard]] const std::filesystem::path &path() const { return path_; }

  private:
    std::filesystem::path path_;
};

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
    throw std::runtime_error("invalid runtime operation was accepted");
}

void checkWorkspaceReinitialization() {
    HostIntegrationWorkspace workspace;
    workspace.initialize(8, 4);
    workspace.initialize(3, 2);
    require(workspace.nonlinearStages[0].size() == 3 && workspace.nonlinearStages[1].size() == 3,
            "active nonlinear workspaces have the wrong size");
    require(workspace.nonlinearStages[2].empty() && workspace.nonlinearStages[3].empty(),
            "inactive nonlinear workspaces were retained");
    require(workspace.stageStates[0].size() == 3 && workspace.stageStates[1].empty() &&
                workspace.stageStates[2].empty(),
            "inactive stage workspaces were retained");
    requireFailure([&] { workspace.initialize(3, 1); }, "invalid host integration stage count");
}

void checkTransactionOrdering(const std::filesystem::path &root) {
    Parameters parameters;
    parameters.dataDirectory = root / "data";
    parameters.outputDirectory = root / "output";
    std::filesystem::create_directory(parameters.dataDirectory);
    std::filesystem::create_directory(parameters.outputDirectory);
    std::ofstream metadata(parameters.dataDirectory / "restart_state.txt");
    metadata << "ns2d_restart_v3\ntime 0\nframe 4\n";
    metadata.close();
    require(static_cast<bool>(metadata), "could not create restart metadata fixture");
    requireFailure([&] { beginOutputTransaction(parameters, 4); },
                   "output transaction frame must follow the committed frame");
    requireFailure([&] { beginOutputTransaction(parameters, 3); },
                   "output transaction frame must follow the committed frame");
    require(!std::filesystem::exists(parameters.dataDirectory / "output_transaction.txt"),
            "rejected transaction created a journal");
}
} // namespace

int main() {
    try {
        TemporaryDirectory temporary;
        checkWorkspaceReinitialization();
        checkTransactionOrdering(temporary.path());
        std::cout << "runtime guard tests passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
