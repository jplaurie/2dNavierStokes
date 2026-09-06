#include "backend.hpp"
#include "parameters.hpp"
#include "solver.hpp"

#include <exception>
#include <filesystem>
#include <iostream>

int main(int argc, char **argv) {
  bool backendReady = false;
  try {
    backendInitialize(argc, argv);
    backendReady = true;
    {
      const std::filesystem::path parameterFile =
          argc > 1 ? argv[1] : "params.txt";
      Parameters parameters = readParameters(parameterFile);
      auto backend = makeBackend(parameters);
      Solver solver(std::move(parameters), std::move(backend));
      solver.run();
    } // Destroy every FFT plan before cleaning up its FFT runtime.
    backendFinalize();
    return 0;
  } catch (const std::exception &error) {
    if (!backendReady || backendIsRoot())
      std::cerr << "error: " << error.what() << '\n';
    if (backendReady) {
      backendAbort(1);
      backendFinalize();
    }
    return 1;
  }
}
