# Diagnostic tools, not registered with ctest: device_smoke needs a GPU and
# must not run as part of the normal test suite. See README.md for how to
# run it under srun.
add_executable(hypre_device_smoke device_smoke.cpp)
target_include_directories(hypre_device_smoke PRIVATE "${PROJECT_SOURCE_DIR}")
target_link_libraries(
    hypre_device_smoke
    PRIVATE GinkgoTpl::hypre Ginkgo::ginkgo MPI::MPI_CXX
)
# Exports symbols so the crash handler's backtrace carries function names
# rather than bare addresses.
target_link_options(hypre_device_smoke PRIVATE -rdynamic)
