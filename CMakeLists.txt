cmake_minimum_required(VERSION 3.21)
project(GinkgoTplWrappers VERSION 0.1.0 LANGUAGES CXX)

# One option per wrapped library. Each component is an independent target,
# GinkgoTpl::<component>, and only looks for its own third-party library.
option(GKO_TPL_BUILD_PETSC "Build the PETSc wrappers (GinkgoTpl::petsc)" ON)
option(GKO_TPL_BUILD_TESTS "Build the tests" ${PROJECT_IS_TOP_LEVEL})
option(
    GKO_TPL_INSTALL
    "Generate install rules and the package config"
    ${PROJECT_IS_TOP_LEVEL}
)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(GNUInstallDirs)
set(GKO_TPL_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/GinkgoTplWrappers")

# A project that already provides Ginkgo, e.g. Ginkgo's own examples adding
# this repository with add_subdirectory or FetchContent, keeps its Ginkgo:
# looking for an installed one as well would link two Ginkgo builds into one
# executable.
if(NOT TARGET Ginkgo::ginkgo)
    find_package(Ginkgo REQUIRED)
endif()
find_package(MPI REQUIRED COMPONENTS CXX)

if(GKO_TPL_BUILD_TESTS)
    enable_testing()
    include(gko_tpl_testing)
endif()

set(GKO_TPL_COMPONENTS "")
if(GKO_TPL_BUILD_PETSC)
    add_subdirectory(petsc)
    list(APPEND GKO_TPL_COMPONENTS petsc)
endif()

if(GKO_TPL_INSTALL)
    include(CMakePackageConfigHelpers)
    configure_package_config_file(
        cmake/GinkgoTplWrappersConfig.cmake.in
        "${PROJECT_BINARY_DIR}/GinkgoTplWrappersConfig.cmake"
        INSTALL_DESTINATION "${GKO_TPL_INSTALL_CMAKEDIR}"
    )
    write_basic_package_version_file(
        "${PROJECT_BINARY_DIR}/GinkgoTplWrappersConfigVersion.cmake"
        COMPATIBILITY SameMinorVersion
    )
    install(
        FILES
            "${PROJECT_BINARY_DIR}/GinkgoTplWrappersConfig.cmake"
            "${PROJECT_BINARY_DIR}/GinkgoTplWrappersConfigVersion.cmake"
        DESTINATION "${GKO_TPL_INSTALL_CMAKEDIR}"
    )
endif()
