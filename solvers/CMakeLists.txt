project(tpl_bench)
cmake_minimum_required(VERSION 3.24)
find_package(Ginkgo REQUIRED)
if (GINKGO_SOLVERS_BUILD_TRILINOS)
    find_package(Trilinos REQUIRED COMPONENTS Tpetra Belos)

    # Echo trilinos build info just for fun
    MESSAGE("\nFound Trilinos!  Here are the details: ")
    MESSAGE("   Trilinos_DIR = ${Trilinos_DIR}")
    MESSAGE("   Trilinos_VERSION = ${Trilinos_VERSION}")
    MESSAGE("   Trilinos_PACKAGE_LIST = ${Trilinos_PACKAGE_LIST}")
    MESSAGE("   Trilinos_LIBRARIES = ${Trilinos_LIBRARIES}")
    MESSAGE("   Trilinos_INCLUDE_DIRS = ${Trilinos_INCLUDE_DIRS}")
    MESSAGE("   Trilinos_TPL_LIST = ${Trilinos_TPL_LIST}")
    MESSAGE("   Trilinos_TPL_LIBRARIES = ${Trilinos_TPL_LIBRARIES}")
    MESSAGE("   Trilinos_BUILD_SHARED_LIBS = ${Trilinos_BUILD_SHARED_LIBS}")
    MESSAGE("End of Trilinos details\n")

    add_library(solver-wrapper-trilinos trilinos_wrapper.cpp)
    target_link_libraries(solver-wrapper-trilinos Ginkgo::ginkgo Trilinos::all_selected_libs)
endif()

if (GINKGO_SOLVERS_BUILD_HYPRE)
    find_package(HYPRE REQUIRED)

    add_library(solver-wrapper-hypre hypre_wrapper.cpp)
    target_link_libraries(solver-wrapper-hypre Ginkgo::ginkgo HYPRE::HYPRE)
endif()
