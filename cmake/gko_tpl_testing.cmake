# GoogleTest: an installed one if available, otherwise fetched. For an offline
# build, point FETCHCONTENT_SOURCE_DIR_GOOGLETEST at a local source tree.
find_package(GTest QUIET)
if(NOT TARGET GTest::gtest)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL
            https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

# A main that only initializes MPI, for tests that set up everything else
# themselves.
set(GKO_TPL_MPI_TEST_MAIN "${PROJECT_SOURCE_DIR}/common/test/mpi_main.cpp")

# gko_tpl_add_test(<name> SOURCES <src>... [LIBRARIES <lib>...] [MPI_SIZE <n>])
#
# Builds a GoogleTest executable and registers it to run on <n> MPI ranks
# (default 1). The sources must contain a main, e.g. GKO_TPL_MPI_TEST_MAIN.
# Set MPIEXEC_PREFLAGS for launcher flags such as --oversubscribe.
function(gko_tpl_add_test name)
    cmake_parse_arguments(PARSE_ARGV 1 arg "" "MPI_SIZE" "SOURCES;LIBRARIES")
    if(NOT arg_MPI_SIZE)
        set(arg_MPI_SIZE 1)
    endif()
    add_executable(${name} ${arg_SOURCES})
    target_include_directories(${name} PRIVATE "${PROJECT_SOURCE_DIR}")
    target_link_libraries(
        ${name}
        PRIVATE ${arg_LIBRARIES} Ginkgo::ginkgo MPI::MPI_CXX GTest::gtest
    )
    add_test(
        NAME ${name}
        COMMAND
            ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${arg_MPI_SIZE}
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${name}> ${MPIEXEC_POSTFLAGS}
    )
    set_tests_properties(
        ${name}
        PROPERTIES PROCESSORS ${arg_MPI_SIZE} TIMEOUT 300
    )
endfunction()
