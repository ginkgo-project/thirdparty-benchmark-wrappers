include(gko_tpl_find_petsc)
# Built on its own, a missing PETSc is an error. As a subproject, e.g. fetched
# by an example that also runs without PETSc, it only skips this component;
# the consumer checks if(TARGET GinkgoTpl::petsc).
if(PROJECT_IS_TOP_LEVEL)
    gko_tpl_find_petsc(REQUIRED)
else()
    gko_tpl_find_petsc()
    if(NOT PETSC_FOUND)
        message(
            STATUS
            "GinkgoTpl::petsc: PETSc not found through pkg-config (set PETSC_DIR), skipping it"
        )
        return()
    endif()
endif()
message(
    STATUS
    "GinkgoTpl::petsc: using PETSc ${PETSC_VERSION} from ${PETSC_PREFIX}"
)

# Header-only: nothing here is compiled against Ginkgo, so a consumer's Ginkgo
# is the only one in its executable.
add_library(gko_tpl_petsc INTERFACE)
add_library(GinkgoTpl::petsc ALIAS gko_tpl_petsc)
set_target_properties(gko_tpl_petsc PROPERTIES EXPORT_NAME petsc)
target_include_directories(
    gko_tpl_petsc
    INTERFACE
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)
target_link_libraries(
    gko_tpl_petsc
    INTERFACE Ginkgo::ginkgo PkgConfig::PETSC MPI::MPI_CXX
)
target_compile_features(gko_tpl_petsc INTERFACE cxx_std_17)

if(GKO_TPL_BUILD_TESTS)
    add_subdirectory(test)
endif()

if(GKO_TPL_INSTALL)
    install(DIRECTORY include/ DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
    install(TARGETS gko_tpl_petsc EXPORT GinkgoTplWrappersPetsc)
    install(
        EXPORT GinkgoTplWrappersPetsc
        NAMESPACE GinkgoTpl::
        FILE GinkgoTplWrappers-petsc-targets.cmake
        DESTINATION "${GKO_TPL_INSTALL_CMAKEDIR}"
    )
    # Records the PETSc this package was built against as a search hint.
    configure_file(
        cmake/GinkgoTplWrappers-petsc-dependencies.cmake.in
        "${PROJECT_BINARY_DIR}/GinkgoTplWrappers-petsc-dependencies.cmake"
        @ONLY
    )
    install(
        FILES
            "${PROJECT_BINARY_DIR}/GinkgoTplWrappers-petsc-dependencies.cmake"
            "${PROJECT_SOURCE_DIR}/cmake/gko_tpl_find_petsc.cmake"
        DESTINATION "${GKO_TPL_INSTALL_CMAKEDIR}"
    )
endif()
