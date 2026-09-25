include(gko_tpl_find_hypre)
# Built on its own, a missing hypre is an error. As a subproject, e.g. fetched
# by an example that also runs without hypre, it only skips this component;
# the consumer checks if(TARGET GinkgoTpl::hypre). Mirrors petsc/CMakeLists.txt.
if(PROJECT_IS_TOP_LEVEL)
    gko_tpl_find_hypre(REQUIRED)
else()
    gko_tpl_find_hypre()
    if(NOT HYPRE_FOUND)
        message(
            STATUS
            "GinkgoTpl::hypre: hypre not found (set HYPRE_ROOT), skipping it"
        )
        return()
    endif()
endif()
message(
    STATUS
    "GinkgoTpl::hypre: using hypre ${HYPRE_VERSION} from ${HYPRE_PREFIX} (CUDA: ${HYPRE_USING_CUDA})"
)

# Header-only: nothing here is compiled against Ginkgo, so a consumer's Ginkgo
# is the only one in its executable.
add_library(gko_tpl_hypre INTERFACE)
add_library(GinkgoTpl::hypre ALIAS gko_tpl_hypre)
set_target_properties(gko_tpl_hypre PROPERTIES EXPORT_NAME hypre)
target_include_directories(
    gko_tpl_hypre
    INTERFACE
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)
target_link_libraries(
    gko_tpl_hypre
    INTERFACE Ginkgo::ginkgo HYPRE::HYPRE MPI::MPI_CXX
)
target_compile_features(gko_tpl_hypre INTERFACE cxx_std_17)

if(GKO_TPL_BUILD_TESTS)
    add_subdirectory(test)
    # Diagnostic tools, gated the same way but never registered with ctest,
    # see tools/CMakeLists.txt.
    add_subdirectory(tools)
endif()

if(GKO_TPL_INSTALL)
    install(DIRECTORY include/ DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
    install(TARGETS gko_tpl_hypre EXPORT GinkgoTplWrappersHypre)
    install(
        EXPORT GinkgoTplWrappersHypre
        NAMESPACE GinkgoTpl::
        FILE GinkgoTplWrappers-hypre-targets.cmake
        DESTINATION "${GKO_TPL_INSTALL_CMAKEDIR}"
    )
    # Records the hypre this package was built against as a search hint.
    configure_file(
        cmake/GinkgoTplWrappers-hypre-dependencies.cmake.in
        "${PROJECT_BINARY_DIR}/GinkgoTplWrappers-hypre-dependencies.cmake"
        @ONLY
    )
    install(
        FILES
            "${PROJECT_BINARY_DIR}/GinkgoTplWrappers-hypre-dependencies.cmake"
            "${PROJECT_SOURCE_DIR}/cmake/gko_tpl_find_hypre.cmake"
        DESTINATION "${GKO_TPL_INSTALL_CMAKEDIR}"
    )
endif()
