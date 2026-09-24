gko_tpl_add_test(
    hypre_found
    SOURCES found.cpp ${GKO_TPL_MPI_TEST_MAIN}
    LIBRARIES GinkgoTpl::hypre
)

# Runs without hypre initialized, so it uses the MPI-only main.
gko_tpl_add_test(
    hypre_environment
    SOURCES environment.cpp ${GKO_TPL_MPI_TEST_MAIN}
    LIBRARIES GinkgoTpl::hypre
)

# Pure host-side name-to-integer lookups, no hypre initialization needed.
gko_tpl_add_test(
    hypre_options
    SOURCES options.cpp ${GKO_TPL_MPI_TEST_MAIN}
    LIBRARIES GinkgoTpl::hypre
)

gko_tpl_add_test(
    hypre_import
    SOURCES import.cpp hypre_main.cpp
    LIBRARIES GinkgoTpl::hypre
)

gko_tpl_add_test(
    hypre_import_distributed
    SOURCES import_distributed.cpp hypre_main.cpp
    LIBRARIES GinkgoTpl::hypre
    MPI_SIZE 3
)

gko_tpl_add_test(
    hypre_vectors
    SOURCES vectors.cpp hypre_main.cpp
    LIBRARIES GinkgoTpl::hypre
)

gko_tpl_add_test(
    hypre_pcg
    SOURCES pcg.cpp hypre_main.cpp
    LIBRARIES GinkgoTpl::hypre
)

gko_tpl_add_test(
    hypre_pcg_distributed
    SOURCES pcg_distributed.cpp hypre_main.cpp
    LIBRARIES GinkgoTpl::hypre
    MPI_SIZE 3
)

# Runs without hypre initialized, so it uses the MPI-only main.
gko_tpl_add_test(
    hypre_uninitialized
    SOURCES uninitialized.cpp ${GKO_TPL_MPI_TEST_MAIN}
    LIBRARIES GinkgoTpl::hypre
)

gko_tpl_add_test(
    hypre_config
    SOURCES config.cpp hypre_main.cpp
    LIBRARIES GinkgoTpl::hypre
)
