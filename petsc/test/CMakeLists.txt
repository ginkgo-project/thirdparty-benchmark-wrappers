gko_tpl_add_test(
    petsc_ksp
    SOURCES ksp.cpp petsc_main.cpp
    LIBRARIES GinkgoTpl::petsc
)
# Runs without PETSc initialized, so it uses the MPI-only main.
gko_tpl_add_test(
    petsc_ksp_uninitialized
    SOURCES ksp_uninitialized.cpp ${GKO_TPL_MPI_TEST_MAIN}
    LIBRARIES GinkgoTpl::petsc
)
gko_tpl_add_test(
    petsc_ksp_distributed
    SOURCES ksp_distributed.cpp petsc_main.cpp
    LIBRARIES GinkgoTpl::petsc
    MPI_SIZE 3
)
