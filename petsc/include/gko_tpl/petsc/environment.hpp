// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_TPL_PETSC_ENVIRONMENT_HPP_
#define GKO_TPL_PETSC_ENVIRONMENT_HPP_


#include <mpi.h>

#include <petscsys.h>

#include <ginkgo/core/base/exception_helpers.hpp>

#include <gko_tpl/petsc/detail/error.hpp>


namespace gko {
namespace ext {
namespace petsc {


/**
 * RAII guard for PETSc's global state.
 *
 * PETSc must be initialized before a gko::ext::petsc::solver::Ksp is
 * generated. MPI must already be initialized when this object is created,
 * for example by a gko::experimental::mpi::environment that outlives it.
 *
 * The constructor initializes PETSc unless it is initialized already, e.g. by
 * the application. The destructor finalizes PETSc only if this object
 * initialized it and MPI has not been finalized yet; since PETSc did not
 * initialize MPI, finalizing PETSc leaves MPI running. Every Ksp must be
 * destroyed before PETSc is finalized.
 */
class environment {
public:
    /**
     * Initializes PETSc if it is not initialized yet.
     *
     * @throws InvalidStateError  if MPI is not initialized or PETSc fails to
     *                            initialize
     */
    environment() : owns_petsc_{false}
    {
        int mpi_initialized = 0;
        MPI_Initialized(&mpi_initialized);
        if (!mpi_initialized) {
            GKO_INVALID_STATE(
                "MPI must be initialized before creating a "
                "gko::ext::petsc::environment, e.g. by a "
                "gko::experimental::mpi::environment");
        }
        PetscBool petsc_initialized = PETSC_FALSE;
        GKO_TPL_ASSERT_NO_PETSC_ERRORS(PetscInitialized(&petsc_initialized));
        if (!petsc_initialized) {
            GKO_TPL_ASSERT_NO_PETSC_ERRORS(PetscInitializeNoArguments());
            owns_petsc_ = true;
        }
    }

    ~environment()
    {
        if (!owns_petsc_) {
            return;
        }
        // PetscFinalize calls into MPI, so it must be skipped once MPI is
        // finalized, e.g. for a static environment destroyed after
        // MPI_Finalize.
        int mpi_finalized = 0;
        MPI_Finalized(&mpi_finalized);
        PetscBool petsc_finalized = PETSC_FALSE;
        PetscFinalized(&petsc_finalized);
        if (!mpi_finalized && !petsc_finalized) {
            PetscFinalize();
        }
    }

    environment(const environment&) = delete;
    environment(environment&&) = delete;
    environment& operator=(const environment&) = delete;
    environment& operator=(environment&&) = delete;

private:
    bool owns_petsc_;
};


}  // namespace petsc
}  // namespace ext
}  // namespace gko


#endif  // GKO_TPL_PETSC_ENVIRONMENT_HPP_
