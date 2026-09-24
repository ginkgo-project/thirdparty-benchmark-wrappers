// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_TPL_HYPRE_ENVIRONMENT_HPP_
#define GKO_TPL_HYPRE_ENVIRONMENT_HPP_


#include <mpi.h>

#include <HYPRE_utilities.h>

#include <ginkgo/core/base/exception_helpers.hpp>

#include <gko_tpl/hypre/detail/error.hpp>


namespace gko {
namespace ext {
namespace hypre {


/**
 * RAII guard for hypre's global state.
 *
 * hypre must be initialized before a gko::ext::hypre::solver::Pcg is
 * generated. MPI must already be initialized when this object is created, for
 * example by a gko::experimental::mpi::environment that outlives it.
 *
 * The constructor initializes hypre unless it is initialized already, e.g. by
 * the application. The destructor finalizes hypre only if this object
 * initialized it and MPI has not been finalized yet. Every solver must be
 * destroyed before hypre is finalized.
 */
class environment {
public:
    /**
     * Initializes hypre if it is not initialized yet.
     *
     * @throws InvalidStateError  if MPI is not initialized or hypre fails to
     *                            initialize
     */
    environment() : owns_hypre_{false}
    {
        int mpi_initialized = 0;
        MPI_Initialized(&mpi_initialized);
        if (!mpi_initialized) {
            GKO_INVALID_STATE(
                "MPI must be initialized before creating a "
                "gko::ext::hypre::environment, e.g. by a "
                "gko::experimental::mpi::environment");
        }
        if (!HYPRE_Initialized()) {
            GKO_TPL_ASSERT_NO_HYPRE_ERRORS(HYPRE_Initialize());
            owns_hypre_ = true;
        }
    }

    ~environment()
    {
        if (!owns_hypre_) {
            return;
        }
        // HYPRE_Finalize calls into MPI, so it must be skipped once MPI is
        // finalized, e.g. for a static environment destroyed after
        // MPI_Finalize.
        int mpi_finalized = 0;
        MPI_Finalized(&mpi_finalized);
        if (!mpi_finalized && HYPRE_Initialized() && !HYPRE_Finalized()) {
            HYPRE_Finalize();
        }
    }

    environment(const environment&) = delete;
    environment(environment&&) = delete;
    environment& operator=(const environment&) = delete;
    environment& operator=(environment&&) = delete;

private:
    bool owns_hypre_;
};


}  // namespace hypre
}  // namespace ext
}  // namespace gko


#endif  // GKO_TPL_HYPRE_ENVIRONMENT_HPP_
