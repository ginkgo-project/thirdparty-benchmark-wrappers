// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_TPL_PETSC_DETAIL_TIMING_HPP_
#define GKO_TPL_PETSC_DETAIL_TIMING_HPP_


#include <chrono>

#include <mpi.h>

#include <petscdevice.h>

#include <ginkgo/core/base/exception_helpers.hpp>

#include <gko_tpl/petsc/detail/error.hpp>


namespace gko {
namespace ext {
namespace petsc {
namespace detail {


// Ends a timed phase: waits for the work PETSc queued on its current device
// context if the PETSc objects live on the device, then for every rank of
// `comm`, and returns the time. The device context is only touched for
// device objects, since getting it can initialize a device.
inline std::chrono::steady_clock::time_point end_of_phase(MPI_Comm comm,
                                                          bool device)
{
    if (device) {
        PetscDeviceContext dctx = nullptr;
        GKO_TPL_ASSERT_NO_PETSC_ERRORS(
            PetscDeviceContextGetCurrentContext(&dctx));
        GKO_TPL_ASSERT_NO_PETSC_ERRORS(PetscDeviceContextSynchronize(dctx));
    }
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Barrier(comm));
    return std::chrono::steady_clock::now();
}


}  // namespace detail
}  // namespace petsc
}  // namespace ext
}  // namespace gko


#endif  // GKO_TPL_PETSC_DETAIL_TIMING_HPP_
