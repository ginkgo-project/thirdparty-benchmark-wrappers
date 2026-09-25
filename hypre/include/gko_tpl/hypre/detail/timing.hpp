// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_TPL_HYPRE_DETAIL_TIMING_HPP_
#define GKO_TPL_HYPRE_DETAIL_TIMING_HPP_


#include <chrono>

#include <mpi.h>

#include <_hypre_utilities.h>

#include <ginkgo/core/base/exception_helpers.hpp>

#include <gko_tpl/hypre/detail/error.hpp>


namespace gko {
namespace ext {
namespace hypre {
namespace detail {


// Ends a timed phase: waits for hypre's device work if any, then for every
// rank of `comm`, and returns the time. Ginkgo's executor synchronization
// does not cover hypre's stream, hence hypre's own synchronization here.
inline std::chrono::steady_clock::time_point end_of_phase(MPI_Comm comm,
                                                          bool device)
{
    // hypre_SyncDevice is only declared by a GPU-enabled build; `device` can
    // only be true then, since set_process_policy rejects
    // HYPRE_MEMORY_DEVICE otherwise.
#if defined(HYPRE_USING_GPU)
    if (device) {
        // hypre 2.33's hypre_SyncDevice() replaces the older
        // hypre_SyncCudaDevice(hypre_Handle*): same synchronization, but the
        // handle moved from an explicit argument to internal state, reached
        // here via hypre_handle(), which both versions declare
        // unconditionally.
#if HYPRE_RELEASE_NUMBER >= 23300
        GKO_TPL_ASSERT_NO_HYPRE_ERRORS(hypre_SyncDevice());
#else
        GKO_TPL_ASSERT_NO_HYPRE_ERRORS(hypre_SyncCudaDevice(hypre_handle()));
#endif
    }
#else
    (void)device;
#endif
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Barrier(comm));
    return std::chrono::steady_clock::now();
}


}  // namespace detail
}  // namespace hypre
}  // namespace ext
}  // namespace gko


#endif  // GKO_TPL_HYPRE_DETAIL_TIMING_HPP_
