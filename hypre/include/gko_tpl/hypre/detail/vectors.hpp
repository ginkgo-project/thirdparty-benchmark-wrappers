// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_TPL_HYPRE_DETAIL_VECTORS_HPP_
#define GKO_TPL_HYPRE_DETAIL_VECTORS_HPP_


#include <mpi.h>

// Not <_hypre_seq_mv.h> directly: it was renamed between versions
// (seq_mv.h in 2.32, _hypre_seq_mv.h in 3.2); _hypre_parcsr_mv.h below
// includes whichever name is correct for the installed version.
#include <_hypre_parcsr_mv.h>
#include <HYPRE.h>
#include <HYPRE_IJ_mv.h>
#include <HYPRE_utilities.h>

#include <ginkgo/core/base/exception_helpers.hpp>

#include <gko_tpl/hypre/detail/error.hpp>


namespace gko {
namespace ext {
namespace hypre {
namespace detail {


// A hypre vector over caller-owned memory, host or device: hypre allocates
// and frees nothing, so a solve copies no vector data. Equivalent of PETSc's
// VecPlaceArray.
//
// 3.2 offers this through its public IJ interface, used here. 2.32 has
// neither HYPRE_IJVectorInitializeShell nor HYPRE_IJVectorSetData, so it
// goes through hypre's internals instead: hypre_SeqVectorInitialize_v2 only
// allocates when the data pointer is still null, so setting it first and
// clearing the local vector's owner flag leaves the caller's array in place.
//
// Construction is not free -- 3.2's HYPRE_IJVectorAssemble does an
// unconditional Allreduce -- so a solver builds its vectors once and
// re-points them with set_values() per apply instead, see there.
class placed_vector {
public:
    placed_vector(MPI_Comm comm, HYPRE_BigInt global_size,
                  HYPRE_BigInt row_start, HYPRE_Int local_size,
                  const double* values, HYPRE_MemoryLocation location)
    {
        auto* data = const_cast<HYPRE_Complex*>(
            static_cast<const HYPRE_Complex*>(values));
        // A throw after HYPRE_IJVectorCreate (or hypre_ParVectorCreate) would
        // leak the hypre object, since a failed constructor's destructor
        // never runs; caught below and released via destroy() instead.
        try {
#if HYPRE_RELEASE_NUMBER >= 30200
            // Unused here: the IJ interface takes hypre's process-global
            // location and derives global_size from the range. Kept in the
            // signature since the 2.32 branch below does need `location`.
            (void)location;
            (void)global_size;
            // An empty range is jlower > jupper, which hypre accepts.
            GKO_TPL_ASSERT_NO_HYPRE_ERRORS(HYPRE_IJVectorCreate(
                comm, row_start, row_start + local_size - 1, &ij_vector_));
            GKO_TPL_ASSERT_NO_HYPRE_ERRORS(
                HYPRE_IJVectorSetObjectType(ij_vector_, HYPRE_PARCSR));
            GKO_TPL_ASSERT_NO_HYPRE_ERRORS(
                HYPRE_IJVectorInitializeShell(ij_vector_));
            GKO_TPL_ASSERT_NO_HYPRE_ERRORS(
                HYPRE_IJVectorSetData(ij_vector_, data));
            GKO_TPL_ASSERT_NO_HYPRE_ERRORS(HYPRE_IJVectorAssemble(ij_vector_));
            void* object = nullptr;
            GKO_TPL_ASSERT_NO_HYPRE_ERRORS(
                HYPRE_IJVectorGetObject(ij_vector_, &object));
            vector_ = static_cast<hypre_ParVector*>(object);
#else
            HYPRE_BigInt partitioning[2] = {
                row_start, row_start + static_cast<HYPRE_BigInt>(local_size)};
            vector_ = hypre_ParVectorCreate(comm, global_size, partitioning);
            if (!vector_) {
                GKO_INVALID_STATE("hypre_ParVectorCreate returned null");
            }
            owns_par_vector_ = true;
            // 2.32 has no hypre_ParVectorSetData (a 3.2 addition), so the
            // local vector's data pointer is assigned directly through the
            // struct macros, set before Initialize_v2 since
            // hypre_SeqVectorInitialize_v2 only allocates when it is null.
            // Only the *local* vector's owner flag is cleared, not the
            // ParVector's: that keeps hypre_ParVectorDestroy calling
            // hypre_SeqVectorDestroy, which frees hypre's own hypre_Vector
            // struct while leaving the caller's data alone; clearing the
            // ParVector-level flag instead would skip that call and leak
            // the struct.
            auto* local_vector = hypre_ParVectorLocalVector(vector_);
            hypre_VectorData(local_vector) = data;
            GKO_TPL_ASSERT_NO_HYPRE_ERRORS(
                hypre_SeqVectorSetDataOwner(local_vector, 0));
            GKO_TPL_ASSERT_NO_HYPRE_ERRORS(
                hypre_ParVectorInitialize_v2(vector_, location));
#endif
        } catch (...) {
            destroy();
            throw;
        }
    }

    ~placed_vector() { destroy(); }

    /**
     * Points the vector at a different array, still owned by the caller:
     * nothing is copied and hypre does not free it, exactly as at
     * construction.
     *
     * Needs no re-assembly and does no collective work. On 3.2,
     * hypre_SeqVectorSetData only frees the old array if the vector owns it
     * (it doesn't, the owner flag was already cleared) and otherwise just
     * stores the new pointer; size, partitioning and memory location are
     * untouched, so the assembled state stays valid. On 2.32 it is the same
     * struct-macro assignment the constructor uses.
     */
    void set_values(const double* values)
    {
        auto* data = const_cast<HYPRE_Complex*>(
            static_cast<const HYPRE_Complex*>(values));
#if HYPRE_RELEASE_NUMBER >= 30200
        GKO_TPL_ASSERT_NO_HYPRE_ERRORS(HYPRE_IJVectorSetData(ij_vector_, data));
#else
        hypre_VectorData(hypre_ParVectorLocalVector(vector_)) = data;
#endif
    }

    hypre_ParVector* get() const { return vector_; }

    placed_vector(const placed_vector&) = delete;
    placed_vector& operator=(const placed_vector&) = delete;

private:
    void destroy() noexcept
    {
        if (!HYPRE_Initialized() || HYPRE_Finalized()) {
            return;
        }
        if (ij_vector_) {
            // Destroys the ParVector it owns; the data pointer is not freed,
            // because the vector never owned it.
            HYPRE_IJVectorDestroy(ij_vector_);
            ij_vector_ = nullptr;
            vector_ = nullptr;
        } else if (owns_par_vector_ && vector_) {
            hypre_ParVectorDestroy(vector_);
            vector_ = nullptr;
            owns_par_vector_ = false;
        }
    }

    HYPRE_IJVector ij_vector_ = nullptr;
    hypre_ParVector* vector_ = nullptr;
    bool owns_par_vector_ = false;
};


}  // namespace detail
}  // namespace hypre
}  // namespace ext
}  // namespace gko


#endif  // GKO_TPL_HYPRE_DETAIL_VECTORS_HPP_
