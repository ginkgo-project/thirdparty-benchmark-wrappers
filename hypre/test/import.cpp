// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <memory>

#include <gtest/gtest.h>

#include <_hypre_parcsr_mv.h>

#include <ginkgo/ginkgo.hpp>

#include <gko_tpl/hypre/detail/import.hpp>

#include "common/test/assertions.hpp"
#include "common/test/laplacian.hpp"


class Import : public ::testing::Test {
protected:
    using value_type = double;
    using index_type = gko::int32;
    using csr = gko::matrix::Csr<value_type, index_type>;
    using dense = gko::matrix::Dense<value_type>;

    Import()
        : ref{gko::ReferenceExecutor::create()},
          mtx{gko::share(csr::create(ref))}
    {
        mtx->read(laplacian_3d<index_type>(grid_dim));
    }

    static constexpr index_type grid_dim = 6;
    static constexpr gko::size_type num_rows = 216;

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<csr> mtx;
};


TEST_F(Import, SerialMatvecMatchesGinkgo)
{
    auto x = dense::create(ref, gko::dim<2>{num_rows, 1});
    for (gko::size_type i = 0; i < num_rows; i++) {
        x->at(i, 0) = 1.0 + 0.01 * static_cast<double>(i);
    }
    auto expected = dense::create(ref, gko::dim<2>{num_rows, 1});
    mtx->apply(x, expected);

    auto imported =
        gko::ext::hypre::detail::build_serial_par_csr<index_type>(mtx);
    ASSERT_EQ(imported.num_local_rows, static_cast<HYPRE_Int>(num_rows));
    ASSERT_EQ(imported.global_rows, static_cast<HYPRE_BigInt>(num_rows));

    // hypre owns these two vectors; only the matrix is under test here.
    HYPRE_BigInt partitioning[2] = {0, static_cast<HYPRE_BigInt>(num_rows)};
    auto* hypre_x = hypre_ParVectorCreate(imported.comm, imported.global_rows,
                                          partitioning);
    auto* hypre_y = hypre_ParVectorCreate(imported.comm, imported.global_rows,
                                          partitioning);
    hypre_ParVectorInitialize(hypre_x);
    hypre_ParVectorInitialize(hypre_y);
    auto* x_data = hypre_VectorData(hypre_ParVectorLocalVector(hypre_x));
    for (gko::size_type i = 0; i < num_rows; i++) {
        x_data[i] = x->at(i, 0);
    }
    hypre_ParVectorSetConstantValues(hypre_y, 0.0);
    ASSERT_EQ(
        hypre_ParCSRMatrixMatvec(1.0, imported.matrix, hypre_x, 0.0, hypre_y),
        0);

    auto actual = dense::create(ref, gko::dim<2>{num_rows, 1});
    const auto* y_data = hypre_VectorData(hypre_ParVectorLocalVector(hypre_y));
    for (gko::size_type i = 0; i < num_rows; i++) {
        actual->at(i, 0) = y_data[i];
    }
    hypre_ParVectorDestroy(hypre_x);
    hypre_ParVectorDestroy(hypre_y);

    GKO_TPL_ASSERT_MTX_NEAR(actual, expected, 1e-14);
}


TEST_F(Import, KeepsTheSystemMatrixAlive)
{
    auto imported =
        gko::ext::hypre::detail::build_serial_par_csr<index_type>(mtx);
    ASSERT_GT(mtx.use_count(), 1);
}


// hypre reads a row's diagonal as its first stored entry (see
// reorder_diag_first in detail/import.hpp); Ginkgo stores columns in
// ascending order, so without that reordering an interior row's first entry
// is an off-diagonal coefficient instead. This is what pins the invariant
// down directly, rather than only through convergence, the way the bug that
// motivated it was originally (and only indirectly) caught.
//
// The off-diagonal block is not checked here: it holds no diagonal entry at
// all (the diagonal, by definition, is always in the diag block), so it is
// never reordered and the invariant does not apply to it, see
// build_distributed_par_csr's comment in detail/import.hpp.
TEST_F(Import, DiagonalIsFirstInEachRow)
{
    auto imported =
        gko::ext::hypre::detail::build_serial_par_csr<index_type>(mtx);
    // par_csr::diag_row_ptrs/diag_col_idxs are only populated when
    // as_hypre_int/copy_as_hypre_int actually convert or copy; when
    // IndexType already matches HYPRE_Int, as_hypre_int returns Ginkgo's own
    // pointer without touching its `storage` argument, so diag_row_ptrs can
    // be an empty array. The hypre matrix's own diag block is what
    // set_block always points at the real, in-use arrays, so it is read
    // directly here instead.
    auto* diag = hypre_ParCSRMatrixDiag(imported.matrix);
    const auto* row_ptrs = hypre_CSRMatrixI(diag);
    const auto* col_idxs = hypre_CSRMatrixJ(diag);
    for (HYPRE_Int row = 0; row < imported.num_local_rows; row++) {
        if (row_ptrs[row + 1] > row_ptrs[row]) {
            EXPECT_EQ(col_idxs[row_ptrs[row]], row)
                << "row " << row
                << " of the diagonal block does not have its diagonal entry "
                   "first";
        }
    }
}
