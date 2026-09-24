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
