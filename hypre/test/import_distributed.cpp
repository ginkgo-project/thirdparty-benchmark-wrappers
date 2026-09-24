// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cmath>
#include <memory>

#include <gtest/gtest.h>

#include <_hypre_parcsr_mv.h>

#include <ginkgo/ginkgo.hpp>

#include <gko_tpl/hypre/detail/import.hpp>

#include "common/test/laplacian.hpp"


class ImportDistributed : public ::testing::Test {
protected:
    using value_type = double;
    using local_index_type = gko::int32;
    using global_index_type = gko::int64;
    using dist_mtx =
        gko::experimental::distributed::Matrix<value_type, local_index_type,
                                               global_index_type>;
    using dist_vec = gko::experimental::distributed::Vector<value_type>;
    using partition_type =
        gko::experimental::distributed::Partition<local_index_type,
                                                  global_index_type>;
    using md_type = gko::matrix_data<value_type, global_index_type>;

    ImportDistributed()
        : ref{gko::ReferenceExecutor::create()},
          comm{MPI_COMM_WORLD},
          part{gko::share(partition_type::build_from_global_size_uniform(
              ref, comm.size(), static_cast<global_index_type>(num_rows)))},
          mtx{gko::share(dist_mtx::create(ref, comm))},
          x{dist_vec::create(ref, comm)},
          y{dist_vec::create(ref, comm)}
    {
        mtx->read_distributed(laplacian_3d<global_index_type>(grid_dim), part);
        md_type x_data(gko::dim<2>{num_rows, 1});
        for (gko::size_type i = 0; i < num_rows; i++) {
            x_data.nonzeros.emplace_back(static_cast<global_index_type>(i), 0,
                                         1.0 + 0.01 * static_cast<double>(i));
        }
        x->read_distributed(x_data, part);
        y->read_distributed(md_type(gko::dim<2>{num_rows, 1}), part);
    }

    void SetUp() override { ASSERT_EQ(comm.size(), 3); }

    static constexpr global_index_type grid_dim = 8;
    static constexpr gko::size_type num_rows = 512;

    std::shared_ptr<gko::ReferenceExecutor> ref;
    gko::experimental::mpi::communicator comm;
    std::shared_ptr<partition_type> part;
    std::shared_ptr<dist_mtx> mtx;
    std::unique_ptr<dist_vec> x;
    std::unique_ptr<dist_vec> y;
};


TEST_F(ImportDistributed, MatvecMatchesGinkgo)
{
    mtx->apply(x, y);
    auto imported = gko::ext::hypre::detail::build_any_distributed_par_csr<
        local_index_type>(mtx);
    ASSERT_TRUE(imported.has_value());
    const auto local_rows =
        static_cast<gko::size_type>(imported->num_local_rows);
    ASSERT_EQ(local_rows, x->get_local_vector()->get_size()[0]);

    HYPRE_BigInt partitioning[2] = {
        imported->row_start,
        imported->row_start + static_cast<HYPRE_BigInt>(local_rows)};
    auto* hypre_x = hypre_ParVectorCreate(imported->comm, imported->global_rows,
                                          partitioning);
    auto* hypre_y = hypre_ParVectorCreate(imported->comm, imported->global_rows,
                                          partitioning);
    hypre_ParVectorInitialize(hypre_x);
    hypre_ParVectorInitialize(hypre_y);
    auto* x_data = hypre_VectorData(hypre_ParVectorLocalVector(hypre_x));
    const auto* local_x = x->get_local_vector()->get_const_values();
    for (gko::size_type i = 0; i < local_rows; i++) {
        x_data[i] = local_x[i];
    }
    hypre_ParVectorSetConstantValues(hypre_y, 0.0);
    ASSERT_EQ(
        hypre_ParCSRMatrixMatvec(1.0, imported->matrix, hypre_x, 0.0, hypre_y),
        0);

    const auto* y_data = hypre_VectorData(hypre_ParVectorLocalVector(hypre_y));
    const auto* expected = y->get_local_vector()->get_const_values();
    double error = 0.0;
    double scale = 0.0;
    for (gko::size_type i = 0; i < local_rows; i++) {
        error += (y_data[i] - expected[i]) * (y_data[i] - expected[i]);
        scale += expected[i] * expected[i];
    }
    hypre_ParVectorDestroy(hypre_x);
    hypre_ParVectorDestroy(hypre_y);
    ASSERT_LE(std::sqrt(error), 1e-12 * std::sqrt(scale));
}


TEST_F(ImportDistributed, RejectsAnUnsupportedOperator)
{
    auto dense = gko::share(
        gko::matrix::Dense<value_type>::create(ref, gko::dim<2>{2, 2}));
    ASSERT_FALSE(gko::ext::hypre::detail::build_any_distributed_par_csr<
                     local_index_type>(dense)
                     .has_value());
}
