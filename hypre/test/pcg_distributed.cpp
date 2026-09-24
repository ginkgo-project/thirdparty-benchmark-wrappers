// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cmath>
#include <memory>

#include <gtest/gtest.h>

#include <ginkgo/ginkgo.hpp>

#include <gko_tpl/hypre/pcg.hpp>

#include "common/test/assertions.hpp"
#include "common/test/laplacian.hpp"


class PcgDistributed : public ::testing::Test {
protected:
    using value_type = double;
    using local_index_type = gko::int32;
    using global_index_type = gko::int64;
    using pcg_type = gko::ext::hypre::solver::Pcg<value_type, local_index_type,
                                                  global_index_type>;
    using csr = gko::matrix::Csr<value_type, local_index_type>;
    using dense = gko::matrix::Dense<value_type>;
    using dist_mtx =
        gko::experimental::distributed::Matrix<value_type, local_index_type,
                                               global_index_type>;
    using dist_vec = gko::experimental::distributed::Vector<value_type>;
    using partition_type =
        gko::experimental::distributed::Partition<local_index_type,
                                                  global_index_type>;
    using md_type = gko::matrix_data<value_type, global_index_type>;

    PcgDistributed()
        : ref{gko::ReferenceExecutor::create()},
          comm{MPI_COMM_WORLD},
          part{gko::share(partition_type::build_from_global_size_uniform(
              ref, comm.size(), static_cast<global_index_type>(num_rows)))},
          mtx{gko::share(dist_mtx::create(ref, comm))},
          b{dist_vec::create(ref, comm)},
          x{dist_vec::create(ref, comm)}
    {
        mtx->read_distributed(laplacian_3d<global_index_type>(grid_dim), part);
        b->read_distributed(rhs_data(), part);
        // read_distributed zero-fills the entries missing from the data.
        x->read_distributed(md_type(gko::dim<2>{num_rows, 1}), part);
    }

    void SetUp() override { ASSERT_EQ(comm.size(), 3); }

    // b_i = sin(0.1 * i), the same right-hand side as the serial tests
    static md_type rhs_data()
    {
        md_type data(gko::dim<2>{num_rows, 1});
        for (gko::size_type i = 0; i < num_rows; i++) {
            data.nonzeros.emplace_back(static_cast<global_index_type>(i), 0,
                                       std::sin(0.1 * static_cast<double>(i)));
        }
        return data;
    }

    static constexpr global_index_type grid_dim = 8;
    static constexpr gko::size_type num_rows = 512;

    std::shared_ptr<gko::ReferenceExecutor> ref;
    gko::experimental::mpi::communicator comm;
    std::shared_ptr<partition_type> part;
    std::shared_ptr<dist_mtx> mtx;
    std::unique_ptr<dist_vec> b;
    std::unique_ptr<dist_vec> x;
};


TEST_F(PcgDistributed, MatchesTheSerialSolution)
{
    auto solver = pcg_type::build()
                      .with_tolerance(1e-10)
                      .with_max_iters(200)
                      .on(ref)
                      ->generate(mtx);

    solver->apply(b, x);

    ASSERT_TRUE(solver->has_converged());
    // Every rank's slice of the distributed solution must match the serial
    // solve of the same system.
    auto serial_mtx = gko::share(csr::create(ref));
    serial_mtx->read(laplacian_3d<local_index_type>(grid_dim));
    auto serial_solver = pcg_type::build()
                             .with_tolerance(1e-10)
                             .with_max_iters(200)
                             .on(ref)
                             ->generate(serial_mtx);
    auto serial_b = dense::create(ref);
    serial_b->read(rhs_data());
    auto serial_x = dense::create(ref, gko::dim<2>{num_rows, 1});
    serial_x->fill(0.0);
    serial_solver->apply(serial_b, serial_x);

    // part's ranges are not evenly divisible by comm.size() (512 / 3), so
    // the first row owned by this rank must come from the partition's actual
    // range bounds rather than an evenly-divided formula.
    const auto first_row =
        static_cast<gko::size_type>(part->get_range_bounds()[comm.rank()]);
    const auto* local = x->get_local_vector()->get_const_values();
    for (gko::size_type i = 0; i < x->get_local_vector()->get_size()[0]; i++) {
        ASSERT_NEAR(local[i], serial_x->at(first_row + i, 0), 1e-7);
    }
}


TEST_F(PcgDistributed, ReportsTheSameIterationCountOnEveryRank)
{
    auto solver =
        pcg_type::build().with_tolerance(1e-10).on(ref)->generate(mtx);
    solver->apply(b, x);

    auto iterations = static_cast<int>(solver->get_num_iterations());
    auto max_iterations = iterations;
    ASSERT_EQ(MPI_Allreduce(&iterations, &max_iterations, 1, MPI_INT, MPI_MAX,
                            comm.get()),
              MPI_SUCCESS);
    ASSERT_EQ(iterations, max_iterations);
}


TEST_F(PcgDistributed, RepeatedApplyGivesTheSameAnswer)
{
    // apply re-points the generation-time vectors rather than rebuilding
    // them; a pointer left over from the first apply would make the second
    // solve read or write the first one's memory.
    auto solver = pcg_type::build()
                      .with_tolerance(1e-10)
                      .with_max_iters(200)
                      .on(ref)
                      ->generate(mtx);
    auto x_first = gko::clone(x);
    auto x_second = gko::clone(x);

    solver->apply(b, x_first);
    const auto first_iterations = solver->get_num_iterations();
    auto x_first_before = gko::clone(x_first);
    solver->apply(b, x_second);

    ASSERT_TRUE(solver->has_converged());
    ASSERT_EQ(solver->get_num_iterations(), first_iterations);
    GKO_TPL_ASSERT_MTX_NEAR(x_first->get_local_vector(),
                            x_first_before->get_local_vector(), 0.0);
    GKO_TPL_ASSERT_MTX_NEAR(x_second->get_local_vector(),
                            x_first->get_local_vector(), 1e-14);
}


TEST_F(PcgDistributed, RejectsMismatchedLocalSizesOnEveryRank)
{
    auto solver = pcg_type::build().on(ref)->generate(mtx);
    // One rank too few local rows: the check is collective, so every rank
    // must throw rather than deadlock in the solve. Vector::create only
    // asserts the column counts agree, with no collective communication, so
    // it happily builds a vector whose local row count disagrees with the
    // partition and the other ranks'.
    auto wrong =
        dist_vec::create(ref, comm, gko::dim<2>{num_rows, 1},
                         gko::dim<2>{x->get_local_vector()->get_size()[0] -
                                         (comm.rank() == 1 ? 1 : 0),
                                     1});
    wrong->fill(0.0);

    ASSERT_THROW(solver->apply(b, wrong), gko::DimensionMismatch);
}
