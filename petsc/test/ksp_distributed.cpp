// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cmath>
#include <memory>

#include <gtest/gtest.h>

#include <ginkgo/ginkgo.hpp>

#include <gko_tpl/petsc/ksp.hpp>

#include "common/test/assertions.hpp"
#include "common/test/laplacian.hpp"


class KspDistributed : public ::testing::Test {
protected:
    using value_type = double;
    using local_index_type = gko::int32;
    using global_index_type = gko::int64;
    using ksp_type = gko::ext::petsc::solver::Ksp<value_type, local_index_type>;
    using dense = gko::matrix::Dense<value_type>;
    using dist_mtx =
        gko::experimental::distributed::Matrix<value_type, local_index_type,
                                               global_index_type>;
    using dist_vec = gko::experimental::distributed::Vector<value_type>;
    using partition_type =
        gko::experimental::distributed::Partition<local_index_type,
                                                  global_index_type>;
    using md_type = gko::matrix_data<value_type, global_index_type>;

    KspDistributed()
        : ref{gko::ReferenceExecutor::create()},
          comm{MPI_COMM_WORLD},
          global_data{laplacian_3d<global_index_type>(grid_dim)},
          part{gko::share(partition_type::build_from_global_size_uniform(
              ref, comm.size(), static_cast<global_index_type>(num_rows)))},
          mtx{gko::share(dist_mtx::create(ref, comm))},
          b{dist_vec::create(ref, comm)},
          x{dist_vec::create(ref, comm)}
    {
        mtx->read_distributed(global_data, part);
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
    // A tight tolerance, so that solutions of different solvers agree to
    // well within the comparison tolerance (cond(A) is about 32).
    static constexpr const char* cg_boomeramg =
        "-ksp_type cg -ksp_norm_type unpreconditioned -ksp_rtol 1e-10 "
        "-ksp_max_it 200 -pc_type hypre -pc_hypre_type boomeramg";

    std::shared_ptr<gko::ReferenceExecutor> ref;
    gko::experimental::mpi::communicator comm;
    md_type global_data;
    std::shared_ptr<partition_type> part;
    std::shared_ptr<dist_mtx> mtx;
    std::unique_ptr<dist_vec> b;
    std::unique_ptr<dist_vec> x;
};


TEST_F(KspDistributed, MatchesSerialKspAndGinkgoCg)
{
    auto serial_mtx =
        gko::share(gko::matrix::Csr<value_type, local_index_type>::create(ref));
    serial_mtx->read(laplacian_3d<local_index_type>(grid_dim));
    auto serial_b = dense::create(ref);
    serial_b->read(rhs_data());
    auto serial_x = dense::create(ref, gko::dim<2>{num_rows, 1});
    serial_x->fill(0.0);
    auto x_ginkgo = gko::clone(x);
    auto solver =
        ksp_type::build().with_options(cg_boomeramg).on(ref)->generate(mtx);
    auto serial_solver = ksp_type::build()
                             .with_options(cg_boomeramg)
                             .on(ref)
                             ->generate(serial_mtx);
    auto ginkgo_solver =
        gko::solver::Cg<value_type>::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(1000u),
                           gko::stop::ResidualNorm<value_type>::build()
                               .with_reduction_factor(1e-12))
            .on(ref)
            ->generate(mtx);

    solver->apply(b, x);
    serial_solver->apply(serial_b, serial_x);
    ginkgo_solver->apply(b, x_ginkgo);

    ASSERT_TRUE(solver->has_converged());
    ASSERT_TRUE(serial_solver->has_converged());
    const auto local_start =
        static_cast<gko::size_type>(part->get_range_bounds()[comm.rank()]);
    const auto local_rows = x->get_local_vector()->get_size()[0];
    auto serial_local = serial_x->create_submatrix(
        gko::span{local_start, local_start + local_rows}, gko::span{0, 1});
    GKO_TPL_ASSERT_MTX_NEAR(x->get_local_vector(), serial_local, 1e-7);
    GKO_TPL_ASSERT_MTX_NEAR(x->get_local_vector(), x_ginkgo->get_local_vector(),
                            1e-7);
}


TEST_F(KspDistributed, AdvancedApplyScalesAndAddsTheSolution)
{
    auto solver =
        ksp_type::build().with_options(cg_boomeramg).on(ref)->generate(mtx);
    auto alpha = gko::initialize<dense>({2.0}, ref);
    auto beta = gko::initialize<dense>({-0.5}, ref);
    x->fill(1.0);
    auto expected = gko::clone(x);
    auto x_solve = gko::clone(x);
    solver->apply(b, x_solve);
    expected->scale(beta);
    expected->add_scaled(alpha, x_solve);

    solver->apply(alpha, b, beta, x);

    GKO_TPL_ASSERT_MTX_NEAR(x->get_local_vector(), expected->get_local_vector(),
                            1e-14);
}


TEST_F(KspDistributed, ThrowsForPartitionNotOrderedByRank)
{
    // Contiguous thirds owned by ranks 1, 0, 2: rank 0's rows do not come
    // first, which PETSc's parallel row ownership requires.
    gko::array<gko::experimental::distributed::comm_index_type> mapping(
        ref, num_rows);
    for (gko::size_type i = 0; i < num_rows; i++) {
        mapping.get_data()[i] =
            i < num_rows / 3 ? 1 : (i < 2 * num_rows / 3 ? 0 : 2);
    }
    auto unordered_part = gko::share(
        partition_type::build_from_mapping(ref, mapping, comm.size()));
    auto unordered_mtx = gko::share(dist_mtx::create(ref, comm));
    unordered_mtx->read_distributed(global_data, unordered_part);
    auto factory = ksp_type::build().with_options(cg_boomeramg).on(ref);

    ASSERT_THROW(factory->generate(unordered_mtx), gko::NotSupported);
}


TEST_F(KspDistributed, ThrowsForNonUnitLocalStride)
{
    auto solver =
        ksp_type::build().with_options(cg_boomeramg).on(ref)->generate(mtx);
    const auto local_rows = x->get_local_vector()->get_size()[0];
    // A stride of 2 for a single column: apply places the local values
    // straight into PETSc's vectors, which assumes a unit stride.
    auto strided_local_b = dense::create(ref, gko::dim<2>{local_rows, 1}, 2);
    strided_local_b->fill(0.0);
    auto strided_b = dist_vec::create(ref, comm, gko::dim<2>{num_rows, 1},
                                      std::move(strided_local_b));

    ASSERT_THROW(solver->apply(strided_b, x), gko::NotSupported);
}


TEST_F(KspDistributed, ThrowsForNonUnitLocalStrideOnOnlyOneRank)
{
    auto solver =
        ksp_type::build().with_options(cg_boomeramg).on(ref)->generate(mtx);
    const auto local_rows = x->get_local_vector()->get_size()[0];
    // Only rank 0's local b has a non-unit stride; every other rank builds a
    // perfectly valid unit-stride local vector. Without reducing the
    // rejection across ranks, only rank 0 would throw here while the others
    // proceeded into the collective KSPSolve, deadlocking instead of failing
    // cleanly (and identically) on every rank.
    const auto stride = comm.rank() == 0 ? 2 : 1;
    auto local_b = dense::create(ref, gko::dim<2>{local_rows, 1}, stride);
    local_b->fill(0.0);
    auto skewed_b = dist_vec::create(ref, comm, gko::dim<2>{num_rows, 1},
                                     std::move(local_b));

    ASSERT_THROW(solver->apply(skewed_b, x), gko::NotSupported);
}


TEST_F(KspDistributed,
       ThrowsDimensionMismatchOnEveryRankForMismatchedLocalPartition)
{
    auto solver =
        ksp_type::build().with_options(cg_boomeramg).on(ref)->generate(mtx);
    // Same global size (512) as `part`, but split 171/172/169 instead of
    // `part`'s 171/171/170: rank 0's local size still matches the matrix's
    // row partition, so only ranks 1 and 2 would notice a mismatch if the
    // check were not reduced across ranks. Without the fix, rank 0 would
    // proceed into the collective KSPSolve while ranks 1 and 2 have already
    // thrown, hanging instead of failing cleanly on every rank.
    gko::array<global_index_type> other_ranges(
        ref,
        {static_cast<global_index_type>(0), static_cast<global_index_type>(171),
         static_cast<global_index_type>(343),
         static_cast<global_index_type>(num_rows)});
    auto other_part =
        gko::share(partition_type::build_from_contiguous(ref, other_ranges));
    auto other_b = dist_vec::create(ref, comm);
    other_b->read_distributed(rhs_data(), other_part);
    auto other_x = dist_vec::create(ref, comm);
    other_x->read_distributed(md_type(gko::dim<2>{num_rows, 1}), other_part);

    ASSERT_THROW(solver->apply(other_b, other_x), gko::DimensionMismatch);
}
