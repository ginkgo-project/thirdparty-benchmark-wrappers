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


class Pcg : public ::testing::Test {
protected:
    using value_type = double;
    using index_type = gko::int32;
    using pcg_type = gko::ext::hypre::solver::Pcg<value_type, index_type>;
    using csr = gko::matrix::Csr<value_type, index_type>;
    using dense = gko::matrix::Dense<value_type>;

    Pcg()
        : ref{gko::ReferenceExecutor::create()},
          mtx{gko::share(csr::create(ref))},
          b{dense::create(ref, gko::dim<2>{num_rows, 1})},
          x{dense::create(ref, gko::dim<2>{num_rows, 1})}
    {
        mtx->read(laplacian_3d<index_type>(grid_dim));
        for (gko::size_type i = 0; i < num_rows; i++) {
            b->at(i, 0) = std::sin(0.1 * static_cast<double>(i));
        }
        x->fill(0.0);
    }

    double relative_residual(const dense* solution) const
    {
        auto residual = gko::clone(b);
        auto one = gko::initialize<dense>({1.0}, ref);
        auto neg_one = gko::initialize<dense>({-1.0}, ref);
        mtx->apply(neg_one, solution, one, residual);
        auto norm = dense::create(ref, gko::dim<2>{1, 1});
        auto b_norm = dense::create(ref, gko::dim<2>{1, 1});
        residual->compute_norm2(norm);
        b->compute_norm2(b_norm);
        return norm->at(0, 0) / b_norm->at(0, 0);
    }

    static constexpr index_type grid_dim = 8;
    static constexpr gko::size_type num_rows = 512;

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<csr> mtx;
    std::unique_ptr<dense> b;
    std::unique_ptr<dense> x;
};


TEST_F(Pcg, SolvesToTheRequestedTolerance)
{
    auto solver = pcg_type::build()
                      .with_tolerance(1e-10)
                      .with_max_iters(200)
                      .on(ref)
                      ->generate(mtx);

    solver->apply(b, x);

    ASSERT_TRUE(solver->has_converged());
    ASSERT_GT(solver->get_num_iterations(), 0u);
    ASSERT_LT(solver->get_num_iterations(), 50u);
    ASSERT_LE(relative_residual(x.get()), 1e-9);
}


TEST_F(Pcg, MatchesGinkgoCg)
{
    auto x_hypre = gko::clone(x);
    auto x_ginkgo = gko::clone(x);
    auto hypre_solver =
        pcg_type::build().with_tolerance(1e-12).on(ref)->generate(mtx);
    auto ginkgo_solver =
        gko::solver::Cg<value_type>::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(500u),
                           gko::stop::ResidualNorm<value_type>::build()
                               .with_reduction_factor(1e-12))
            .on(ref)
            ->generate(mtx);

    hypre_solver->apply(b, x_hypre);
    ginkgo_solver->apply(b, x_ginkgo);

    GKO_TPL_ASSERT_MTX_NEAR(x_hypre, x_ginkgo, 1e-8);
}


TEST_F(Pcg, ReportsImportAndSetupTimes)
{
    auto solver = pcg_type::build().on(ref)->generate(mtx);

    ASSERT_GT(solver->get_import_time().count(), 0);
    ASSERT_GT(solver->get_setup_time().count(), 0);
}


TEST_F(Pcg, PassesTheParametersToHypre)
{
    // hypre 3.2 exposes getters for what it was configured with; the
    // callback runs after the typed setters, so the handle carries them.
    HYPRE_Solver amg = nullptr;
    auto solver = pcg_type::build()
                      .with_coarsen_type("PMIS")
                      .with_strong_threshold(0.25)
                      .with_max_levels(7)
                      .with_amg_setup([&amg](HYPRE_Solver s) { amg = s; })
                      .on(ref)
                      ->generate(mtx);

    ASSERT_NE(amg, nullptr);
#if HYPRE_RELEASE_NUMBER >= 30200
    HYPRE_Int coarsen_type = 0;
    HYPRE_Real threshold = 0.0;
    HYPRE_Int max_levels = 0;
    ASSERT_EQ(HYPRE_BoomerAMGGetCoarsenType(amg, &coarsen_type), 0);
    ASSERT_EQ(HYPRE_BoomerAMGGetStrongThreshold(amg, &threshold), 0);
    ASSERT_EQ(HYPRE_BoomerAMGGetMaxLevels(amg, &max_levels), 0);
    ASSERT_EQ(coarsen_type, 8);
    ASSERT_EQ(threshold, 0.25);
    ASSERT_EQ(max_levels, 7);
#endif
}


TEST_F(Pcg, ReportsTheHierarchy)
{
    auto solver =
        pcg_type::build().with_coarsen_type("PMIS").on(ref)->generate(mtx);

    // HYPRE_BoomerAMGGetGridHierarchy is public API on every supported hypre,
    // 2.32 included, so this is unconditional.
    const auto& hierarchy = solver->get_hierarchy();
    ASSERT_TRUE(hierarchy.has_value());
    ASSERT_GT(hierarchy->num_levels, 1u);
    ASSERT_EQ(hierarchy->rows_per_level.size(), hierarchy->num_levels);
    // The finest level is the matrix itself, and each level is coarser than
    // the one above it.
    ASSERT_EQ(hierarchy->rows_per_level.front(), num_rows);
    for (gko::size_type l = 1; l < hierarchy->num_levels; l++) {
        ASSERT_LT(hierarchy->rows_per_level[l],
                  hierarchy->rows_per_level[l - 1]);
    }
}


TEST_F(Pcg, DifferentCoarseningsAgreeOnTheSolution)
{
    auto pmis = pcg_type::build()
                    .with_tolerance(1e-10)
                    .with_coarsen_type("PMIS")
                    .on(ref)
                    ->generate(mtx);
    auto falgout = pcg_type::build()
                       .with_tolerance(1e-10)
                       .with_coarsen_type("Falgout")
                       .on(ref)
                       ->generate(mtx);
    auto x_pmis = gko::clone(x);
    auto x_falgout = gko::clone(x);

    pmis->apply(b, x_pmis);
    falgout->apply(b, x_falgout);

    ASSERT_TRUE(pmis->has_converged());
    ASSERT_TRUE(falgout->has_converged());
    GKO_TPL_ASSERT_MTX_NEAR(x_pmis, x_falgout, 1e-6);
}


TEST_F(Pcg, RejectsAnUnknownCoarseningName)
{
    ASSERT_THROW(
        pcg_type::build().with_coarsen_type("PIMS").on(ref)->generate(mtx),
        gko::InvalidStateError);
}


TEST_F(Pcg, RepeatedApplyGivesTheSameAnswer)
{
    // apply re-points the generation-time vectors rather than rebuilding
    // them; a pointer left over from the first apply would make the second
    // solve read or write the first one's memory. Checks the two solves
    // agree and the first output is undisturbed by the second.
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
    GKO_TPL_ASSERT_MTX_NEAR(x_first, x_first_before, 0.0);
    GKO_TPL_ASSERT_MTX_NEAR(x_second, x_first, 1e-14);
}


TEST_F(Pcg, RejectsMultipleRightHandSides)
{
    auto solver = pcg_type::build().on(ref)->generate(mtx);
    auto wide_b = dense::create(ref, gko::dim<2>{num_rows, 2});
    auto wide_x = dense::create(ref, gko::dim<2>{num_rows, 2});
    wide_b->fill(1.0);
    wide_x->fill(0.0);

    ASSERT_THROW(solver->apply(wide_b, wide_x), gko::DimensionMismatch);
}


TEST_F(Pcg, UsesTheInputAsInitialGuess)
{
    auto solver = pcg_type::build()
                      .with_tolerance(1e-10)
                      .with_max_iters(200)
                      .on(ref)
                      ->generate(mtx);
    auto x_cold = gko::clone(x);
    solver->apply(b, x_cold);
    const auto cold_iterations = solver->get_num_iterations();

    // Starting from the solution must take fewer iterations than starting
    // from zero.
    auto x_warm = gko::clone(x_cold);
    solver->apply(b, x_warm);

    ASSERT_LT(solver->get_num_iterations(), cold_iterations);
}
