// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cmath>
#include <memory>
#include <set>
#include <string>

#include <gtest/gtest.h>

#include <petscsys.h>

#include <ginkgo/ginkgo.hpp>

#include <gko_tpl/petsc/ksp.hpp>

#include "common/test/assertions.hpp"
#include "petsc/test/laplacian.hpp"


// Names of the options in PETSc's database that carry a Ksp instance prefix
// and that nothing has queried so far.
std::set<std::string> unused_ksp_options()
{
    PetscInt count = 0;
    char** names = nullptr;
    char** values = nullptr;
    EXPECT_EQ(PetscOptionsLeftGet(nullptr, &count, &names, &values),
              PETSC_SUCCESS);
    std::set<std::string> result;
    for (PetscInt i = 0; i < count; i++) {
        const std::string name{names[i]};
        if (name.find("gko_ksp_") != std::string::npos) {
            result.insert(name);
        }
    }
    EXPECT_EQ(PetscOptionsLeftRestore(nullptr, &count, &names, &values),
              PETSC_SUCCESS);
    return result;
}


class Ksp : public ::testing::Test {
protected:
    using value_type = double;
    using index_type = gko::int32;
    using ksp_type = gko::ext::petsc::solver::Ksp<value_type, index_type>;
    using csr = gko::matrix::Csr<value_type, index_type>;
    using dense = gko::matrix::Dense<value_type>;
    using cg = gko::solver::Cg<value_type>;

    Ksp()
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

    double b_norm() const
    {
        auto norm = dense::create(ref, gko::dim<2>{1, 1});
        b->compute_norm2(norm);
        return norm->at(0, 0);
    }

    // ||b - A * solution||_2 / ||b||_2
    double relative_residual(const dense* solution) const
    {
        auto residual = gko::clone(b);
        auto one = gko::initialize<dense>({1.0}, ref);
        auto neg_one = gko::initialize<dense>({-1.0}, ref);
        mtx->apply(neg_one, solution, one, residual);
        auto norm = dense::create(ref, gko::dim<2>{1, 1});
        residual->compute_norm2(norm);
        return norm->at(0, 0) / b_norm();
    }

    static constexpr index_type grid_dim = 10;
    static constexpr gko::size_type num_rows = 1000;
    // CG + BoomerAMG stopping like Ginkgo's ResidualNorm criterion: on the
    // unpreconditioned residual relative to the initial residual, which is
    // ||b|| for the zero initial guess.
    static constexpr const char* cg_boomeramg =
        "-ksp_type cg -ksp_norm_type unpreconditioned -ksp_rtol 1e-8 "
        "-ksp_max_it 200 -pc_type hypre -pc_hypre_type boomeramg";

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<csr> mtx;
    std::unique_ptr<dense> b;
    std::unique_ptr<dense> x;
};


TEST_F(Ksp, ReportsNoSolveBeforeApply)
{
    auto solver =
        ksp_type::build().with_options(cg_boomeramg).on(ref)->generate(mtx);

    ASSERT_EQ(solver->get_num_iterations(), 0u);
    ASSERT_EQ(solver->get_residual_norm(), 0.0);
    ASSERT_EQ(solver->get_converged_reason(), 0);
    ASSERT_FALSE(solver->has_converged());
}


TEST_F(Ksp, ReportsImportAndSetupTimes)
{
    auto solver =
        ksp_type::build().with_options(cg_boomeramg).on(ref)->generate(mtx);
    // A copy shares the PETSc objects, and with them the generation times.
    ksp_type copy{*solver};

    ASSERT_GT(solver->get_import_time().count(), 0);
    ASSERT_GT(solver->get_setup_time().count(), 0);
    ASSERT_EQ(copy.get_import_time(), solver->get_import_time());
    ASSERT_EQ(copy.get_setup_time(), solver->get_setup_time());
}


TEST_F(Ksp, CgWithBoomerAmgConvergesAndReportsTheSolve)
{
    auto solver =
        ksp_type::build().with_options(cg_boomeramg).on(ref)->generate(mtx);

    solver->apply(b, x);

    ASSERT_TRUE(solver->has_converged());
    ASSERT_GT(solver->get_converged_reason(), 0);
    ASSERT_GT(solver->get_num_iterations(), 0u);
    ASSERT_LT(solver->get_num_iterations(), 30u);
    const auto true_residual = relative_residual(x.get());
    ASSERT_LT(true_residual, 2e-8);
    // PETSc's norm is the unpreconditioned recurrence residual, which agrees
    // with the true residual up to roundoff.
    ASSERT_NEAR(solver->get_residual_norm() / b_norm(), true_residual, 1e-10);
}


TEST_F(Ksp, CgWithBoomerAmgMatchesGinkgoCg)
{
    auto solver =
        ksp_type::build().with_options(cg_boomeramg).on(ref)->generate(mtx);
    auto ginkgo_solver =
        cg::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(1000u),
                           gko::stop::ResidualNorm<value_type>::build()
                               .with_reduction_factor(1e-12))
            .on(ref)
            ->generate(mtx);
    auto x_ginkgo = gko::clone(x);

    solver->apply(b, x);
    ginkgo_solver->apply(b, x_ginkgo);

    // Both are within about cond(A) * 1e-8 (cond(A) is about 50) of the
    // exact solution.
    GKO_TPL_ASSERT_MTX_NEAR(x, x_ginkgo, 1e-6);
}


TEST_F(Ksp, PreonlyBoomerAmgPreconditionsGinkgoCg)
{
    auto amg = gko::share(ksp_type::build()
                              .with_options("-ksp_type preonly -pc_type hypre "
                                            "-pc_hypre_type boomeramg")
                              .on(ref)
                              ->generate(mtx));
    auto solver =
        cg::build()
            .with_generated_preconditioner(amg)
            .with_criteria(gko::stop::Iteration::build().with_max_iters(200u),
                           gko::stop::ResidualNorm<value_type>::build()
                               .with_reduction_factor(1e-8))
            .on(ref)
            ->generate(mtx);
    auto logger = gko::share(gko::log::Convergence<value_type>::create());
    solver->add_logger(logger);

    solver->apply(b, x);

    ASSERT_TRUE(logger->has_converged());
    ASSERT_LT(logger->get_num_iterations(), 30u);
    ASSERT_LT(relative_residual(x.get()), 2e-8);
}


TEST_F(Ksp, InstancesKeepTheirOwnOptions)
{
    // Without per-instance prefixes, the second instance would read the
    // first instance's -ksp_max_it 1 from PETSc's global options database.
    auto capped =
        ksp_type::build()
            .with_options("-ksp_type cg -ksp_max_it 1 -pc_type jacobi")
            .on(ref)
            ->generate(mtx);
    auto uncapped =
        ksp_type::build()
            .with_options("-ksp_type cg -ksp_rtol 1e-8 -pc_type jacobi")
            .on(ref)
            ->generate(mtx);
    auto x_capped = gko::clone(x);

    capped->apply(b, x_capped);
    uncapped->apply(b, x);

    ASSERT_EQ(capped->get_num_iterations(), 1u);
    ASSERT_FALSE(capped->has_converged());
    ASSERT_TRUE(uncapped->has_converged());
    ASSERT_GT(uncapped->get_num_iterations(), 1u);
}


TEST_F(Ksp, AdvancedApplyScalesAndAddsTheSolution)
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

    GKO_TPL_ASSERT_MTX_NEAR(x, expected, 1e-14);
}


TEST_F(Ksp, ThrowsForMultipleRightHandSides)
{
    auto solver =
        ksp_type::build().with_options(cg_boomeramg).on(ref)->generate(mtx);
    auto b2 = dense::create(ref, gko::dim<2>{num_rows, 2});
    auto x2 = dense::create(ref, gko::dim<2>{num_rows, 2});

    ASSERT_THROW(solver->apply(b2, x2), gko::DimensionMismatch);
}


TEST_F(Ksp, ExamplePresetsLeaveNoUnusedOptions)
{
    // The presets of examples/distributed-multigrid-preconditioned-solver
    // with its default --rel-residual, --max-iters, --strength-threshold and
    // --max-levels. A misspelled option name stays unused and fails here.
    const std::string hypre =
        "-ksp_type cg -ksp_norm_type unpreconditioned -ksp_rtol 1e-08 "
        "-ksp_max_it 1000 -pc_type hypre -pc_hypre_type boomeramg";
    const std::string hypre_pmis = hypre +
                                   " -pc_hypre_boomeramg_coarsen_type PMIS"
                                   " -pc_hypre_boomeramg_strong_threshold 0.25"
                                   " -pc_hypre_boomeramg_relax_type_all Jacobi"
                                   " -pc_hypre_boomeramg_relax_weight_all 0.9"
                                   " -pc_hypre_boomeramg_grid_sweeps_all 2"
                                   " -pc_hypre_boomeramg_max_levels 11";
    const auto unused_before = unused_ksp_options();

    for (const auto& options : {hypre, hypre_pmis}) {
        auto solver =
            ksp_type::build().with_options(options).on(ref)->generate(mtx);
        auto x_run = gko::clone(x);
        solver->apply(b, x_run);
        ASSERT_TRUE(solver->has_converged()) << options;
    }

    ASSERT_EQ(unused_ksp_options(), unused_before);
}


TEST_F(Ksp, ParseConfigSetsOptions)
{
    auto config_map = ksp_type::get_config_map();
    auto reg = gko::config::registry{config_map};
    gko::config::pnode::map_type conf_map;
    conf_map["type"] = gko::config::pnode{"ext::petsc::solver::Ksp"};
    conf_map["options"] = gko::config::pnode{"-ksp_type cg -pc_type jacobi"};
    auto conf = gko::config::pnode{conf_map};

    auto params = ksp_type::parse(conf, reg);

    ASSERT_EQ(params.options, "-ksp_type cg -pc_type jacobi");
}


TEST_F(Ksp, ParseConfigThrowsOnUnknownKey)
{
    auto config_map = ksp_type::get_config_map();
    auto reg = gko::config::registry{config_map};
    gko::config::pnode::map_type conf_map;
    conf_map["type"] = gko::config::pnode{"ext::petsc::solver::Ksp"};
    conf_map["invalid_key"] = gko::config::pnode{42};
    auto conf = gko::config::pnode{conf_map};

    ASSERT_THROW(ksp_type::parse(conf, reg), gko::InvalidStateError);
}
