// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include <ginkgo/ginkgo.hpp>

#include <gko_tpl/hypre/pcg.hpp>


using pcg_type = gko::ext::hypre::solver::Pcg<double, gko::int32>;


TEST(Config, ParsesTheDocumentedKeys)
{
    auto config_map = pcg_type::get_config_map();
    ASSERT_EQ(config_map.count("ext::hypre::solver::Pcg"), 1u);
    auto reg = gko::config::registry{config_map};
    gko::config::pnode::map_type conf_map;
    conf_map["type"] = gko::config::pnode{"ext::hypre::solver::Pcg"};
    conf_map["tolerance"] = gko::config::pnode{1e-9};
    conf_map["max_iters"] = gko::config::pnode{42};
    conf_map["coarsen_type"] = gko::config::pnode{"PMIS"};
    conf_map["strong_threshold"] = gko::config::pnode{0.25};

    auto params = pcg_type::parse(gko::config::pnode{conf_map}, reg);

    ASSERT_EQ(params.tolerance, 1e-9);
    ASSERT_EQ(params.max_iters, 42);
    ASSERT_EQ(params.coarsen_type, "PMIS");
    ASSERT_EQ(params.strong_threshold, 0.25);
}


TEST(Config, ParsesEveryScalarParameter)
{
    auto reg = gko::config::registry{pcg_type::get_config_map()};
    gko::config::pnode::map_type conf_map;
    conf_map["type"] = gko::config::pnode{"ext::hypre::solver::Pcg"};
    conf_map["tolerance"] = gko::config::pnode{1e-10};
    conf_map["max_iters"] = gko::config::pnode{200};
    conf_map["two_norm"] = gko::config::pnode{false};
    conf_map["coarsen_type"] = gko::config::pnode{"PMIS"};
    conf_map["interp_type"] = gko::config::pnode{"ext+i"};
    conf_map["relax_type"] = gko::config::pnode{"l1-Jacobi"};
    conf_map["relax_weight"] = gko::config::pnode{0.8};
    conf_map["num_sweeps"] = gko::config::pnode{2};
    conf_map["max_levels"] = gko::config::pnode{10};
    conf_map["max_coarse_size"] = gko::config::pnode{9};
    conf_map["strong_threshold"] = gko::config::pnode{0.5};
    conf_map["print_statistics"] = gko::config::pnode{true};

    auto params = pcg_type::parse(gko::config::pnode{conf_map}, reg);

    ASSERT_EQ(params.tolerance, 1e-10);
    ASSERT_EQ(params.max_iters, 200);
    ASSERT_EQ(params.two_norm, false);
    ASSERT_EQ(params.coarsen_type, "PMIS");
    ASSERT_EQ(params.interp_type, "ext+i");
    ASSERT_EQ(params.relax_type, "l1-Jacobi");
    ASSERT_EQ(params.relax_weight, 0.8);
    ASSERT_EQ(params.num_sweeps, 2);
    ASSERT_EQ(params.max_levels, 10);
    ASSERT_EQ(params.max_coarse_size, 9);
    ASSERT_EQ(params.strong_threshold, 0.5);
    ASSERT_EQ(params.print_statistics, true);
}


// A key absent from the configuration must leave its parameter at the
// sentinel default ("leave hypre's own default alone"), not a materialized
// stand-in value.
TEST(Config, OmittedKeysKeepTheirSentinelDefaults)
{
    auto reg = gko::config::registry{pcg_type::get_config_map()};
    gko::config::pnode::map_type conf_map;
    conf_map["type"] = gko::config::pnode{"ext::hypre::solver::Pcg"};

    auto params = pcg_type::parse(gko::config::pnode{conf_map}, reg);
    auto defaults = pcg_type::build();

    ASSERT_EQ(params.tolerance, defaults.tolerance);
    ASSERT_EQ(params.max_iters, defaults.max_iters);
    ASSERT_EQ(params.two_norm, defaults.two_norm);
    ASSERT_EQ(params.coarsen_type, "");
    ASSERT_EQ(params.interp_type, "");
    ASSERT_EQ(params.relax_type, "");
    ASSERT_EQ(params.relax_weight, -1.0);
    ASSERT_EQ(params.num_sweeps, 0);
    ASSERT_EQ(params.max_levels, 0);
    ASSERT_EQ(params.max_coarse_size, 0);
    ASSERT_EQ(params.strong_threshold, -1.0);
    ASSERT_EQ(params.print_statistics, defaults.print_statistics);
}


TEST(Config, RejectsAnUnknownKey)
{
    auto reg = gko::config::registry{pcg_type::get_config_map()};
    gko::config::pnode::map_type conf_map;
    conf_map["type"] = gko::config::pnode{"ext::hypre::solver::Pcg"};
    conf_map["coarsening"] = gko::config::pnode{"PMIS"};

    ASSERT_THROW(pcg_type::parse(gko::config::pnode{conf_map}, reg),
                 gko::InvalidStateError);
}
