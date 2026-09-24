// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include <ginkgo/core/base/mpi.hpp>

#include <gko_tpl/hypre/environment.hpp>


// Runs the tests with MPI and hypre initialized. hypre is finalized before
// MPI, after the tests have destroyed every solver they created.
int main(int argc, char** argv)
{
    gko::experimental::mpi::environment mpi_env(argc, argv);
    gko::ext::hypre::environment hypre_env;
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
